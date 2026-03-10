import json
import re
from typing import Any

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.model_handler.utils import convert_to_function_call
from overrides import override


class Phi4ReasonFCHandler(OSSHandler):
    """
    Function-calling handler for custom Phi-4 reasoning FC prompting.
    """

    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        dtype="bfloat16",
        **kwargs,
    ) -> None:
        # Preserve old behavior: fixed temperature for this reasoning handler.
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_name_huggingface = model_name.replace("-FC", "")
        self.is_fc_model = True

    @override
    def _format_prompt(self, messages, function):
        # sanity check
        system_messages = [msg for msg in messages if msg["role"] == "system"]
        assert 0 <= len(system_messages) <= 1

        # set the system message
        system_message = (
            "You are a reasoning language model that can reach precise answers through "
            "careful reasoning and tool use when needed. \n\nStructure Rules:\n1. All "
            "reasoning goes between <think> and </think> (thinking block). \n2. Whenever "
            "a tool would improve your answer, invoke it using <tool_call>...</tool_call> "
            "instead of relying solely on memory.\n3. Issue one or multiple tool calls "
            "<tool_call></tool_call>...<tool_call></tool_call> at a time; when tool calls "
            "can't be called in parallel (you need the result of one to call the other) you can sequentially interleave throughout the "
            "reasoning process (using the result of one to guide the call of the "
            "other). \n4. After each tool call or calls, the results of each tool call "
            "will be provided in the <tool_response></tool_response>..."
            "<tool_response></tool_response> tags.\n5. Stop the generation only after "
            "reaching the final answer.\n\nYou can utilize the tools as many times as "
            "required. For example, <think> reasoning here  </think> <tool_call> tool "
            "call here </tool_call> <tool_response> output of tool call </tool_response> "
            "<think> reasoning process here </think> final answer here (or more tool "
            "calls).\n\n# Format for tool calls: <tool_call>{\"name\": "
            "<function-name>,\"arguments\": <dict-of-arguments>}</tool_call>\n\n"
            "# Available Tools\nYou are provided with function signatures within "
            "<tool></tool> tags.\n"
        )
        system_message_end = (
            "\n\nYou SHOULD NOT include any other text in the response or ask the user "
            "for information, use reasonable assumptions for function calls. Only if the "
            "available functions are not relevant or there are mandatory missing params "
            "that you should output text and point it out.\nAt each turn, you should try "
            "your best to complete the tasks requested by the user within the current "
            "turn. Continue to output functions to call until you have fulfilled the "
            "user's request to the best of your ability. Once you have output a text "
            "with no function calls (could be empty or with any text), the system will "
            "consider the current turn complete and proceed to the next turn or task."
        )
        if messages and messages[0]["role"] == "system":
            system_message = messages[0]["content"]
            messages = messages[1:]

        # extract the tool contents
        tool_contents = json.dumps([func for func in function])

        # format the rest of the prompt
        formatted_prompt = (
            f"<|im_start|>system<|im_sep|>{system_message}<tool>{tool_contents}</tool>"
            f"{system_message_end}<|im_end|>"
        )

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "tool":
                formatted_prompt += (
                    f"<|im_start|>user<|im_sep|><tool_response>{content}</tool_response>"
                    "<|im_end|>"
                )
            elif role == "assistant":
                assistant_content = content if isinstance(content, str) else ""
                reasoning_content = msg.get("reasoning_content", "")

                # Preserve previously generated reasoning for stable multi-step behavior.
                if reasoning_content:
                    assistant_content = (
                        "<think>"
                        + reasoning_content.strip("\n")
                        + "</think>"
                        + assistant_content.lstrip("\n")
                    )

                # Preserve previous tool calls in chat history.
                for idx, tool_call in enumerate(msg.get("tool_calls", [])):
                    if isinstance(tool_call, dict) and "function" in tool_call:
                        tool_call = tool_call["function"]
                    if not isinstance(tool_call, dict):
                        continue

                    name = tool_call.get("name", "")
                    arguments = tool_call.get("arguments", {})
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments)

                    assistant_content += (
                        '<tool_call>{"name": "'
                        + name
                        + '", "arguments": '
                        + arguments
                        + "}</tool_call>"
                    )

                formatted_prompt += (
                    f"<|im_start|>{role}<|im_sep|>{assistant_content}<|im_end|>"
                )
            else:
                formatted_prompt += f"<|im_start|>{role}<|im_sep|>{content}<|im_end|>"

        # provide the generation prompt token
        formatted_prompt += "<|im_start|>assistant<|im_sep|><think>"
        return formatted_prompt

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        # Model response is of the form:
        # "<tool_call>\n{\"name\": \"spotify.play\", \"arguments\": {\"artist\": \"Taylor Swift\", \"duration\": 20}}\n</tool_call>\n<tool_call>\n{\"name\": \"spotify.play\", \"arguments\": {\"artist\": \"Maroon 5\", \"duration\": 15}}\n</tool_call>"
        tool_calls = self._extract_tool_calls(result)
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        return [
            {call["name"]: {k: v for k, v in call["arguments"].items()}}
            for call in tool_calls
        ]

    @override
    def decode_execute(self, result, has_tool_call_tag):
        tool_calls = self._extract_tool_calls(result)
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        decoded_result = []
        for item in tool_calls:
            if type(item) == str:
                item = eval(item)
            decoded_result.append({item["name"]: item["arguments"]})
        return convert_to_function_call(decoded_result)

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        raw_text = api_response.choices[0].text

        # Keep the old behavior: strip visible thinking from text response.
        reasoning_content = ""
        cleaned_response = raw_text

        if "</think>" in raw_text:
            parts = raw_text.split("</think>")
            reasoning_content = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
            cleaned_response = parts[-1].lstrip("\n")
        elif "<think>" in raw_text:
            # Graceful fallback when the model misses a closing </think>.
            reasoning_content = raw_text.split("<think>")[-1].strip("\n")
            cleaned_response = ""

        extracted_tool_calls = self._extract_tool_calls(raw_text)

        # If tool calls are present, store them structurally; otherwise keep cleaned text.
        if len(extracted_tool_calls) > 0:
            # model_responses = [
            #     {item["name"]: item["arguments"]} for item in extracted_tool_calls
            # ]
            model_responses_message_for_chat_history: dict[str, Any] = {
                "role": "assistant",
                "content": "",
                "tool_calls": extracted_tool_calls,
                "reasoning_content": reasoning_content,
            }
        else:
            model_responses = cleaned_response
            model_responses_message_for_chat_history = {
                "role": "assistant",
                "content": cleaned_response,
                "reasoning_content": reasoning_content,
            }

        return {
            "model_responses": cleaned_response,
            "reasoning_content": reasoning_content,
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }


    @staticmethod
    def _extract_tool_calls(input_string: str) -> list[Any]:
        if "<think>" in input_string:
            input_string = input_string.split("<think>")[-1]
        if "</think>" in input_string:
            input_string = input_string.split("</think>")[-1]

        # Match: <tool_call>...</tool_call>
        pattern = r"<tool_call>(.*?)</tool_call>"
        matches = re.findall(pattern, input_string, re.DOTALL)

        # Tolerate missing closing tag.
        if not matches:
            pattern = r"<tool_call>(.*?)(?:</tool_call>)?$"
            matches = re.findall(pattern, input_string, re.DOTALL)

        result: list[Any] = []
        for match in matches:
            # Tolerate comma-separated objects without surrounding list.
            if not match.startswith("[") and not match.endswith("]"):
                match = "[" + match + "]"

            try:
                parsed = json.loads(match)
            except json.JSONDecodeError:
                parsed = match

            if type(parsed) is list:
                for item in parsed:
                    if type(item) is str:
                        item = eval(item)
                    result.append(item)
            else:
                result.append(parsed)

        return result

    # @staticmethod
    # def _is_tool_call_response_format(parsed_calls: list[Any]) -> bool:
    #     if type(parsed_calls) != list:
    #         return False

    #     for item in parsed_calls:
    #         if type(item) != dict:
    #             return False
    #         if "name" not in item:
    #             return False
    #         if "arguments" not in item:
    #             return False
    #         if len(item) != 2:
    #             return False

    #     return True

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]

        # FC models use its own system prompt, so no need to add any message

        return {"message": [], "function": functions}

    @override
    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data
