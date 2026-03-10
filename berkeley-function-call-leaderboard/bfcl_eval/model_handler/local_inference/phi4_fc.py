import json
import re
from typing import Any

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.model_handler.utils import convert_to_function_call
from overrides import override


class Phi4FCHandler(OSSHandler):
    """
    Function-calling handler for custom Phi-4 FC prompting.
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
            "You are a helpful assistant that can answer questions and provide "
            "information based on the provided context.\n\n# Tools\n\nYou may call one "
            "or more functions to assist with the user query.\nIf none of the functions "
            "can be used, point it out. If no function is relevant to the given task, "
            "point it out. If the given question lacks the parameters required by the "
            "function, also point it out. You should only return the function calls in "
            "your response.\n\nYou are provided with function signatures within "
            "<tool></tool> tags.\n"
        )
        system_message_end = (
            "\n\nIf you decide to invoke any of the function(s), for each function call, "
            "return a json object with function name and arguments within "
            "<tool_call></tool_call> tags.\nExample: <tool_call>{\"name\": "
            "<function-name>, \"arguments\": <arguments-dict>}</tool_call>\nYou SHOULD "
            "NOT include any other text in the response.\nAt each turn, you should try "
            "your best to complete the tasks requested by the user within the current "
            "turn. Continue to output functions to call until you have fulfilled the "
            "user's request to the best of your ability. Once you have no more functions "
            "to call, the system will consider the current turn complete and proceed to "
            "the next turn or task."
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
            else:
                formatted_prompt += f"<|im_start|>{role}<|im_sep|>{content}<|im_end|>"

        # provide the generation prompt token
        formatted_prompt += "<|im_start|>assistant<|im_sep|><think></think>"
        return formatted_prompt

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        del language, has_tool_call_tag
        # The input is already a list of dictionaries.
        if type(result) != list or any(type(item) != dict for item in result):
            return []
        return result

    @override
    def decode_execute(self, result, has_tool_call_tag):
        del has_tool_call_tag
        if type(result) != list or any(type(item) != dict for item in result):
            return []
        return convert_to_function_call(result)

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        # FC models use their own system prompt.
        return {"message": [], "function": test_entry["function"]}

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        model_responses_message_for_chat_history = api_response.choices[0].text
        model_responses = api_response.choices[0].text

        extracted_tool_calls = self._extract_tool_calls(model_responses)
        if (
            self._is_tool_call_response_format(extracted_tool_calls)
            and len(extracted_tool_calls) > 0
        ):
            model_responses = [
                {item["name"]: item["arguments"]} for item in extracted_tool_calls
            ]

        return {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    @override
    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            {
                "role": "assistant",
                "content": model_response_data["model_responses_message_for_chat_history"],
            }
        )
        return inference_data

    @staticmethod
    def _extract_tool_calls(input_string: str) -> list[Any]:
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

    @staticmethod
    def _is_tool_call_response_format(parsed_calls: list[Any]) -> bool:
        if type(parsed_calls) != list:
            return False

        for item in parsed_calls:
            if type(item) != dict:
                return False
            if "name" not in item:
                return False
            if "arguments" not in item:
                return False
            if len(item) != 2:
                return False

        return True
