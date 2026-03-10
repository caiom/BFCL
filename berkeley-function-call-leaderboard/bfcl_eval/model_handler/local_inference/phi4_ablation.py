import json
from typing import Any

from bfcl_eval.model_handler.local_inference.phi4_reason_fc import Phi4ReasonFCHandler
from bfcl_eval.model_handler.local_inference.phi4_special import Phi4SpecialHandler
from bfcl_eval.model_handler.local_inference.prompts import prompts
from overrides import override


def _extract_system_prompt(messages: list[dict], default_system: str) -> tuple[str, list[dict]]:
    if messages and messages[0]["role"] == "system":
        return messages[0]["content"], messages[1:]
    return default_system, messages


def _serialize_reason_history(formatted_prompt: str, messages: list[dict]) -> str:
    """
    Serialize chat history with the same message-formatting logic as Phi4ReasonFC.
    This is used by prompt ablations that only change the system prompt content.
    """
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

            if reasoning_content:
                assistant_content = (
                    "<think>"
                    + reasoning_content.strip("\n")
                    + "</think>"
                    + assistant_content.lstrip("\n")
                )

            for tool_call in msg.get("tool_calls", []):
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

    return formatted_prompt


def _build_system_prompt_from_template(prompt_key: str, function: list[dict]) -> str:
    return prompts[prompt_key].replace(
        "{tools}",
        "\n".join(json.dumps(t) for t in function),
    )


def _build_special_system_prompt(function: list[dict]) -> str:
    return _build_system_prompt_from_template("bfcl_simple_ll", function)


def _build_special_system_prompt_no_answer(function: list[dict]) -> str:
    prompt_template = prompts["bfcl_simple_ll"]
    lines = []
    for line in prompt_template.splitlines():
        stripped = line.strip()
        if stripped.startswith("4. Once the task is complete"):
            lines.append(
                "4. Once the task is complete, and there are no more tools to call, provide the final answer in plain text."
            )
            continue
        if "<answer>" in line and "final turn" in line:
            lines.append(
                "You can utilize the thinking and tool call loop as many times as required; in the final turn provide a plain-text final answer instead of a tool call."
            )
            continue
        lines.append(line)

    return "\n".join(lines).replace(
        "{tools}",
        "\n".join(json.dumps(t) for t in function),
    )


class Phi4ReasonFCNoThinkPrefixHandler(Phi4ReasonFCHandler):
    """
    Ablation #1:
    Keep Phi4ReasonFC behavior, but stop injecting an extra '<think>' token into
    assistant history, which can create malformed context like '</think><think><tool_call>'.
    """

    @override
    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data


class Phi4ReasonFCNoThinkPrefixSpecialPromptHandler(Phi4ReasonFCNoThinkPrefixHandler):
    """
    Ablation #2:
    Use the Phi4Special prompt template while keeping Phi4ReasonFC parsing/extraction.
    """

    @override
    def _format_prompt(self, messages, function):
        return Phi4SpecialHandler._format_prompt(self, messages, function)


class Phi4ReasonFCNoThinkPrefixSpecialParseHandler(Phi4ReasonFCNoThinkPrefixHandler):
    """
    Ablation #3:
    Keep Phi4ReasonFC prompting, but use Phi4Special response parsing/extraction.
    """

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        return Phi4SpecialHandler._parse_query_response_prompting(self, api_response)

    @staticmethod
    @override
    def _extract_tool_calls(input_string: str) -> list[Any]:
        return Phi4SpecialHandler._extract_tool_calls(input_string)


class Phi4SpecialNoThinkPrefixHandler(Phi4SpecialHandler):
    """
    Ablation #4:
    Keep Phi4Special behavior, but remove extra '<think>' prefix injection.
    """

    @override
    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data


class Phi4SpecialNoThinkPrefixPromptLLLHandler(Phi4SpecialNoThinkPrefixHandler):
    """
    Prompt ablation:
    Keep Phi4Special_NoThinkPrefix behavior with bfcl_simple_lll system template.
    """

    system_prompt_key = "bfcl_simple_lll"


class Phi4SpecialNoThinkPrefixPromptLLLLHandler(Phi4SpecialNoThinkPrefixHandler):
    """
    Prompt ablation:
    Keep Phi4Special_NoThinkPrefix behavior with bfcl_simple_llll system template.
    """

    system_prompt_key = "bfcl_simple_llll"


class Phi4SpecialNoThinkPrefixRobustParseHandler(Phi4SpecialNoThinkPrefixHandler):
    """
    Hybrid candidate:
    Keep the best-performing Phi4Special_NoThinkPrefix prompting/history behavior,
    but fall back to Phi4ReasonFC extraction when tool-call tags are malformed.
    """

    @staticmethod
    @override
    def _extract_tool_calls(input_string) -> list[Any]:
        tool_calls = Phi4SpecialHandler._extract_tool_calls(input_string)
        if tool_calls:
            return tool_calls

        # Fallback parser is more tolerant to partial/malformed tool-call spans.
        return Phi4ReasonFCHandler._extract_tool_calls(input_string)


class Phi4ReasonFCNoThinkPrefixSpecialSystemOnlyHandler(
    Phi4ReasonFCNoThinkPrefixHandler
):
    """
    Prompt ablation:
    Use only the Phi4Special system prompt/template text while keeping
    Phi4ReasonFC-style history serialization and parsing behavior.
    """

    @override
    def _format_prompt(self, messages, function):
        system_prompt = _build_special_system_prompt(function)
        system_prompt, messages = _extract_system_prompt(messages, system_prompt)

        formatted_prompt = f"<|im_start|>system<|im_sep|>{system_prompt}<|im_end|>"
        formatted_prompt = _serialize_reason_history(formatted_prompt, messages)
        formatted_prompt += "<|im_start|>assistant<|im_sep|><think>"
        return formatted_prompt


class Phi4ReasonFCNoThinkPrefixSpecialSystemOnlyPromptLLLLHandler(
    Phi4ReasonFCNoThinkPrefixHandler
):
    """
    Prompt ablation:
    Keep SpecialSystemOnly logic, but use bfcl_simple_llll as the system template.
    """

    @override
    def _format_prompt(self, messages, function):
        system_prompt = _build_system_prompt_from_template("bfcl_simple_llll", function)
        system_prompt, messages = _extract_system_prompt(messages, system_prompt)

        formatted_prompt = f"<|im_start|>system<|im_sep|>{system_prompt}<|im_end|>"
        formatted_prompt = _serialize_reason_history(formatted_prompt, messages)
        formatted_prompt += "<|im_start|>assistant<|im_sep|><think>"
        return formatted_prompt


class Phi4ReasonFCNoThinkPrefixSpecialSystemOnlyPromptParallelHandler(
    Phi4ReasonFCNoThinkPrefixHandler
):
    """
    Prompt ablation:
    Keep SpecialSystemOnly logic, but use bfcl_simple_parallel as the system template.
    """

    @override
    def _format_prompt(self, messages, function):
        system_prompt = _build_system_prompt_from_template("bfcl_simple_parallel", function)
        system_prompt, messages = _extract_system_prompt(messages, system_prompt)

        formatted_prompt = f"<|im_start|>system<|im_sep|>{system_prompt}<|im_end|>"
        formatted_prompt = _serialize_reason_history(formatted_prompt, messages)
        formatted_prompt += "<|im_start|>assistant<|im_sep|><think>"
        return formatted_prompt


class Phi4ReasonFCNoThinkPrefixSpecialSystemNoAnswerHandler(
    Phi4ReasonFCNoThinkPrefixHandler
):
    """
    Prompt ablation:
    Same as SpecialSystemOnly, but removes `<answer>`-tag requirements from the
    system prompt to isolate that specific instruction.
    """

    @override
    def _format_prompt(self, messages, function):
        system_prompt = _build_special_system_prompt_no_answer(function)
        system_prompt, messages = _extract_system_prompt(messages, system_prompt)

        formatted_prompt = f"<|im_start|>system<|im_sep|>{system_prompt}<|im_end|>"
        formatted_prompt = _serialize_reason_history(formatted_prompt, messages)
        formatted_prompt += "<|im_start|>assistant<|im_sep|><think>"
        return formatted_prompt


class Phi4ReasonFCNoThinkPrefixPluralToolsTagHandler(
    Phi4ReasonFCNoThinkPrefixHandler
):
    """
    Prompt ablation:
    Keep Phi4ReasonFC prompt as-is, except switch tool-signature wrapper tags
    from `<tool>...</tool>` to `<tools>...</tools>`.
    """

    @override
    def _format_prompt(self, messages, function):
        formatted_prompt = super()._format_prompt(messages, function)
        formatted_prompt = formatted_prompt.replace("<tool>", "<tools>", 1)
        formatted_prompt = formatted_prompt.replace("</tool>", "</tools>", 1)
        return formatted_prompt
