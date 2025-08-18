from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from overrides import override


class Phi7bHandler(OSSHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)

    @override
    def decode_ast(self, result, language="Python"):
        result = result.strip()
        if result.startswith("```json"):
            result = result[len("```json") :]
        if result.startswith("```python"):
            result = result[len("```python") :]
        return super().decode_ast(result, language)

    @override
    def decode_execute(self, result):
        if result.startswith("```json"):
            result = result[len("```json") :]
        if result.startswith("```python"):
            result = result[len("```python") :]
        return super().decode_execute(result)

    @override
    def _format_prompt(self, messages, function):
        formatted_prompt = ""

        # Phi-4-mini
        """
        "chat_template": "{% for message in messages %}{% if message['role'] == 'system' and 'tools' in message and message['tools'] is not none %}{{ '<|' + message['role'] + '|>' + message['content'] + '<|tool|>' + message['tools'] + '<|/tool|>' + '<|end|>' }}{% else %}{{ '<|' + message['role'] + '|>' + message['content'] + '<|end|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>' }}{% else %}{{ eos_token }}{% endif %}"
        """
        for message in messages:
            formatted_prompt += f"<|{message['role']}|>{message['content']}<|end|>"
        formatted_prompt += "<|assistant|>"

        return formatted_prompt
