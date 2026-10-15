# [MODULE DOC] Conversation title generator for the smol_tools demo, backed by SmolLM2-1.7B.
from .base import SmolTool
from typing import Generator

class SmolTitler(SmolTool):
    # [CLASS DOC] SmolTool with no system prompt and prefix "Create a title for this conversation:".
    def __init__(self):
        # [FUNCTION DOC] Loads/reuses the shared SmolLM2-1.7B-Instruct GGUF model with an empty system
        # prompt and the title prefix.
        super().__init__(
            model_repo="andito/SmolLM2-1.7B-Instruct-F16-GGUF",
            model_filename="smollm2-1.7b-8k-dpo-f16.gguf",
            system_prompt="",
            prefix_text="Create a title for this conversation:",
        )

    def process(self, text: str) -> Generator[str, None, None]:
        # [FUNCTION DOC] Sends a single user message "<prefix>\n<text>" and yields the cumulative title text
        # (max 128 tokens,
        # temperature 0.6, top_p 0.9, top_k 0, repeat_penalty 1.1).
        messages = [
            {"role": "user", "content": f"{self.prefix_text}\n{text}"}
        ]
        yield from self._create_chat_completion(messages, max_tokens=128, temperature=0.6, top_p=0.9, top_k=0, repeat_penalty=1.1)