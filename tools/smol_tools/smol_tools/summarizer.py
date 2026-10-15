# [MODULE DOC] Summarization and question-answering tool for the smol_tools demo, backed by SmolLM2-1.7B.
from .base import SmolTool
from typing import Generator, Optional
from dataclasses import dataclass
from datetime import datetime
from typing import List

@dataclass
class SummaryMessage:
    # [CLASS DOC] Dataclass for a summary conversation message: role, content, timestamp. Not used within
    # this module.
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime

class SmolSummarizer(SmolTool):
    # [CLASS DOC] SmolTool that summarizes text in up to three sentences, or answers a follow-up question
    # about the text.
    def __init__(self):
        # [FUNCTION DOC] Sets display name "SmolLM2-1.7B" and loads/reuses the shared GGUF model with the
        # summarization system
        # prompt (prefix_text stays "").
        self.name = "SmolLM2-1.7B"
        
        super().__init__(
            model_repo="andito/SmolLM2-1.7B-Instruct-F16-GGUF",
            model_filename="smollm2-1.7b-8k-dpo-f16.gguf",
            system_prompt="Concisely summarize the main points of the input text in up to three sentences, focusing on key information and events.",
        )

    def process(self, text: str, question: Optional[str] = None) -> Generator[str, None, None]:
        # [FUNCTION DOC] If question is None: prints "Summarizing text" and sends system prompt, the text as
        # user message and a
        # prefilled assistant turn "This is a short summary of the text:". Otherwise prints "Answering
        # question" and
        # sends only a user message "Original text:\n<text>\n\nQuestion: <question>". Yields cumulative output
        # (max 1024 tokens, temperature 0.1, top_p 0.9).
        # KNOWN ISSUE: since prefix_text is empty the summary prompt starts with a stray newline.
        if question is None:
            print("Summarizing text")
            prompt = f"{self.prefix_text}\n{text}"
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "This is a short summary of the text:"}
            ]
        else:
            print("Answering question")
            prompt = f"Original text:\n{text}\n\nQuestion: {question}"
            messages = [
                {"role": "user", "content": prompt},
            ]

        for chunk in self._create_chat_completion(messages, max_tokens=1024, temperature=0.1, top_p=0.9):
            yield chunk
