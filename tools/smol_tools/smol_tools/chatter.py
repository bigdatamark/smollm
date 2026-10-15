# [MODULE DOC] Multi-turn chat tool for the smol_tools demo with chat persistence: chats are saved as JSON
# files
# "saved_chats/chat_<id>.json" (relative to the current working directory) containing {"id", "messages"}.
from .base import SmolTool
from typing import Generator, List, Dict
from dataclasses import dataclass
from datetime import datetime
import json
import os

@dataclass
class ChatMessage:
    # [CLASS DOC] Dataclass for one chat message: role ("user" or "assistant"), content text, timestamp
    # (datetime).
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime

    # Add methods to convert to/from dict for JSON serialization
    def to_dict(self):
        # [FUNCTION DOC] Returns a JSON-serializable mapping with role, content and the timestamp as an
        # ISO-8601 string.
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data):
        # [FUNCTION DOC] Classmethod: builds a ChatMessage from a mapping produced by to_dict (parses the
        # ISO timestamp).
        return cls(
            role=data['role'],
            content=data['content'],
            timestamp=datetime.fromisoformat(data['timestamp'])
        )

class SmolChatter(SmolTool):
    # [CLASS DOC] SmolTool holding the current chat history (list of ChatMessage), current_chat_id, the
    # chats directory and
    # a snapshot of the last saved/loaded state (_original_chat_state) for modification tracking.
    def __init__(self):
        # [FUNCTION DOC] Initializes empty history/archive, no current chat, chats_dir "saved_chats"
        # (created if missing),
        # display name "SmolLM2-1.7B", then loads the shared SmolLM2-1.7B GGUF model with a helpful-assistant
        # system prompt via the base class.
        self.chat_history: List[ChatMessage] = []
        self.chat_archive: Dict[str, List[ChatMessage]] = {}
        self.current_chat_id = None
        self.chats_dir = "saved_chats"
        self._original_chat_state = None  # To track modifications
        self.name = "SmolLM2-1.7B"
        
        # Create chats directory if it doesn't exist
        if not os.path.exists(self.chats_dir):
            os.makedirs(self.chats_dir)
            
        super().__init__(
            model_repo="andito/SmolLM2-1.7B-Instruct-F16-GGUF",
            model_filename="smollm2-1.7b-8k-dpo-f16.gguf",
            system_prompt="You are a helpful AI assistant named SmolLM, trained by Hugging Face..",
        )

    def start_new_chat(self):
        # [FUNCTION DOC] Starts a new chat: id from the current local time "YYYYMMDD_HHMMSS", empty history,
        # no saved snapshot.
        """Start a new chat with a unique ID"""
        self.current_chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.chat_history = []
        self._original_chat_state = None

    def has_current_chat(self) -> bool:
        # [FUNCTION DOC] Returns True if the current history contains at least one message.
        """Check if there are any messages in the current chat"""
        return len(self.chat_history) > 0

    def save_current_chat(self, title: str = None, overwrite: bool = False):
        # [FUNCTION DOC] No-op when history is empty. Otherwise sets current_chat_id to title when given
        # (else a timestamp id if
        # none), writes the chat JSON to "<chats_dir>/chat_<id>.json" (overwriting) and records the saved
        # snapshot.
        # KNOWN ISSUE: the overwrite flag has no effect; with a title the id always becomes the title.
        """Save the current chat to disk if it has any messages"""
        if not self.chat_history:
            return
            
        if title:
            # If overwriting, use existing chat_id if it matches the title
            if not overwrite or self.current_chat_id != title:
                self.current_chat_id = title
        elif not self.current_chat_id:
            self.current_chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # Convert chat history to serializable format
        chat_data = {
            'id': self.current_chat_id,
            'messages': [msg.to_dict() for msg in self.chat_history]
        }
        
        # Save to file
        filename = f"{self.chats_dir}/chat_{self.current_chat_id}.json"
        with open(filename, 'w') as f:
            json.dump(chat_data, f)
            
        # Update original state to reflect saved state
        self._original_chat_state = [msg.to_dict() for msg in self.chat_history]

    def load_chat(self, chat_id: str):
        # [FUNCTION DOC] Reads "<chats_dir>/chat_<chat_id>.json", sets current_chat_id and history from it
        # and records the snapshot.
        # If the file is missing prints a message and leaves state unchanged (other errors propagate).
        """Load a specific chat from disk"""
        filename = f"{self.chats_dir}/chat_{chat_id}.json"
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                self.current_chat_id = data['id']
                self.chat_history = [ChatMessage.from_dict(msg) for msg in data['messages']]
                # Store original state for modification tracking
                self._original_chat_state = [msg.to_dict() for msg in self.chat_history]
        except FileNotFoundError:
            print(f"Chat {chat_id} not found")

    def is_chat_modified(self) -> bool:
        # [FUNCTION DOC] Returns True if the history differs from the last saved/loaded snapshot; with no
        # snapshot, True if the
        # history is non-empty.
        """Check if the current chat has been modified since loading"""
        if self._original_chat_state is None:
            # New chat that hasn't been saved yet
            return len(self.chat_history) > 0
            
        current_state = [msg.to_dict() for msg in self.chat_history]
        return current_state != self._original_chat_state

    def get_saved_chats(self) -> List[str]:
        # [FUNCTION DOC] Lists ids of files named "chat_<id>.json" in chats_dir, sorted descending (newest
        # first for timestamp ids).
        """Get list of saved chat IDs"""
        chats = []
        for filename in os.listdir(self.chats_dir):
            if filename.startswith('chat_') and filename.endswith('.json'):
                chat_id = filename[5:-5]  # Remove 'chat_' prefix and '.json' suffix
                chats.append(chat_id)
        return sorted(chats, reverse=True)  # Most recent first

    def _warm_up(self):
        # [FUNCTION DOC] Runs the base warm-up, then clears the history so the warm-up exchange is not kept.
        super()._warm_up()
        self.clear_chat_history()

    def process(self, text: str) -> Generator[str, None, None]:
        # [FUNCTION DOC] Appends the user message, builds messages (system prompt + whole history), streams
        # a completion
        # (max 1024 tokens), yielding the cumulative response text, then appends the final response as an
        # assistant
        # message.
        # KNOWN ISSUE: the comment says only the last 5 messages are used, but the whole history is sent,
        # which can
        # exceed the model context on long chats.
        # Add user message to history
        self.chat_history.append(ChatMessage(
            role="user",
            content=text,
            timestamp=datetime.now()
        ))
        
        # Build messages including chat history
        messages = [{"role": "system", "content": self.system_prompt}]
        # Include last 5 messages for context
        for msg in self.chat_history:
            messages.append({"role": msg.role, "content": msg.content})

        # Generate response
        response = ""
        for chunk in self._create_chat_completion(messages, max_tokens=1024):
            response = chunk
            yield chunk
        
        # Add assistant's response to history
        self.chat_history.append(ChatMessage(
            role="assistant",
            content=response,
            timestamp=datetime.now()
        ))

    def get_chat_history(self) -> List[ChatMessage]:
        # [FUNCTION DOC] Returns the current history list (the live object, not a copy).
        return self.chat_history
    
    def clear_chat_history(self):
        # [FUNCTION DOC] Empties the current history (does not touch the id or saved snapshot).
        self.chat_history = []

    def get_current_chat_id(self) -> str:
        # [FUNCTION DOC] Returns the current chat id (None if no chat was started/loaded).
        """Get the ID of the current chat"""
        return self.current_chat_id