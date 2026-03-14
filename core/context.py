import asyncio
from typing import List, Dict, Any

BASE_SYSTEM_PROMPT = (
    "You are Jarvis, a helpful voice assistant. You are fast, concise, and conversational. "
    "Keep responses short since they will be spoken aloud. Never use markdown, bullet points, or formatting — "
    "plain conversational sentences only."
)

MAX_HISTORY = 20


class ConversationContext:
    def __init__(self):
        self._history: List[Dict[str, Any]] = [
            {"role": "system", "content": BASE_SYSTEM_PROMPT}
        ]

    def add_user_message(self, text: str) -> None:
        self._history.append({"role": "user", "content": text})
        self._trim_history()

    def add_assistant_message(self, text: str) -> None:
        self._history.append({"role": "assistant", "content": text})
        self._trim_history()

    def get_history(self) -> List[Dict[str, Any]]:
        return self._history.copy()

    def clear(self) -> None:
        self._history = [
            {"role": "system", "content": BASE_SYSTEM_PROMPT}
        ]

    def _trim_history(self) -> None:
        if len(self._history) > MAX_HISTORY + 1:
            system_msg = self._history[0]
            self._history = [system_msg] + self._history[-(MAX_HISTORY):]