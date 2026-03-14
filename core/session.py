import asyncio
from typing import Optional
from core.context import ConversationContext


class Session:
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.context = ConversationContext()
        self.is_speaking = False
        self.cancel_event = asyncio.Event()
        self._tasks: list = []

    def trigger_barge_in(self) -> None:
        self.is_speaking = False
        self.cancel_event.set()

    def reset_cancel(self) -> None:
        self.cancel_event.clear()

    def add_task(self, task: asyncio.Task) -> None:
        self._tasks.append(task)

    async def cancel_all_tasks(self) -> None:
        for task in self._tasks:
            if not task.done():
                task.cancel()
        self._tasks.clear()