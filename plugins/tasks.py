import json
import logging
import os

import httpx
from typing import Dict, Any

logger = logging.getLogger(__name__)

UPSTASH_REDIS_URL = os.getenv("UPSTASH_REDIS_URL")
UPSTASH_REDIS_TOKEN = os.getenv("UPSTASH_REDIS_TOKEN")
QUEUE_NAME = "jarvis_tasks"


async def push_task(task: dict) -> None:
    serialized = json.dumps(task)
    async with httpx.AsyncClient() as client:
        await client.post(
            f"{UPSTASH_REDIS_URL}",
            headers={"Authorization": f"Bearer {UPSTASH_REDIS_TOKEN}"},
            json=["RPUSH", QUEUE_NAME, serialized],
        )


def register() -> Dict[str, Any]:
    return {
        "name": "tasks",
        "description": "Device control and web search tools",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "open_app",
                    "description": "Opens an application on the target device",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "app_name": {"type": "string", "description": "Name of the app to open"},
                            "device_id": {"type": "string", "description": "Target device ID"},
                        },
                        "required": ["app_name", "device_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "send_message",
                    "description": "Sends a message via the specified platform",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "platform": {"type": "string", "enum": ["whatsapp", "telegram", "sms"], "description": "Messaging platform"},
                            "recipient": {"type": "string", "description": "Recipient phone number or username"},
                            "message": {"type": "string", "description": "Message content"},
                            "device_id": {"type": "string", "description": "Target device ID"},
                        },
                        "required": ["platform", "recipient", "message", "device_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Performs a web search and returns top results",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_status",
                    "description": "Gets device status including battery, volume, running apps",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "device_id": {"type": "string", "description": "Target device ID"},
                        },
                        "required": ["device_id"],
                    },
                },
            },
        ],
        "handler": handle_task,
    }


async def handle_task(tool_name: str, args: Dict[str, Any], context: Any, session: Any) -> Any:
    logger.info(f"Task tool called: {tool_name} with args: {args}")
    
    task = {
        "task": tool_name,
        "args": args,
    }
    
    await push_task(task)
    return {"status": "queued", "task": tool_name}
