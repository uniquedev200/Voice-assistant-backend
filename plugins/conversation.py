from typing import Dict, Any


def register() -> Dict[str, Any]:
    return {
        "name": "conversation",
        "description": "Handles plain conversation with no tools",
        "tools": [],
        "handler": handle_conversation
    }


def handle_conversation(tool_name: str, args: Dict[str, Any], context: Any, session: Any) -> Any:
    return None