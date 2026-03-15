import asyncio
import os
import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, AsyncGenerator
from groq import AsyncGroq

logger = logging.getLogger(__name__)


@dataclass
class ToolCallEvent:
    tool_name: str
    args: Dict[str, Any]


async def stream_response(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    cancel_event: asyncio.Event
) -> AsyncGenerator[Any, None]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("[Groq] API key not configured")
        return

    client = AsyncGroq(api_key=api_key)
    logger.info(f"[Groq] Calling with {len(messages)} messages")

    try:
        kwargs = {
            "model": "llama-3.3-70b-versatile",
            "messages": messages,
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 500
        }

        # Only pass tools if there are any
        if tools:
            kwargs["tools"] = tools

        response = await client.chat.completions.create(**kwargs)

        tool_call_accumulator = {}

        async for chunk in response:
            if cancel_event.is_set():
                logger.info("[Groq] Stream cancelled")
                break

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index if hasattr(tc, 'index') else 0
                    if idx not in tool_call_accumulator:
                        tool_call_accumulator[idx] = {
                            "name": "",
                            "arguments": ""
                        }
                    if tc.function:
                        if tc.function.name:
                            tool_call_accumulator[idx]["name"] += tc.function.name
                        if tc.function.arguments:
                            tool_call_accumulator[idx]["arguments"] += tc.function.arguments

            elif delta.content:
                yield delta.content

        # Yield any accumulated tool calls
        for idx, tc in tool_call_accumulator.items():
            if tc["name"]:
                try:
                    args = json.loads(tc["arguments"] or "{}")
                except Exception:
                    args = {}
                yield ToolCallEvent(tool_name=tc["name"], args=args)

    except Exception as e:
        logger.error(f"[Groq] Error: {e}", exc_info=True)
        if "429" in str(e):
            logger.info("[Groq] Rate limited — retrying in 2s")
            await asyncio.sleep(2)
            try:
                response = await client.chat.completions.create(**kwargs)
                async for chunk in response:
                    if cancel_event.is_set():
                        break
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            except Exception as retry_err:
                logger.error(f"[Groq] Retry failed: {retry_err}")
    finally:
        await client.close()
        logger.info("[Groq] Stream complete")