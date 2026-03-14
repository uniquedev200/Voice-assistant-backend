import asyncio
import os
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, AsyncGenerator, Optional
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
        logger.error("Groq API key not configured")
        return

    client = AsyncGroq(api_key=api_key)
    
    try:
        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=tools if tools else None,
            stream=True,
            temperature=0.7
        )
        
        async for chunk in response:
            if cancel_event.is_set():
                logger.info("Groq stream cancelled by barge-in")
                break
                
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                
                if delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        if tool_call.function:
                            import json
                            try:
                                args_dict = json.loads(tool_call.function.arguments or "{}")
                            except:
                                args_dict = {}
                            yield ToolCallEvent(
                                tool_name=tool_call.function.name or "",
                                args=args_dict
                            )
                elif delta.content:
                    yield delta.content
                    
    except Exception as e:
        logger.error(f"Groq error: {e}")
        if "429" in str(e):
            logger.info("Rate limited, retrying once after 2 seconds...")
            await asyncio.sleep(2)
            try:
                response = await client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    tools=tools if tools else None,
                    stream=True,
                    temperature=0.7
                )
                
                async for chunk in response:
                    if cancel_event.is_set():
                        break
                        
                    if chunk.choices and chunk.choices[0].delta:
                        delta = chunk.choices[0].delta
                        
                        if delta.tool_calls:
                            for tool_call in delta.tool_calls:
                                if tool_call.function:
                                    import json
                                    try:
                                        args_dict = json.loads(tool_call.function.arguments or "{}")
                                    except:
                                        args_dict = {}
                                    yield ToolCallEvent(
                                        tool_name=tool_call.function.name or "",
                                        args=args_dict
                                    )
                        elif delta.content:
                            yield delta.content
            except Exception as retry_error:
                logger.error(f"Groq retry failed: {retry_error}")
        else:
            logger.error(f"Groq stream error: {e}")
    finally:
        await client.close()