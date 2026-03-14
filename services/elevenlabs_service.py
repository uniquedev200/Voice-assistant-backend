import asyncio
import os
import logging
from typing import AsyncGenerator, Optional

logger = logging.getLogger(__name__)

ELEVEN_TURBO_V2 = "eleven_turbo_v2"


async def stream_tts(
    text: str,
    cancel_event: asyncio.Event
) -> AsyncGenerator[bytes, None]:
    api_key = os.getenv("ELEVENLABS_API_KEY")
    voice_id = os.getenv("ELEVENLABS_VOICE_ID")
    
    if not api_key or not voice_id:
        logger.error("ElevenLabs API key or voice ID not configured")
        return

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    
    data = {
        "text": text,
        "model_id": ELEVEN_TURBO_V2,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }

    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                if response.status != 200:
                    logger.error(f"ElevenLabs error: {response.status}")
                    return
                    
                async for chunk in response.content.iter_chunked(1024):
                    if cancel_event.is_set():
                        logger.info("TTS cancelled by barge-in")
                        break
                    if chunk:
                        yield chunk
                        
    except Exception as e:
        logger.error(f"TTS stream error: {e}")