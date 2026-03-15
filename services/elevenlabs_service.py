import asyncio
import os
import logging
from typing import AsyncGenerator

logger = logging.getLogger(__name__)


async def stream_tts(
    text: str,
    cancel_event: asyncio.Event
) -> AsyncGenerator[bytes, None]:
    api_key = os.getenv("ELEVENLABS_API_KEY")
    voice_id = os.getenv("ELEVENLABS_VOICE_ID")

    if not api_key or not voice_id:
        logger.error("[ElevenLabs] API key or voice ID not configured")
        return

    if not text.strip():
        return

    logger.info(f"[ElevenLabs] TTS: {text[:60]}...")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }

    payload = {
        "text": text,
        "model_id": "eleven_turbo_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }

    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    body = await response.text()
                    logger.error(f"[ElevenLabs] Error {response.status}: {body}")
                    return

                logger.info("[ElevenLabs] Streaming audio chunks...")
                async for chunk in response.content.iter_chunked(4096):
                    if cancel_event.is_set():
                        logger.info("[ElevenLabs] Cancelled")
                        break
                    if chunk:
                        yield chunk

                logger.info("[ElevenLabs] Stream complete")

    except Exception as e:
        logger.error(f"[ElevenLabs] Error: {e}", exc_info=True)