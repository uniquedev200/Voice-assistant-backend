import asyncio
import os
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


async def stream_audio(
    audio_queue: asyncio.Queue,
    on_partial: Callable[[str], None],
    on_final: Callable[[str], None],
    cancel_event: asyncio.Event,
    on_speech_started: Optional[Callable[[], None]] = None,
    is_speaking_getter: Optional[Callable[[], bool]] = None
) -> None:
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        logger.error("Deepgram API key not configured")
        return

    from deepgram import AsyncDeepgramClient
    from deepgram.listen import v1

    dg_client = AsyncDeepgramClient(api_key)

    dg_connection = None
    try:
        async with dg_client.listen.v1.connect(
            model="nova-2",
            language="en",
            smart_format="true",
            interim_results="true",
            vad_events="true",
            punctuate="true"
        ) as connection:
            dg_connection = connection

            async def send_audio():
                while not cancel_event.is_set():
                    try:
                        audio_data = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                        if audio_data:
                            await connection.send(audio_data)
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        logger.error(f"Error sending audio: {e}")
                        break

            async def receive_events():
                async for result in connection:
                    if cancel_event.is_set():
                        break

                    if isinstance(result, v1.ListenV1Results):
                        if result.channel and result.channel.alternatives:
                            transcript = result.channel.alternatives[0].transcript
                            if transcript:
                                if result.is_final:
                                    logger.info(f"Deepgram final: {transcript}")
                                    on_final(transcript)
                                else:
                                    on_partial(transcript)

                    elif isinstance(result, v1.ListenV1SpeechStarted):
                        logger.info("Deepgram: SpeechStarted event")
                        if on_speech_started and is_speaking_getter and is_speaking_getter():
                            on_speech_started()

                    elif isinstance(result, v1.ListenV1UtteranceEnd):
                        logger.info("Deepgram: SpeechEnded event")

            send_task = asyncio.create_task(send_audio())
            receive_task = asyncio.create_task(receive_events())

            await asyncio.gather(send_task, receive_task, return_exceptions=True)

    except Exception as e:
        logger.error(f"Deepgram connection error: {e}")
    finally:
        logger.info("Deepgram stream closed")