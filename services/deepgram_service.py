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
        logger.error("[Deepgram] API key not configured")
        return

    while not cancel_event.is_set():
        try:
            await _connect_and_stream(
                api_key,
                audio_queue,
                on_partial,
                on_final,
                cancel_event,
                on_speech_started,
                is_speaking_getter
            )
        except Exception as e:
            logger.error(f"[Deepgram] Stream error: {e}")

        if cancel_event.is_set():
            break

        logger.info("[Deepgram] Reconnecting in 2 seconds...")
        await asyncio.sleep(2)


async def _connect_and_stream(
    api_key: str,
    audio_queue: asyncio.Queue,
    on_partial: Callable[[str], None],
    on_final: Callable[[str], None],
    cancel_event: asyncio.Event,
    on_speech_started: Optional[Callable[[], None]] = None,
    is_speaking_getter: Optional[Callable[[], bool]] = None
) -> None:
    logger.info("[Deepgram] Connecting...")

    from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions

    dg_client = DeepgramClient(api_key)
    connection = dg_client.listen.live.v("1")

    def on_message(self, result, **kwargs):
        try:
            transcript = result.channel.alternatives[0].transcript
            if transcript:
                if result.is_final:
                    logger.info(f"[Deepgram] Final: {transcript}")
                    on_final(transcript)
                else:
                    logger.info(f"[Deepgram] Partial: {transcript}")
                    on_partial(transcript)
        except Exception as e:
            logger.error(f"[Deepgram] Message error: {e}")

    def on_speech_started_event(self, speech_started, **kwargs):
        logger.info("[Deepgram] Speech started")
        if on_speech_started and is_speaking_getter and is_speaking_getter():
            on_speech_started()

    def on_utterance_end(self, utterance_end, **kwargs):
        logger.info("[Deepgram] Utterance end")

    def on_error(self, error, **kwargs):
        logger.error(f"[Deepgram] Error: {error}")

    def on_close(self, close, **kwargs):
        logger.info("[Deepgram] Connection closed")

    connection.on(LiveTranscriptionEvents.Transcript, on_message)
    connection.on(LiveTranscriptionEvents.SpeechStarted, on_speech_started_event)
    connection.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end)
    connection.on(LiveTranscriptionEvents.Error, on_error)
    connection.on(LiveTranscriptionEvents.Close, on_close)

    options = LiveOptions(
        model="nova-2",
        language="en",
        smart_format=True,
        interim_results=True,
        vad_events=True,
        punctuate=True,
        utterance_end_ms=1000,
        endpointing=500,
        encoding="linear16",
        sample_rate=16000,
        channels=1
    )

    started = connection.start(options)
    if not started:
        logger.error("[Deepgram] Failed to start connection")
        return

    logger.info("[Deepgram] Connected and listening")

    while not cancel_event.is_set():
        try:
            audio_data = await asyncio.wait_for(
                audio_queue.get(), timeout=0.1
            )
            if audio_data:
                connection.send(audio_data)
        except asyncio.TimeoutError:
            continue
        except Exception as e:
            logger.error(f"[Deepgram] Send error: {e}")
            break

    connection.finish()
    logger.info("[Deepgram] Stream finished")