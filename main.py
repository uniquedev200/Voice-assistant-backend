import asyncio
import logging
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from core.session import Session
from core.registry import discover_plugins, get_all_tools, route
from services.deepgram_service import stream_audio
from services.groq_service import stream_response, ToolCallEvent
from services.elevenlabs_service import stream_tts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sessions: dict[str, Session] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    discover_plugins()
    logger.info("Plugins loaded")
    tools = get_all_tools()
    logger.info(f"Registered tools: {tools}")
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.websocket("/ws/{device_id}")
async def websocket_endpoint(websocket: WebSocket, device_id: str):
    await websocket.accept()
    logger.info(f"[WS] Client connected: {device_id}")

    session = Session(device_id)
    sessions[device_id] = session

    audio_queue: asyncio.Queue = asyncio.Queue()

    def on_partial(text: str):
        logger.info(f"[Deepgram] Partial: {text}")
        asyncio.create_task(
            websocket.send_json({"type": "PARTIAL_TRANSCRIPT", "text": text})
        )

    def on_final(text: str):
        logger.info(f"[Deepgram] Final: {text}")
        asyncio.create_task(
            websocket.send_json({"type": "FINAL_TRANSCRIPT", "text": text})
        )
        asyncio.create_task(
            process_turn(text, websocket, session, audio_queue)
        )

    def on_speech_started():
        logger.info("[Barge-in] Speech started while speaking")
        session.trigger_barge_in()
        asyncio.create_task(websocket.send_json({"type": "STOP_AUDIO"}))

    is_speaking_getter = lambda: session.is_speaking

    deepgram_task = asyncio.create_task(
        stream_audio(
            audio_queue,
            on_partial,
            on_final,
            session.cancel_event,
            on_speech_started,
            is_speaking_getter
        )
    )
    session.add_task(deepgram_task)

    try:
        while True:
            data = await websocket.receive()

            if "bytes" in data:
                audio_bytes = data["bytes"]
                if audio_bytes:
                    await audio_queue.put(audio_bytes)
                    logger.debug(f"[WS] Audio chunk queued: {len(audio_bytes)} bytes")

            elif "text" in data:
                text_data = data["text"]
                try:
                    control = json.loads(text_data)
                    msg_type = control.get("type")

                    if msg_type == "REGISTER":
                        logger.info(f"[WS] Device registered: {control.get('device_id')}")
                        await websocket.send_json({"type": "REGISTERED", "device_id": device_id})

                    elif msg_type == "STOP":
                        logger.info("[WS] Client sent STOP")
                        session.trigger_barge_in()
                        await session.cancel_all_tasks()
                        await websocket.send_json({"type": "STOP_ACK"})

                    elif msg_type == "BARGE_IN":
                        logger.info("[WS] Client sent BARGE_IN")
                        session.trigger_barge_in()
                        await websocket.send_json({"type": "STOP_AUDIO"})

                    elif msg_type == "PING":
                        await websocket.send_json({"type": "PONG"})

                except json.JSONDecodeError:
                    logger.warning(f"[WS] Invalid JSON: {text_data}")

    except WebSocketDisconnect:
        logger.info(f"[WS] Client disconnected: {device_id}")
    finally:
        if deepgram_task and not deepgram_task.done():
            deepgram_task.cancel()
        await session.cancel_all_tasks()
        sessions.pop(device_id, None)
        logger.info(f"[WS] Session cleaned up: {device_id}")


async def process_turn(
    transcript: str,
    websocket: WebSocket,
    session: Session,
    audio_queue: asyncio.Queue
):
    logger.info(f"[Turn] Processing: {transcript}")

    session.context.add_user_message(transcript)
    messages = session.context.get_history()
    tools = get_all_tools()

    session.is_speaking = True
    session.reset_cancel()

    sentence_buffer = ""
    full_response = ""

    try:
        async for token in stream_response(messages, tools, session.cancel_event):
            if session.cancel_event.is_set():
                logger.info("[Turn] Cancelled by barge-in")
                break

            if isinstance(token, ToolCallEvent):
                logger.info(f"[Turn] Tool call: {token.tool_name} args={token.args}")
                result = route(token.tool_name, token.args, session.context, session)
                session.context.add_assistant_message(f"[Tool: {token.tool_name}]")
                messages = session.context.get_history()

            else:
                sentence_buffer += token
                full_response += token

                # Send token to client for UI display
                try:
                    await websocket.send_json({
                        "type": "ASSISTANT_TOKEN",
                        "token": token
                    })
                except Exception:
                    pass

                # Flush to TTS on sentence boundary
                for end_mark in [".", "?", "!"]:
                    if sentence_buffer.strip().endswith(end_mark):
                        chunk_text = sentence_buffer.strip()
                        if chunk_text:
                            logger.info(f"[TTS] Sending: {chunk_text}")
                            async for audio_chunk in stream_tts(chunk_text, session.cancel_event):
                                if session.cancel_event.is_set():
                                    break
                                await websocket.send_bytes(audio_chunk)
                        sentence_buffer = ""
                        break

        # Flush any remaining buffer
        if sentence_buffer.strip() and not session.cancel_event.is_set():
            logger.info(f"[TTS] Flushing: {sentence_buffer.strip()}")
            async for audio_chunk in stream_tts(sentence_buffer.strip(), session.cancel_event):
                if session.cancel_event.is_set():
                    break
                await websocket.send_bytes(audio_chunk)

        # Signal response complete
        try:
            await websocket.send_json({"type": "ASSISTANT_DONE"})
        except Exception:
            pass

        # Save full response to context
        if full_response:
            session.context.add_assistant_message(full_response)
            logger.info(f"[Turn] Complete. Response: {full_response[:100]}...")

    except Exception as e:
        logger.error(f"[Turn] Error: {e}", exc_info=True)
    finally:
        session.is_speaking = False
        logger.info("[Turn] Done")