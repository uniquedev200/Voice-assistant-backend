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
    logger.info(f"Client connected: {device_id}")
    
    session = Session(device_id)
    sessions[device_id] = session
    
    audio_queue: asyncio.Queue = asyncio.Queue()
    deepgram_task = None
    
    def on_partial(text: str):
        logger.info(f"Deepgram partial: {text}")
        
    def on_final(text: str):
        logger.info(f"Deepgram final: {text}")
        asyncio.create_task(process_turn(text, websocket, session, audio_queue))
        
    def on_speech_started():
        logger.info("Barge-in detected: speech started while speaking")
        session.trigger_barge_in()
        asyncio.create_task(websocket.send_json({"type": "STOP_AUDIO"}))
        asyncio.create_task(session.cancel_all_tasks())
        asyncio.create_task(asyncio.sleep(0.3, session.reset_cancel()))
        
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
                    
            elif "text" in data:
                text_data = data["text"]
                try:
                    control = json.loads(text_data)
                    if control.get("type") == "STOP":
                        logger.info("Client sent STOP")
                        session.trigger_barge_in()
                        await session.cancel_all_tasks()
                        await websocket.send_json({"type": "STOP_ACK"})
                    elif control.get("type") == "PING":
                        await websocket.send_json({"type": "PONG"})
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received: {text_data}")
                    
    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {device_id}")
    finally:
        if deepgram_task and not deepgram_task.done():
            deepgram_task.cancel()
        await session.cancel_all_tasks()
        sessions.pop(device_id, None)
        logger.info(f"Session cleaned up for {device_id}")


async def process_turn(transcript: str, websocket: WebSocket, session: Session, audio_queue: asyncio.Queue):
    logger.info(f"Processing turn: {transcript}")
    
    session.context.add_user_message(transcript)
    messages = session.context.get_history()
    tools = get_all_tools()
    
    session.is_speaking = True
    session.reset_cancel()
    
    sentence_buffer = ""
    
    try:
        async for token in stream_response(messages, tools, session.cancel_event):
            if session.cancel_event.is_set():
                logger.info("Groq stream cancelled")
                break
                
            if isinstance(token, ToolCallEvent):
                logger.info(f"Tool call: {token.tool_name} {token.args}")
                result = route(token.tool_name, token.args, session.context, session)
                session.context.add_assistant_message(f"[Tool: {token.tool_name}]")
                messages = session.context.get_history()
            else:
                sentence_buffer += token
                
                for end_mark in [".", ",", "?", "!"]:
                    if sentence_buffer.endswith(end_mark):
                        if sentence_buffer.strip():
                            logger.info(f"TTS: {sentence_buffer}")
                            async for audio_chunk in stream_tts(sentence_buffer, session.cancel_event):
                                if session.cancel_event.is_set():
                                    break
                                await websocket.send_bytes(audio_chunk)
                        sentence_buffer = ""
                        break
                        
        if sentence_buffer.strip():
            logger.info(f"TTS flush: {sentence_buffer}")
            async for audio_chunk in stream_tts(sentence_buffer, session.cancel_event):
                if session.cancel_event.is_set():
                    break
                await websocket.send_bytes(audio_chunk)
                
        full_response = messages[-1]["content"] if messages else ""
        session.context.add_assistant_message(full_response)
        
    except Exception as e:
        logger.error(f"Error in process_turn: {e}")
    finally:
        session.is_speaking = False
        logger.info("Turn complete")