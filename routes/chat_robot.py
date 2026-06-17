import logging
import os
import requests
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Request, HTTPException
from schemas.request import QuestionRequest
from services.chat_robot import ChatRobot
from config.config import Settings
from typing import Optional
import json
import base64
import asyncio
from google import genai
from google.genai import types

chat_router = APIRouter()

@chat_router.post("/")
async def chat_mcp(req: QuestionRequest):
    logging.info("Start Processing Chat General")

    chat = ChatRobot()

    try:
        answer = await chat.main(req)
    except ValueError as e: # Handle missing or invalid parameters
        logging.error(
            f"ValueError: {e}\n\n"
            f"**========== [Displaying Received Request Data Because There's An Error] ==========**\n{req}"
        )
        return JSONResponse({"status": 400, "message": str(e)})
    except Exception as e: # Handle any other exceptions
        logging.exception(f"Unexpected error occurred: {e}")
        return JSONResponse({"status": 500, "message": "An unexpected error occurred."})

    logging.info("Finished Processing Chat General")

    return JSONResponse(
        {
            'status': 200, 
            'data': answer
        }
    )

@chat_router.websocket("/ws/live")
async def websocket_live_gemini(websocket: WebSocket, session_id: Optional[int] = None):
    await websocket.accept()
    logging.info(f"WebSocket client connected. session_id: {session_id}")

    chat = ChatRobot()
    db_session_id = session_id

    # If no session_id is provided, create a new one in Supabase
    if db_session_id is None:
        try:
            db_session_id = chat.create_history()
            # Notify the client of the new session_id
            await websocket.send_json({"type": "session_created", "session_id": db_session_id})
        except Exception as e:
            logging.error(f"Failed to create session in Supabase: {e}")
            await websocket.close(code=1011)
            return

    # Initialize Gemini Client
    try:
        settings_obj = Settings()
        gemini_client = genai.Client(api_key=settings_obj.google_key)
    except Exception as e:
        logging.error(f"Failed to initialize Gemini Client: {e}")
        await websocket.close(code=1011)
        return

    # Configuration for Gemini Live API
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=(
            "Kamu adalah asisten pintar robot layanan bernama Raisa. "
            "Berbicaralah dengan ramah, sopan, singkat, dan natural dalam Bahasa Indonesia. "
            "Jangan memberikan jawaban yang terlalu panjang karena ini adalah percakapan suara."
        ),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name="Kore"  # kore is a friendly female voice config
                )
            )
        )
    )

    # State for accumulating the conversation turn text to save in DB history
    class TurnState:
        def __init__(self):
            self.user_text = ""
            self.assistant_text = ""
            self.is_first_turn = True

    state = TurnState()

    def save_turn_to_db():
        if not db_session_id:
            return
        
        user_prompt = state.user_text.strip()
        bot_response = state.assistant_text.strip()
        
        if not user_prompt and not bot_response:
            return
            
        logging.info(f"Saving live turn to DB. User: '{user_prompt}', Bot: '{bot_response}'")
        
        try:
            # 1. Save User Message
            if user_prompt:
                user_content = types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=user_prompt)]
                )
                chat.save_message(str(db_session_id), user_content, showed=True)
            
            # 2. Save Assistant Message
            if bot_response:
                assistant_content = types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=bot_response)]
                )
                chat.save_message(str(db_session_id), assistant_content, showed=True)
                
            # 3. Generate session title if it's the first turn
            chat.generate_session_title(
                session_id=str(db_session_id),
                user_prompt=user_prompt or "[Voice Input]",
                bot_answer=bot_response
            )
            state.is_first_turn = False
        except Exception as db_err:
            logging.error(f"Error saving live turn to DB: {db_err}")

    try:
        # Model for Multimodal Live API (user requested gemini-3.1-flash-live-preview)
        model = "gemini-3.1-flash-live-preview" 
        
        async with gemini_client.aio.live.connect(model=model, config=config) as session:
            logging.info("Connected to Gemini Live API WebSocket")
            await websocket.send_json({"type": "ready"})

            # Define the task to receive messages from Gemini and send them to the frontend
            async def receive_from_gemini():
                try:
                    async for message in session.receive():
                        # Check input_transcription
                        if message.server_content:
                            server_content = message.server_content
                            
                            if server_content.input_transcription:
                                text = server_content.input_transcription.text
                                if text:
                                    logging.info(f"Gemini STT (User): {text}")
                                    state.user_text += " " + text
                                    await websocket.send_json({"type": "user_transcription", "text": text})

                            # Check for interruption
                            if getattr(server_content, 'interrupted', False):
                                logging.info("Gemini live stream interrupted")
                                state.assistant_text += " [Terpotong]"
                                await websocket.send_json({"type": "interrupted"})
                                # Save partial conversation turn
                                save_turn_to_db()
                                state.user_text = ""
                                state.assistant_text = ""
                                continue
                            
                            # Check model turn parts
                            if server_content.model_turn:
                                for part in server_content.model_turn.parts:
                                    # Text part
                                    if part.text:
                                        state.assistant_text += part.text
                                        await websocket.send_json({"type": "assistant_text", "text": part.text})
                                    # Audio / Inline Data part
                                    elif part.inline_data:
                                        # Base64 encode the binary audio PCM data to send to frontend
                                        audio_b64 = base64.b64encode(part.inline_data.data).decode("utf-8")
                                        await websocket.send_json({
                                            "type": "assistant_audio",
                                            "data": audio_b64,
                                            "mimeType": part.inline_data.mime_type
                                        })
                            
                            # Check turn_complete
                            if getattr(server_content, 'turn_complete', False):
                                logging.info("Gemini live turn complete")
                                await websocket.send_json({"type": "turn_complete"})
                                save_turn_to_db()
                                state.user_text = ""
                                state.assistant_text = ""

                except asyncio.CancelledError:
                    pass
                except Exception as gemini_rx_err:
                    logging.error(f"Error receiving from Gemini: {gemini_rx_err}")
                    try:
                        await websocket.send_json({"type": "error", "message": str(gemini_rx_err)})
                    except Exception:
                        pass

            # Start the background task to receive from Gemini
            rx_task = asyncio.create_task(receive_from_gemini())

            # Loop to receive messages from the frontend and send them to Gemini
            try:
                while True:
                    # Receive message from frontend (either binary audio or text/JSON)
                    msg_received = await websocket.receive()
                    
                    if "bytes" in msg_received and msg_received["bytes"]:
                        # Raw binary audio chunk (16kHz, 16-bit, little-endian mono PCM)
                        audio_data = msg_received["bytes"]
                        # Forward to Gemini Live Session
                        await session.send(
                            input={
                                "audio": {
                                    "data": audio_data,
                                    "mime_type": "audio/pcm;rate=16000"
                                }
                            }
                        )
                    elif "text" in msg_received and msg_received["text"]:
                        data = json.loads(msg_received["text"])
                        msg_type = data.get("type")
                        
                        if msg_type == "text":
                            user_text = data.get("text", "")
                            logging.info(f"Frontend text input: {user_text}")
                            state.user_text = user_text
                            # Send text to Gemini Live Session
                            await session.send(input={"text": user_text}, end_of_turn=True)
                        elif msg_type == "ping":
                            await websocket.send_json({"type": "pong"})
            except WebSocketDisconnect:
                logging.info("WebSocket frontend client disconnected")
            finally:
                # Cancel the Gemini receiver task
                rx_task.cancel()
                await rx_task
                
                # Save any remaining text from the current active turn before closing
                save_turn_to_db()

    except Exception as e:
        logging.error(f"Error in WebSocket Live Session: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass