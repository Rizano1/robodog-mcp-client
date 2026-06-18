import asyncio
import logging
import os
import sys
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)

async def test_live_audio():
    from dotenv import load_dotenv
    parent_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
    load_dotenv(parent_env)
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logging.error("GOOGLE_API_KEY not found!")
        return
        
    client = genai.Client(api_key=api_key)
    model = "gemini-3.1-flash-live-preview"
    
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction="Jawab singkat jika mendengar sesuatu.",
        input_audio_transcription=types.AudioTranscriptionConfig()
    )
    
    try:
        logging.info("Connecting to Gemini Live...")
        async with client.aio.live.connect(model=model, config=config) as session:
            logging.info("Connected!")
            
            # Start a receiver task
            async def receive():
                async for response in session.receive():
                    logging.info(f"Received: {response}")
            
            rx_task = asyncio.create_task(receive())
            
            # Send 10 chunks of silence (16kHz, 16-bit mono PCM, 100ms each = 1600 samples = 3200 bytes)
            silent_chunk = b'\x00' * 3200
            
            logging.info("Sending silent audio chunks using send_realtime_input...")
            for i in range(5):
                await session.send_realtime_input(
                    audio=types.Blob(
                        data=silent_chunk,
                        mime_type="audio/pcm;rate=16000"
                    )
                )
                await asyncio.sleep(0.1)
                
            logging.info("Sending some text turn to trigger response...")
            await session.send_client_content(
                turns=types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="Halo")]
                ),
                turn_complete=True
            )
            
            await asyncio.sleep(5)
            rx_task.cancel()
            await rx_task
    except Exception as e:
        logging.exception(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_live_audio())
