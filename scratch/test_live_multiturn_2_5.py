import asyncio
import logging
import os
import sys
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)

async def test_live_multiturn_2_5():
    from dotenv import load_dotenv
    parent_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
    load_dotenv(parent_env)
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logging.error("GOOGLE_API_KEY not found!")
        return
        
    client = genai.Client(api_key=api_key)
    # Test with gemini-live-2.5-flash-preview
    model = "gemini-live-2.5-flash-preview"
    
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction="Jawab singkat dengan sapaan.",
        input_audio_transcription=types.AudioTranscriptionConfig()
    )
    
    try:
        logging.info(f"Connecting to Gemini Live with model {model}...")
        async with client.aio.live.connect(model=model, config=config) as session:
            logging.info("Connected!")
            
            async def receive():
                async for response in session.receive():
                    logging.info(f"Received: {response}")
                    if response.server_content:
                        content = response.server_content
                        if content.output_transcription:
                            logging.info(f"Bot Transcription: {content.output_transcription.text}")
                        if content.turn_complete:
                            logging.info("--- TURN COMPLETE ---")
            
            rx_task = asyncio.create_task(receive())
            
            # Turn 1
            logging.info("--- Starting Turn 1 ---")
            await session.send_client_content(
                turns=types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="Halo")]
                ),
                turn_complete=True
            )
            await asyncio.sleep(10)
            
            # Turn 2
            logging.info("--- Starting Turn 2 ---")
            await session.send_client_content(
                turns=types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="Siapa namamu?")]
                ),
                turn_complete=True
            )
            await asyncio.sleep(10)
            
            rx_task.cancel()
            await rx_task
    except Exception as e:
        logging.exception(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_live_multiturn_2_5())
