import asyncio
import logging
import os
import sys
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)

async def test_live():
    # Load API key using standard python-dotenv or manually from the parent .env
    from dotenv import load_dotenv
    # Find .env in parent folder
    parent_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
    logging.info(f"Loading .env from {parent_env}")
    load_dotenv(parent_env)
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logging.error("GOOGLE_API_KEY not found in environment!")
        return
        
    logging.info(f"API Key found (length {len(api_key)})")
    
    client = genai.Client(api_key=api_key)
    model = "gemini-3.1-flash-live-preview"
    
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],  # gemini-3.1-flash-live-preview requires AUDIO
        system_instruction="Jawab singkat dengan kata 'Halo, saya Raisa!'",
    )
    
    try:
        logging.info("Connecting to Gemini Live...")
        async with client.aio.live.connect(model=model, config=config) as session:
            logging.info("Connected successfully!")
            
            # Send text turn using the modern send_client_content
            logging.info("Sending message...")
            await session.send_client_content(
                turns=types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="Halo")]
                ),
                turn_complete=True
            )
            
            logging.info("Awaiting response...")
            async for response in session.receive():
                logging.info(f"Received from Gemini: {response}")
                if response.server_content:
                    model_turn = response.server_content.model_turn
                    if model_turn:
                        for part in model_turn.parts:
                            if part.text:
                                logging.info(f"Bot response text: {part.text}")
                    if response.server_content.turn_complete:
                        logging.info("Turn complete!")
                        break
    except Exception as e:
        logging.exception(f"Error during live connection: {e}")

if __name__ == "__main__":
    asyncio.run(test_live())
