import os
import sys
import uvicorn
import logging
import asyncio
import config.logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware # Import the CORS middleware
from config.config import Settings
from routes.chat_robot import chat_router

settings = Settings()

app = FastAPI()

# Configure CORS Middleware to allow requests from the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Allow the Next.js origin
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods (POST, GET, OPTIONS, etc.)
    allow_headers=["*"], # Allow all headers
)

app.include_router(chat_router, prefix="/api/chat_robot", tags=['ChatRobot'])

async def start_server():
    host = settings.host
    port = settings.port
    log_level = settings.log_level

    logging.info(f"Starting AI-LLM Robodog Client on {host}:{port}...") 

    config = uvicorn.Config(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_level=log_level,
        workers=4,
    )

    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    # Check the operating system and select an appropriate event loop
    platform = sys.platform
    if platform.startswith('linux'):
        import uvloop
        logging.info(f"=== [Running on {platform} with uvloop as event loop] ===")
        uvloop.run(start_server())
    else:
        logging.info(f"=== [Running on {platform} with asyncio as event loop] ===")
        asyncio.run(start_server())