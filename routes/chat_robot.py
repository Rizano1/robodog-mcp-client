import logging
import os
import requests
from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Request, HTTPException
from schemas.request import QuestionRequest
from services.chat_robot import ChatRobot
from config.config import Settings

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