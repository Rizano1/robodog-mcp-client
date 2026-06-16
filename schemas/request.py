from pydantic import BaseModel
from typing import Optional, List

AVAILABLE_MODELS = [
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "qwen3.5:27b",
    "gpt-4o",
    "gpt-4o-mini",
]

DEFAULT_MODEL = "gemini-2.5-flash"

class QuestionRequest(BaseModel):
  session_id: Optional[int] = None
  user_prompt: str
  user_id: Optional[str] = None
  files: Optional[List[str]] = None
  system_prompt: Optional[str] = None
  model_name: Optional[str] = DEFAULT_MODEL
  tags: Optional[List[str]] = None
  system_instruction: Optional[str] = None