from pydantic import BaseModel
from typing import Optional, List

class QuestionRequest(BaseModel):
  session_id: Optional[int] = None
  user_prompt: str
  user_id: Optional[str] = None
  files: Optional[List[str]] = None  