import os
from dotenv import load_dotenv
load_dotenv()

class Settings:
    def __init__(self):
        self.google_key = os.getenv('GOOGLE_API_KEY')
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", 8000))
        self.log_level = os.getenv("LOG_LEVEL", "warning")
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        self.mcp_url = os.getenv("MCP_URL")
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.bucket_name = "robotics-prata"



settings = Settings()