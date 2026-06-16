import os
import sys
from dotenv import load_dotenv
from supabase import create_client

# Add parent dir to sys.path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Settings

load_dotenv(dotenv_path="../.env")
settings = Settings()
print("Supabase URL:", settings.supabase_url)

supabase = create_client(settings.supabase_url, settings.supabase_key)

# Query last 10 chat messages
res = supabase.table("chat-messages").select("*").order("created_at", desc=True).limit(10).execute()
for row in res.data:
    print(f"ID: {row['id']}, Session ID: {row['session_id']}, Role: {row['role']}")
    print("Content:")
    for part in row['content']:
        if "text" in part:
            print(f"  - text: {part['text'][:100]}")
        elif "inline_data" in part:
            print(f"  - inline_data: mime_type={part['inline_data'].get('mime_type')}, size={len(part['inline_data'].get('data', ''))}")
        else:
            print(f"  - other keys: {list(part.keys())}")
    print("-" * 50)
