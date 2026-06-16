import os
import sys
import base64
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.chat_robot import ChatRobot
from google.genai import types

load_dotenv(dotenv_path="../.env")

chat = ChatRobot()

# Create a test session
session_id = chat.create_history()
print("Created test session:", session_id)

# Create a user message with a dummy file part
parsed_files = [
    (b"dummy_image_data_here_123", "image/png", "test.png")
]
user_msg = chat._build_gemini_user_message("Ini gambar pengujian", parsed_files)

print("Saving message to session:", session_id)
chat.save_message(session_id, user_msg)

# Retrieve the message from the database to see what was saved
res = chat.supabase.table("chat-messages").select("*").eq("session_id", session_id).execute()
for row in res.data:
    print(f"Role: {row['role']}, Content: {row['content']}")

# Clean up test messages and session
chat.supabase.table("chat-messages").delete().eq("session_id", session_id).execute()
chat.supabase.table("chat-sessions").delete().eq("id", session_id).execute()
print("Cleaned up test data.")
