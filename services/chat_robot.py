from pprint import pprint
from textwrap import dedent
from google import genai
from google.genai import types
import asyncio
import json
import os
import base64
import mimetypes
import io
import time
import httpx 
import docx 
from langfuse import Langfuse

from schemas.request import QuestionRequest
from fastmcp.client.transports import StreamableHttpTransport
from utils.tools_converter import convert_mcp_tools_to_gemini
from utils.prompt import system_prompt
from dotenv import load_dotenv
from config.config import Settings
from fastmcp import Client as FastMCPClient
from supabase import create_client, Client as SupabaseClient


class ChatRobot():

    def __init__(self):
        self.settings = Settings()
        load_dotenv()
        self.gemini_client = genai.Client(api_key=self.settings.google_key)
        self.supabase: SupabaseClient = create_client(self.settings.supabase_url, self.settings.supabase_key)
        self.langfuse = Langfuse()
        self.req = None

    # --- HISTORY MANAGEMENT ---

    def generate_session_title(self, session_id: str, user_prompt: str, bot_answer: str):
        """
        Membuat judul sesi berdasarkan konteks percakapan pertama menggunakan Gemini.
        Hanya berjalan jika title belum ada.
        """
        try:
            current = self.supabase.table("chat-sessions").select("title").eq("id", session_id).execute()
            
            if current.data and current.data[0].get('title'):
                print(f"   ℹ️ Session already has title: {current.data[0]['title']}")
                return

            print("   ✨ Generating smart title for this session...")

            title_prompt = (
                f"Berdasarkan percakapan berikut, buatkan judul sesi yang sangat singkat, "
                f"padat, dan deskriptif (maksimal 5 kata). Jangan gunakan tanda kutip.\n\n"
                f"User: {user_prompt}\n"
                f"Model: {bot_answer}"
            )

            # Langfuse: Buat trace dan generation untuk pembuatan judul
            trace = self.langfuse.trace(
                name="generate_session_title",
                session_id=str(session_id),
                input={"user_prompt": user_prompt, "bot_answer": bot_answer}
            )
            generation = trace.generation(
                name="gemini_title_generation",
                model="gemini-2.5-flash",
                input=title_prompt
            )

            resp = self.gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=title_prompt
            )
            
            new_title = resp.text.strip()
            generation.end(output=new_title)
            trace.update(output=new_title)
            
            self.supabase.table("chat-sessions").update({
                "title": new_title
            }).eq("id", session_id).execute()
            
            print(f"   🏷️ Title updated to: '{new_title}'")

        except Exception as e:
            print(f"   ⚠️ Failed to generate title: {e}")

    def create_history(self) -> str:
        """Membuat sesi baru di DB dan mengembalikan ID-nya."""
        res = self.supabase.table("chat-sessions").insert({}).execute()
        if res.data:
            new_id = res.data[0]['id']
            print(f"🆕 Session Created: {new_id}")
            return new_id
        return None

    def save_message(self, session_id: str, content: types.Content, showed: bool = True):
        """
        Menyimpan pesan ke DB.
        PENTING: Kita serialisasi objek Gemini ke JSON.
        Kita TIDAK menyimpan bytes file di sini.
        showed: True untuk user msg & final model text, False untuk function_call & function_response.
        """
        serialized_parts = []
        for part in content.parts:
            if part.text:
                serialized_parts.append({"text": part.text})
            elif part.function_call:
                serialized_parts.append({
                    "function_call": {
                        "name": part.function_call.name,
                        "args": dict(part.function_call.args)
                    }
                })
            elif part.function_response:
                serialized_parts.append({
                    "function_response": {
                        "name": part.function_response.name,
                        "response": part.function_response.response
                    }
                })

        self.supabase.table("chat-messages").insert({
            "session_id": session_id,
            "role": content.role,
            "content": serialized_parts,
            "showed": showed
        }).execute()

    def get_history(self, session_id: str) -> list:
        """
        Mengambil history dan melakukan RE-INJECTION file bytes jika diperlukan.
        """
        res = self.supabase.table("chat-messages").select("*")\
            .eq("session_id", session_id)\
            .order("created_at", desc=False)\
            .execute()
        
        gemini_messages = []

        for row in res.data:
            role = row['role']
            db_content = row['content'] 
            
            parts = []
            tool_response_payload = None 

            for item in db_content:
                if "text" in item:
                    parts.append(types.Part.from_text(text=item["text"]))
                elif "function_call" in item:
                    fc = item["function_call"]
                    parts.append(types.Part.from_function_call(name=fc["name"], args=fc["args"]))
                elif "function_response" in item:
                    fr = item["function_response"]
                    tool_response_payload = fr["response"] 
                    parts.append(types.Part.from_function_response(name=fr["name"], response=fr["response"]))

            gemini_messages.append(types.Content(role=role, parts=parts))

            if role == "tool" and tool_response_payload:
                msg_type = tool_response_payload.get("type")
                status = tool_response_payload.get("status")

                if msg_type == "file_retrieve" and status == "success":
                    data = tool_response_payload.get("data", {})
                    filename = data.get("filename")
                    folder = data.get("folder", "sop") 
                    
                    print(f"   📂 History Replay: Re-downloading '{filename}' for context...")
                    
                    file_injection_msg = self.download_file(filename, folder)
                    
                    if file_injection_msg:
                        gemini_messages.append(file_injection_msg)

                elif msg_type == "image_capture" and status == "success":
                    data = tool_response_payload.get("data", {})
                    filepath = data.get("filepath")
                    
                    print(f"   📸 History Replay: Re-downloading captured image '{filepath}' for context...")
                    
                    image_injection_msg = self.download_image(filepath)
                    
                    if image_injection_msg:
                        gemini_messages.append(image_injection_msg)

        return gemini_messages

    # --- TOOLS & HELPERS ---

    def download_file(self, file_name: str, folder_name: str) -> types.Content:
        """
        Mengunduh file. Jika PDF/Gambar -> kirim sebagai File Bytes.
        Jika DOCX -> ekstrak teksnya -> kirim sebagai Teks.
        """
        try:
            file_path = f"{folder_name}/{file_name}" if folder_name else file_name
            # 1. Download Bytes dari Supabase
            file_bytes = self.supabase.storage.from_(self.settings.bucket_name).download(file_path)
            
            if not file_bytes: return None

            # 2. Deteksi Mime Type
            mime_type, _ = mimetypes.guess_type(file_name)
            if mime_type is None:
                mime_type = "application/octet-stream"

            parts = []

            # 3. Cek apakah ini file DOCX (Word)
            if mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                try:
                    # Gunakan io.BytesIO agar library docx bisa membaca raw bytes seolah-olah file fisik
                    doc_stream = io.BytesIO(file_bytes)
                    doc = docx.Document(doc_stream)
                    
                    # Ekstrak semua paragraf menjadi satu string
                    full_text = [para.text for para in doc.paragraphs]
                    extracted_text = "\n".join(full_text)
                    
                    # Masukkan sebagai Part TEXT
                    parts.append(
                        types.Part.from_text(text=f"--- Content of {file_name} ---\n{extracted_text}")
                    )
                except Exception as e:
                    print(f"⚠️ Gagal parsing DOCX {file_name}: {e}")
                    # Fallback jika gagal parsing, kirim info error saja
                    parts.append(types.Part.from_text(text=f"Error reading docx content: {file_name}"))

            # 4. Jika BUKAN docx (misal: PDF, Gambar, Video), kirim sebagai Bytes (Native)
            else:
                parts.append(types.Part.from_bytes(data=file_bytes, mime_type=mime_type))
                # Tambahkan label nama file (opsional, agar AI tahu nama filenya)
                parts.append(types.Part.from_text(text=f"[System Injection] File uploaded: {file_name}"))

            return types.Content(
                role='user',
                parts=parts
            )

        except Exception as e:
            print(f"❌ Error downloading file for history: {e}")
            return None

    def download_image(self, filepath: str) -> types.Content:
        """
        Mengunduh gambar hasil capture dari Supabase Storage dan mengembalikannya
        sebagai Content dengan image bytes untuk di-inject ke Gemini context.
        """
        try:
            # Download bytes dari Supabase Storage
            file_bytes = self.supabase.storage.from_(self.settings.bucket_name).download(filepath)
            
            if not file_bytes:
                print(f"   ⚠️ Image file empty or not found: {filepath}")
                return None

            filename = filepath.split("/")[-1]

            parts = [
                types.Part.from_bytes(data=file_bytes, mime_type="image/jpeg"),
                types.Part.from_text(text=f"[System Injection] Captured image from robot camera: {filename}")
            ]

            return types.Content(
                role='user',
                parts=parts
            )

        except Exception as e:
            print(f"❌ Error downloading captured image: {e}")
            return None

    # --- MAIN PROCESS ---

    async def main(self, req: QuestionRequest):
        self.req = req
        return await self.process_chat()

    async def process_chat(self):
        transport = StreamableHttpTransport(url=self.settings.mcp_url)
        client = FastMCPClient(transport)

        session_id = self.req.session_id
        messages = []

        if not session_id:
            session_id = self.create_history()
        else:
            print(f"📜 Loading history for session: {session_id}")
            messages = self.get_history(session_id)

        user_msg = types.Content(role='user', parts=[types.Part.from_text(text=self.req.user_prompt)])
        messages.append(user_msg)
        
        self.save_message(session_id, user_msg)
        print(f"User: {self.req.user_prompt}\n")

        # Langfuse: Inisiasi Trace untuk satu alur percakapan
        trace = self.langfuse.trace(
            name="process_chat",
            session_id=str(session_id),
            input=self.req.user_prompt
        )

        async with client:
            await client.ping()
            tools_response = await client.list_tools()
            gemini_tools = convert_mcp_tools_to_gemini(tools_response)

            while True:
                # Langfuse: Catat iterasi ke Gemini
                generation = trace.generation(
                    name="gemini_generation",
                    model="gemini-2.5-pro",
                    input=f"[{len(messages)} messages in context]"
                )

                # Retry logic for transient network errors (DNS, connection)
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = self.gemini_client.models.generate_content(
                            model='gemini-2.5-pro', 
                            contents=messages,
                            config=types.GenerateContentConfig(
                                tools=gemini_tools,
                                system_instruction=system_prompt
                            ),
                        )
                        break  # Success, exit retry loop
                    except (httpx.ConnectError, httpx.TimeoutException) as e:
                        if attempt < max_retries - 1:
                            wait_time = 2 ** (attempt + 1)  # 2s, 4s, 8s
                            print(f"   ⚠️ Network error (attempt {attempt + 1}/{max_retries}): {e}")
                            print(f"   🔄 Retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                        else:
                            print(f"   ❌ Network error persisted after {max_retries} attempts")
                            generation.end(level="ERROR", status_message=str(e))
                            raise

                candidate = response.candidates[0]
                has_function_call = any(part.function_call for part in candidate.content.parts)

                # Langfuse: Selesaikan generation dengan metadata usage dan output
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    generation.end(
                        output="Function Call" if has_function_call else candidate.content.parts[0].text,
                        usage={
                            "input": getattr(response.usage_metadata, 'prompt_token_count', 0),
                            "output": getattr(response.usage_metadata, 'candidates_token_count', 0),
                            "total": getattr(response.usage_metadata, 'total_token_count', 0),
                        }
                    )
                else:
                    generation.end(output="Function Call" if has_function_call else getattr(candidate.content.parts[0], 'text', str(candidate.content.parts)))

                messages.append(candidate.content)
                
                # Cek apakah response berisi function_call (tidak ditampilkan di UI)
                self.save_message(session_id, candidate.content, showed=not has_function_call)

                found_tool_call = False
                
                if candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.function_call:
                            found_tool_call = True
                            tool_name = part.function_call.name
                            tool_args = dict(part.function_call.args) if part.function_call.args else {}
                            
                            # Inject session_id into every tool call
                            tool_args["session_id"] = str(session_id)
                            
                            print(f"🔧 Calling tool: {tool_name}({tool_args})")
                            
                            runtime_file_injection = None 
                            response_payload = {}

                            try:
                                result = await client.call_tool(tool_name, tool_args)
                                raw_output = result.content[0].text if hasattr(result, 'content') else str(result)
                                print(f"🔧 Result tool: {tool_name}: {raw_output}")
                                parsed_output = json.loads(raw_output)
                                response_payload = parsed_output

                                if isinstance(parsed_output, dict):
                                    msg_type = parsed_output.get("type")
                                    status = parsed_output.get("status")
                                    
                                    if msg_type == "file_retrieve" and status == "success":
                                        print("   📄 File detected. Downloading for current context...")
                                        
                                        data = parsed_output.get("data", {})
                                        folder = data.get("folder", "sop")
                                        filename = data.get("filename")
                                        
                                        runtime_file_injection = self.download_file(filename, folder)

                                    elif msg_type == "image_capture" and status == "success":
                                        print("   📸 Image captured. Downloading for current context...")
                                        
                                        data = parsed_output.get("data", {})
                                        filepath = data.get("filepath")
                                        
                                        runtime_file_injection = self.download_image(filepath)

                            except Exception as e:
                                print(f"   ❌ Error: {e}")
                                response_payload = {"error": str(e)}

                            tool_msg = types.Content(
                                role='tool',
                                parts=[types.Part.from_function_response(
                                    name=tool_name,
                                    response=response_payload
                                )]
                            )
                            messages.append(tool_msg)
                            self.save_message(session_id, tool_msg, showed=False) 
                            
           
                            if runtime_file_injection:
                                print("   📎 Injecting file bytes to Gemini context (Runtime)...")
                                messages.append(runtime_file_injection)

                if not found_tool_call:
                    if candidate.content.parts and candidate.content.parts[0].text:
                        print(f"\n✨ Final Response: {candidate.content.parts[0].text}")
                    break

        final_answer = messages[-1].parts[0].text
        
        self.generate_session_title(
            session_id=session_id, 
            user_prompt=self.req.user_prompt, 
            bot_answer=final_answer
        )

        trace.update(output=final_answer)

        # Ensure events are sent before returning
        self.langfuse.flush()

        return {
            "session_id": session_id,
            "answer": final_answer
        }