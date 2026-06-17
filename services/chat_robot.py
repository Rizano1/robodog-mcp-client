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
import pypdf
from langfuse import observe, propagate_attributes, get_client


from schemas.request import QuestionRequest
from utils.prompt import system_prompt
from dotenv import load_dotenv
from config.config import Settings
from supabase import create_client, Client as SupabaseClient

# Models that use OpenAI-compatible API format
OLLAMA_MODELS = {"qwen3.5:27b"}
OPENAI_MODELS = {"gpt-4o", "gpt-4o-mini"}


class ChatRobot():

    def __init__(self):
        self.langfuse_client = get_client()
        self.settings = Settings()
        load_dotenv()
        self.gemini_client = genai.Client(api_key=self.settings.google_key)
        self.supabase: SupabaseClient = create_client(self.settings.supabase_url, self.settings.supabase_key)
        self.req = None

    # --- MODEL DETECTION ---

    def _is_openai_compatible(self, model_name: str) -> bool:
        """Check if the given model uses the OpenAI-compatible API (Ollama or OpenAI GPT)."""
        return model_name in OLLAMA_MODELS or model_name in OPENAI_MODELS

    def _model_supports_vision(self, model_name: str) -> bool:
        """Check if the given model supports multimodal vision input."""
        name_lower = model_name.lower()
        if "gemini" in name_lower:
            return True
        if "gpt-4" in name_lower:
            return True
        if "-vl" in name_lower or "llava" in name_lower:
            return True
        if "qwen3.5" in name_lower:
            return True
        return False

    def _get_openai_endpoint(self, model_name: str) -> tuple[str, dict]:
        """
        Returns (base_url, headers) for the given OpenAI-compatible model.
        - Ollama models → local Ollama server, no auth
        - OpenAI GPT models → api.openai.com, with Bearer token
        """
        if model_name in OPENAI_MODELS:
            return (
                "https://api.openai.com/v1/chat/completions",
                {
                    "Authorization": f"Bearer {self.settings.openai_api_key}",
                    "Content-Type": "application/json"
                }
            )
        else:
            # Ollama
            return (
                f"{self.settings.ollama_host}/v1/chat/completions",
                {"Content-Type": "application/json"}
            )

    # --- HISTORY MANAGEMENT ---

    @observe()
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

            resp = self.gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=title_prompt
            )
            
            new_title = resp.text.strip()
            
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
            elif hasattr(part, 'inline_data') and part.inline_data:
                b64_data = base64.b64encode(part.inline_data.data).decode("utf-8")
                serialized_parts.append({
                    "inline_data": {
                        "mime_type": part.inline_data.mime_type,
                        "data": b64_data
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
        Mengambil history tanpa function calling.
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

            for item in db_content:
                if "text" in item:
                    parts.append(types.Part.from_text(text=item["text"]))
                elif "inline_data" in item:
                    id_data = item["inline_data"]
                    file_bytes = base64.b64decode(id_data["data"])
                    parts.append(types.Part.from_bytes(data=file_bytes, mime_type=id_data["mime_type"]))

            if parts:
                gemini_messages.append(types.Content(role=role, parts=parts))

        return gemini_messages

    def get_history_for_ollama(self, session_id: str, model_name: str = "") -> list:
        """
        Mengambil history dari DB dan convert ke format OpenAI messages untuk Ollama.
        File re-injection dilakukan sebagai text description (Ollama tidak support raw bytes).
        """
        res = self.supabase.table("chat-messages").select("*")\
            .eq("session_id", session_id)\
            .order("created_at", desc=False)\
            .execute()
        
        ollama_messages = []
        supports_vision = self._model_supports_vision(model_name) if model_name else True

        for row in res.data:
            role = row['role']
            db_content = row['content']

            for item in db_content:
                if "text" in item:
                    # Map Gemini roles to OpenAI roles
                    openai_role = "assistant" if role == "model" else role
                    ollama_messages.append({
                        "role": openai_role,
                        "content": item["text"]
                    })

                elif "inline_data" in item:
                    id_data = item["inline_data"]
                    mime = id_data["mime_type"]
                    b64_data = id_data["data"]
                    file_bytes = base64.b64decode(b64_data)

                    if mime.startswith("image/"):
                        if supports_vision:
                            if ollama_messages and ollama_messages[-1]["role"] == "user":
                                prev_content = ollama_messages[-1]["content"]
                                if isinstance(prev_content, str):
                                    ollama_messages[-1]["content"] = [
                                        {"type": "text", "text": prev_content},
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:{mime};base64,{b64_data}"
                                            }
                                        }
                                    ]
                                elif isinstance(prev_content, list):
                                    ollama_messages[-1]["content"].append({
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{mime};base64,{b64_data}"
                                        }
                                    })
                            else:
                                ollama_messages.append({
                                    "role": "user",
                                    "content": [{
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{mime};base64,{b64_data}"
                                        }
                                    }]
                                })
                        else:
                            text_to_append = f"\n\n[System Injection] Image attached but vision is not supported by model '{model_name}'."
                            if ollama_messages and ollama_messages[-1]["role"] == "user":
                                if isinstance(ollama_messages[-1]["content"], str):
                                    ollama_messages[-1]["content"] += text_to_append
                                elif isinstance(ollama_messages[-1]["content"], list):
                                    ollama_messages[-1]["content"].append({"type": "text", "text": text_to_append})
                            else:
                                ollama_messages.append({
                                    "role": "user",
                                    "content": text_to_append
                                })
                    elif mime == "application/pdf":
                        try:
                            pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                            pages_text = []
                            for page in pdf_reader.pages:
                                page_text = page.extract_text()
                                if page_text:
                                    pages_text.append(page_text)
                            extracted = "\n\n".join(pages_text) if pages_text else "(No extractable text found in PDF)"
                        except Exception as e:
                            extracted = f"(Error extracting PDF text: {e})"
                        
                        text_to_append = f"\n\n[System Injection] Extracted PDF content:\n{extracted}"
                        if ollama_messages and ollama_messages[-1]["role"] == "user":
                            if isinstance(ollama_messages[-1]["content"], str):
                                ollama_messages[-1]["content"] += text_to_append
                            elif isinstance(ollama_messages[-1]["content"], list):
                                ollama_messages[-1]["content"].append({"type": "text", "text": text_to_append})
                        else:
                            ollama_messages.append({
                                "role": "user",
                                "content": text_to_append
                            })
                    elif mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        try:
                            doc_stream = io.BytesIO(file_bytes)
                            doc = docx.Document(doc_stream)
                            full_text = [para.text for para in doc.paragraphs]
                            extracted = "\n".join(full_text)
                        except Exception as e:
                            extracted = f"(Error extracting DOCX text: {e})"
                        
                        text_to_append = f"\n\n[System Injection] Extracted DOCX content:\n{extracted}"
                        if ollama_messages and ollama_messages[-1]["role"] == "user":
                            if isinstance(ollama_messages[-1]["content"], str):
                                ollama_messages[-1]["content"] += text_to_append
                            elif isinstance(ollama_messages[-1]["content"], list):
                                ollama_messages[-1]["content"].append({"type": "text", "text": text_to_append})
                        else:
                            ollama_messages.append({
                                "role": "user",
                                "content": text_to_append
                            })
                    else:
                        text_to_append = f"\n\n[System Injection] Binary file attached ({mime}, {file_bytes} bytes). Cannot display content."
                        if ollama_messages and ollama_messages[-1]["role"] == "user":
                            if isinstance(ollama_messages[-1]["content"], str):
                                ollama_messages[-1]["content"] += text_to_append
                            elif isinstance(ollama_messages[-1]["content"], list):
                                ollama_messages[-1]["content"].append({"type": "text", "text": text_to_append})
                        else:
                            ollama_messages.append({
                                "role": "user",
                                "content": text_to_append
                            })

        return ollama_messages

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
            print(f"   📥 Downloaded file '{file_path}' ({len(file_bytes)} bytes)")
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

    def download_file_from_url(self, url: str, object_name: str = "unknown") -> types.Content:
        """
        Mengunduh file SOP dari URL lokal. Jika PDF/Gambar -> kirim sebagai File Bytes.
        Jika DOCX -> ekstrak teksnya -> kirim sebagai Teks.
        URL bersifat lokal dan tidak bisa diakses langsung oleh LLM.
        """
        try:
            import requests as req_lib

            print(f"   📥 Downloading SOP from URL: {url}")
            resp = req_lib.get(url, timeout=30)
            resp.raise_for_status()
            file_bytes = resp.content

            if not file_bytes:
                print(f"   ⚠️ File empty from URL: {url}")
                return None

            print(f"   📥 Downloaded SOP for '{object_name}' ({len(file_bytes)} bytes)")

            # Detect mime type from Content-Type header or URL
            content_type = resp.headers.get("Content-Type", "")
            if ";" in content_type:
                content_type = content_type.split(";")[0].strip()

            if not content_type or content_type == "application/octet-stream":
                # Fallback: guess from URL
                mime_type, _ = mimetypes.guess_type(url)
                if mime_type:
                    content_type = mime_type
                else:
                    content_type = "application/octet-stream"

            parts = []

            # DOCX -> extract text
            if content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                try:
                    doc_stream = io.BytesIO(file_bytes)
                    doc = docx.Document(doc_stream)
                    full_text = [para.text for para in doc.paragraphs]
                    extracted_text = "\n".join(full_text)
                    parts.append(
                        types.Part.from_text(text=f"--- SOP Content for '{object_name}' ---\n{extracted_text}")
                    )
                except Exception as e:
                    print(f"   ⚠️ Gagal parsing DOCX dari URL: {e}")
                    parts.append(types.Part.from_text(text=f"Error reading docx content from URL for '{object_name}'"))

            # PDF, Image, etc -> send as bytes
            else:
                parts.append(types.Part.from_bytes(data=file_bytes, mime_type=content_type))
                parts.append(types.Part.from_text(text=f"[System Injection] SOP file for object '{object_name}'"))

            return types.Content(
                role='user',
                parts=parts
            )

        except Exception as e:
            print(f"❌ Error downloading SOP from URL '{url}': {e}")
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

    def _parse_files(self, files: list[str]) -> list[tuple[bytes, str, str]]:
        """
        Parses a list of data URLs.
        Returns a list of tuples: (file_bytes, mime_type, filename)
        """
        parsed = []
        import re
        for file_data in files:
            if not file_data:
                continue
            # Check if it is a data URL: data:<mime>;base64,<data>
            match = re.match(r'^data:([^;]+);base64,(.*)$', file_data)
            if match:
                mime_type = match.group(1)
                base64_data = match.group(2)
                try:
                    file_bytes = base64.b64decode(base64_data)
                    ext = mimetypes.guess_extension(mime_type) or '.bin'
                    filename = f"uploaded_file{ext}"
                    parsed.append((file_bytes, mime_type, filename))
                except Exception as e:
                    print(f"⚠️ Failed to decode base64 file: {e}")
            else:
                print(f"⚠️ File data is not a base64 data URL: {file_data[:50]}...")
        return parsed

    def _build_gemini_user_message(self, user_prompt: str, parsed_files: list[tuple[bytes, str, str]]) -> types.Content:
        parts = [types.Part.from_text(text=user_prompt)]
        for file_bytes, mime_type, filename in parsed_files:
            if mime_type.startswith("image/") or mime_type == "application/pdf":
                parts.append(types.Part.from_bytes(data=file_bytes, mime_type=mime_type))
            elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                # DOCX extraction
                try:
                    doc_stream = io.BytesIO(file_bytes)
                    doc = docx.Document(doc_stream)
                    full_text = [para.text for para in doc.paragraphs]
                    extracted_text = "\n".join(full_text)
                    parts.append(types.Part.from_text(text=f"--- Content of {filename} ---\n{extracted_text}"))
                except Exception as e:
                    print(f"⚠️ Failed to parse DOCX {filename}: {e}")
                    parts.append(types.Part.from_text(text=f"Error reading docx content: {filename}"))
            else:
                parts.append(types.Part.from_bytes(data=file_bytes, mime_type=mime_type))
        return types.Content(role='user', parts=parts)

    def _build_openai_user_message(self, user_prompt: str, parsed_files: list[tuple[bytes, str, str]], model_name: str = "") -> dict:
        content_array = [{"type": "text", "text": user_prompt}]
        supports_vision = self._model_supports_vision(model_name) if model_name else True
        for file_bytes, mime_type, filename in parsed_files:
            b64_data = base64.b64encode(file_bytes).decode("utf-8")
            if mime_type.startswith("image/"):
                if supports_vision:
                    content_array.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{b64_data}"
                        }
                    })
                else:
                    content_array.append({
                        "type": "text",
                        "text": f"\n\n[System Injection] Image attached ({filename}) but vision is not supported by model '{model_name}'."
                    })
            elif mime_type == "application/pdf":
                try:
                    pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                    pages_text = []
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            pages_text.append(page_text)
                    extracted = "\n\n".join(pages_text) if pages_text else "(No extractable text found in PDF)"
                except Exception as e:
                    extracted = f"(Error extracting PDF text: {e})"
                content_array.append({
                    "type": "text",
                    "text": f"\n\n[System Injection] Extracted PDF content ({filename}):\n{extracted}"
                })
            elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                # DOCX extraction for Ollama/OpenAI
                try:
                    doc_stream = io.BytesIO(file_bytes)
                    doc = docx.Document(doc_stream)
                    full_text = [para.text for para in doc.paragraphs]
                    extracted = "\n".join(full_text)
                except Exception as e:
                    extracted = f"(Error extracting DOCX text: {e})"
                content_array.append({
                    "type": "text",
                    "text": f"\n\n[System Injection] Extracted DOCX content ({filename}):\n{extracted}"
                })
            else:
                content_array.append({
                    "type": "text",
                    "text": f"\n\n[System Injection] Binary file attached ({mime_type}, {len(file_bytes)} bytes). Cannot display content."
                })
        return {"role": "user", "content": content_array}

    # --- MAIN PROCESS ---



    @observe(as_type="generation")
    async def _call_gemini(self, messages, gemini_tools=None, index=0):
        model_name = (self.req.model_name or "gemini-2.5-flash") if self.req else "gemini-2.5-flash"
        # Log input and model before making the call
        self.langfuse_client.update_current_generation(
            input=f"[{len(messages)} messages context]",
            model=model_name,
            metadata={
                "tags": self.req.tags + [f"{index}"] if self.req and self.req.tags else [f"{index}"]
            }
        )

        max_retries = 3
        base_sys_instruction = self.req.system_instruction if (self.req and hasattr(self.req, 'system_instruction') and self.req.system_instruction) else system_prompt
        if self.req and self.req.system_prompt:
            sys_instruction = f"{base_sys_instruction}\n\n[Robot Status]\n{self.req.system_prompt}"
        else:
            sys_instruction = base_sys_instruction

        for attempt in range(max_retries):
            try:
                response = self.gemini_client.models.generate_content(
                    model=model_name,
                    contents=messages,
                    config=types.GenerateContentConfig(
                        tools=None,
                        system_instruction=sys_instruction
                    ),
                )
                
                # Trace token usage
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    self.langfuse_client.update_current_generation(
                        usage_details={
                            "input": getattr(response.usage_metadata, 'prompt_token_count', 0),
                            "output": getattr(response.usage_metadata, 'candidates_token_count', 0)
                        }
                    )
                
                return response
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)  # 2s, 4s, 8s
                    print(f"   ⚠️ Network error (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"   🔄 Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"   ❌ Network error persisted after {max_retries} attempts")
                    raise

    @observe(as_type="generation")
    async def _call_openai_compatible(self, messages: list[dict], tools: list[dict], index) -> dict:
        """
        Call an OpenAI-compatible endpoint (Ollama or OpenAI GPT).
        Automatically routes to the correct URL and auth based on model name.

        Args:
            messages: List of OpenAI-format message dicts.
            tools: List of OpenAI-format tool definitions.

        Returns:
            The response JSON dict.
        """
        model_name = (self.req.model_name or "qwen3.5:27b") if self.req else "qwen3.5:27b"
        
        self.langfuse_client.update_current_generation(
            input=f"[{len(messages)} messages context]",
            model=model_name,
            metadata={
                "tags": self.req.tags + [f"{index}"] if self.req and self.req.tags else [f"{index}"]
            }
        )

        url, headers = self._get_openai_endpoint(model_name)
        provider = "OpenAI" if model_name in OPENAI_MODELS else "Ollama"
        
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.7,
        }
        # Only include tools if there are any
        if tools:
            payload["tools"] = tools

        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=180.0) as http_client:
                    resp = await http_client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                    result = resp.json()

                # Trace token usage
                usage = result.get("usage", {})
                if usage:
                    self.langfuse_client.update_current_generation(
                        usage_details={
                            "input": usage.get("prompt_tokens", 0),
                            "output": usage.get("completion_tokens", 0)
                        }
                    )

                return result
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    print(f"   ⚠️ {provider} network error (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"   🔄 Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"   ❌ {provider} network error persisted after {max_retries} attempts")
                    raise

    @observe()
    async def main(self, req: QuestionRequest):
        self.req = req
        return await self.process_chat()

    # --- GEMINI FLOW ---

    @observe()
    async def process_chat(self):
        model_name = (self.req.model_name or "gemini-2.5-flash") if self.req else "gemini-2.5-flash"

        # Route to OpenAI-compatible flow if the model is Ollama or GPT
        if self._is_openai_compatible(model_name):
            return await self.process_chat_openai_compatible()

        session_id = self.req.session_id
        messages = []

        if not session_id:
            session_id = self.create_history()
        else:
            print(f"📜 Loading history for session: {session_id}")
            messages = self.get_history(session_id)

        with propagate_attributes(session_id=str(session_id), tags=self.req.tags if self.req.tags else None):
            parsed_files = self._parse_files(self.req.files) if self.req.files else []
            user_msg = self._build_gemini_user_message(self.req.user_prompt, parsed_files)
            messages.append(user_msg)
            self.save_message(session_id, user_msg)
            
            print(f"User: {self.req.user_prompt}\n")

            index = 0
            while True:
                response = await self._call_gemini(messages, None, index)
                index += 1
                candidate = response.candidates[0]
                
                # Guard: Gemini may return None parts (safety block, empty response)
                if not candidate.content or not candidate.content.parts:
                    print("   ⚠️ Gemini returned empty response (no parts). Retrying...")
                    # Append a nudge so Gemini knows it needs to respond
                    messages.append(types.Content(
                        role='user',
                        parts=[types.Part.from_text(text="[System] Your previous response was empty. Please try again.")]
                    ))
                    continue

                messages.append(candidate.content)
                self.save_message(session_id, candidate.content, showed=True)
                break

            final_answer = messages[-1].parts[0].text if messages[-1].parts else ""
            
            self.generate_session_title(
                session_id=session_id, 
                user_prompt=self.req.user_prompt, 
                bot_answer=final_answer
            )

            return {
                "session_id": session_id,
                "answer": final_answer
            }

    # --- OPENAI-COMPATIBLE FLOW (Ollama / GPT) ---

    @observe()
    async def process_chat_openai_compatible(self):
        """
        Agentic loop for OpenAI-compatible models (Ollama Qwen, OpenAI GPT, etc.) without function calling.
        """
        model_name = (self.req.model_name or "qwen3.5:27b") if self.req else "qwen3.5:27b"
        provider = "OpenAI" if model_name in OPENAI_MODELS else "Ollama"

        session_id = self.req.session_id
        openai_messages = []

        if not session_id:
            session_id = self.create_history()
        else:
            print(f"📜 Loading history for session ({provider}): {session_id}")
            openai_messages = self.get_history_for_ollama(session_id)

        with propagate_attributes(session_id=str(session_id), tags=self.req.tags if self.req.tags else None):
            # Inject dynamic robot status to system prompt
            base_sys_instruction = self.req.system_instruction if (self.req and hasattr(self.req, 'system_instruction') and self.req.system_instruction) else system_prompt
            if self.req and self.req.system_prompt:
                sys_instruction = f"{base_sys_instruction}\n\n[Robot Status]\n{self.req.system_prompt}"
            else:
                sys_instruction = base_sys_instruction

            # Add system prompt as first message (OpenAI format)
            # Insert at position 0 so it's always first
            openai_messages.insert(0, {
                "role": "system",
                "content": sys_instruction
            })

            # Add user message with files
            parsed_files = self._parse_files(self.req.files) if self.req.files else []
            user_openai_msg = self._build_openai_user_message(self.req.user_prompt, parsed_files)
            openai_messages.append(user_openai_msg)

            # Save user message to DB in Gemini format
            user_gemini_msg = self._build_gemini_user_message(self.req.user_prompt, parsed_files)
            self.save_message(session_id, user_gemini_msg)

            print(f"User: {self.req.user_prompt}\n")

            index = 0
            while True:
                response = await self._call_openai_compatible(openai_messages, None, index)
                index += 1
                choice = response.get("choices", [{}])[0]
                message = choice.get("message", {})

                assistant_content = message.get("content", None)

                # Guard: empty response
                if not assistant_content:
                    print(f"   ⚠️ {provider} returned empty response. Retrying...")
                    openai_messages.append({
                        "role": "user",
                        "content": "[System] Your previous response was empty. Please try again."
                    })
                    continue

                # Append assistant message
                openai_messages.append({
                    "role": "assistant",
                    "content": assistant_content
                })

                # Save final assistant text to DB in Gemini format
                gemini_final = types.Content(
                    role='model',
                    parts=[types.Part.from_text(text=assistant_content)]
                )
                self.save_message(session_id, gemini_final, showed=True)
                break

            final_answer = assistant_content or ""

            self.generate_session_title(
                session_id=session_id,
                user_prompt=self.req.user_prompt,
                bot_answer=final_answer
            )

            return {
                "session_id": session_id,
                "answer": final_answer
            }