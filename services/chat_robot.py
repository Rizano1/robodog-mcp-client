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
from langfuse import observe, propagate_attributes, get_client


from schemas.request import QuestionRequest
from fastmcp.client.transports import StreamableHttpTransport
from utils.tools_converter import convert_mcp_tools_to_gemini, convert_mcp_tools_to_ollama
from utils.prompt import system_prompt
from dotenv import load_dotenv
from config.config import Settings
from fastmcp import Client as FastMCPClient
from supabase import create_client, Client as SupabaseClient

# Models that use OpenAI-compatible API format
OLLAMA_MODELS = {"qwen2.5:7b"}
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

    def get_history_for_ollama(self, session_id: str) -> list:
        """
        Mengambil history dari DB dan convert ke format OpenAI messages untuk Ollama.
        File re-injection dilakukan sebagai text description (Ollama tidak support raw bytes).
        """
        res = self.supabase.table("chat-messages").select("*")\
            .eq("session_id", session_id)\
            .order("created_at", desc=False)\
            .execute()
        
        ollama_messages = []

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
                elif "function_call" in item:
                    fc = item["function_call"]
                    # Reconstruct as assistant message with tool_calls
                    ollama_messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": f"call_{fc['name']}",
                            "type": "function",
                            "function": {
                                "name": fc["name"],
                                "arguments": json.dumps(fc["args"])
                            }
                        }]
                    })
                elif "function_response" in item:
                    fr = item["function_response"]
                    ollama_messages.append({
                        "role": "tool",
                        "tool_call_id": f"call_{fr['name']}",
                        "content": json.dumps(fr["response"])
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

    @observe()
    async def _fetch_mcp_tools(self, client):
        await client.ping()
        tools_response = await client.list_tools()
        return tools_response

    @observe(as_type="generation")
    async def _call_gemini(self, messages, gemini_tools):
        model_name = (self.req.model_name or "gemini-2.5-flash") if self.req else "gemini-2.5-flash"
        # Log input and model before making the call
        self.langfuse_client.update_current_generation(
            input=f"[{len(messages)} messages context]",
            model=model_name
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.gemini_client.models.generate_content(
                    model=model_name,
                    contents=messages,
                    config=types.GenerateContentConfig(
                        tools=gemini_tools,
                        system_instruction=system_prompt
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
    async def _call_openai_compatible(self, messages: list[dict], tools: list[dict]) -> dict:
        """
        Call an OpenAI-compatible endpoint (Ollama or OpenAI GPT).
        Automatically routes to the correct URL and auth based on model name.

        Args:
            messages: List of OpenAI-format message dicts.
            tools: List of OpenAI-format tool definitions.

        Returns:
            The response JSON dict.
        """
        model_name = (self.req.model_name or "qwen2.5:7b") if self.req else "qwen2.5:7b"
        
        self.langfuse_client.update_current_generation(
            input=f"[{len(messages)} messages context]",
            model=model_name
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
                async with httpx.AsyncClient(timeout=120.0) as http_client:
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
    async def _process_tool_call(self, client, tool_name, tool_args, session_id):
        runtime_file_injection = None 
        is_async_running = False
        response_payload = {}
        metadata = {}
        metadata["session_id"] = str(session_id)
        metadata["trace_id"] = self.langfuse_client.get_current_trace_id()
        metadata["observation_id"] = self.langfuse_client.get_current_observation_id()
        metadata["model_name"] = (self.req.model_name or "gemini-2.5-flash") if self.req else "gemini-2.5-flash"
        
        print(f"🔧 Calling tool: {tool_name}({tool_args})")
        
        try:
            result = await client.call_tool(name=tool_name, arguments=tool_args,meta=metadata)
            raw_output = result.content[0].text if hasattr(result, 'content') else str(result)
            print(f"🔧 Result tool: {tool_name}: {raw_output}")
            parsed_output = json.loads(raw_output)
            response_payload = parsed_output

            if isinstance(parsed_output, dict):
                msg_type = parsed_output.get("type")
                status = parsed_output.get("status")
                
                # Detect async tools that are still running
                if status == "running":
                    is_async_running = True
                    print(f"   ⏳ Tool '{tool_name}' is async (status=running). Will break loop.")
                
                if msg_type == "file_retrieve" and status == "success":
                    print("   📄 File detected. Downloading for current context...")
                    data = parsed_output.get("data", {})
                    folder = data.get("folder", "sop")
                    filename = data.get("filename")
                    runtime_file_injection = self.download_file(filename, folder)

                # elif msg_type == "image_capture" and status == "success":
                #     print("   📸 Image captured. Downloading for current context...")
                #     data = parsed_output.get("data", {})
                #     filepath = data.get("filepath")
                #     runtime_file_injection = self.download_image(filepath)

        except Exception as e:
            print(f"   ❌ Error: {e}")
            response_payload = {"error": str(e)}

        return tool_name, tool_args, response_payload, runtime_file_injection, is_async_running

    def _build_gemini_tool_msg(self, tool_name, response_payload):
        """Build a Gemini-format tool response Content object."""
        return types.Content(
            role='tool',
            parts=[types.Part.from_function_response(
                name=tool_name,
                response=response_payload
            )]
        )

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

        transport = StreamableHttpTransport(url=self.settings.mcp_url)
        client = FastMCPClient(transport)

        session_id = self.req.session_id
        messages = []

        if not session_id:
            session_id = self.create_history()
        else:
            print(f"📜 Loading history for session: {session_id}")
            messages = self.get_history(session_id)

        with propagate_attributes(session_id=str(session_id)):
            user_msg = types.Content(role='user', parts=[types.Part.from_text(text=self.req.user_prompt)])
            messages.append(user_msg)
            self.save_message(session_id, user_msg)

            # Inject robot status as a function_call/function_response pair
            # so Gemini treats it as tool output (high attention) not user command
            # Must come AFTER user turn (Gemini requires function_call after user or function_response turn)
            if self.req.system_prompt:
                status_call_msg = types.Content(
                    role='model',
                    parts=[types.Part.from_function_call(
                        name='get_robot_status',
                        args={}
                    )]
                )
                status_response_msg = types.Content(
                    role='tool',
                    parts=[types.Part.from_function_response(
                        name='get_robot_status',
                        response={"status": self.req.system_prompt}
                    )]
                )
                messages.append(status_call_msg)
                messages.append(status_response_msg)
                self.save_message(session_id, status_call_msg, showed=False)
                self.save_message(session_id, status_response_msg, showed=False)
            
            print(f"User: {self.req.user_prompt}\n")

            async with client:
                mcp_tools_raw = await self._fetch_mcp_tools(client)
                gemini_tools = convert_mcp_tools_to_gemini(mcp_tools_raw)

                while True:
                    response = await self._call_gemini(messages, gemini_tools)
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
                    
                    # Cek apakah response berisi function_call (tidak ditampilkan di UI)
                    has_function_call = any(part.function_call for part in candidate.content.parts)
                    self.save_message(session_id, candidate.content, showed=not has_function_call)

                    found_tool_call = False
                    hit_async_running = False
                    
                    for part in candidate.content.parts:
                        if part.function_call:
                            found_tool_call = True
                            tool_name = part.function_call.name
                            tool_args = dict(part.function_call.args) if part.function_call.args else {}
                            
                            tool_name, tool_args, response_payload, runtime_file_injection, is_async_running = await self._process_tool_call(
                                client=client, 
                                tool_name=tool_name, 
                                tool_args=tool_args, 
                                session_id=session_id
                            )
                            
                            tool_msg = self._build_gemini_tool_msg(tool_name, response_payload)
                            messages.append(tool_msg)
                            self.save_message(session_id, tool_msg, showed=False) 
           
                            if runtime_file_injection:
                                print("   📎 Injecting file bytes to Gemini context (Runtime)...")
                                messages.append(runtime_file_injection)
                            
                            if is_async_running:
                                hit_async_running = True
                                break  # Stop processing further tool calls in this turn

                    # If an async tool is running, break loop and wait for robot webhook callback
                    if hit_async_running:
                        # Extract any text the model said before the tool call (e.g. "Robot sedang menuju...")
                        model_text_parts = [p.text for p in candidate.content.parts if p.text]
                        waiting_msg = model_text_parts[0] if model_text_parts else "Robot sedang menjalankan perintah..."
                        print(f"   ⏳ Async tool running. Breaking loop. Waiting for robot webhook callback.")
                        
                        self.generate_session_title(
                            session_id=session_id,
                            user_prompt=self.req.user_prompt,
                            bot_answer=waiting_msg
                        )

                        return {
                            "session_id": session_id,
                            "answer": waiting_msg
                        }

                    if not found_tool_call:
                        if candidate.content.parts[0].text:
                            print(f"\n✨ Final Response: {candidate.content.parts[0].text}")
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
        Agentic loop for OpenAI-compatible models (Ollama Qwen, OpenAI GPT, etc.).
        Handles tool injection and tool result in OpenAI message format,
        but persists history in Gemini format for DB consistency.
        """
        model_name = (self.req.model_name or "qwen2.5:7b") if self.req else "qwen2.5:7b"
        provider = "OpenAI" if model_name in OPENAI_MODELS else "Ollama"

        transport = StreamableHttpTransport(url=self.settings.mcp_url)
        client = FastMCPClient(transport)

        session_id = self.req.session_id
        openai_messages = []

        if not session_id:
            session_id = self.create_history()
        else:
            print(f"📜 Loading history for session ({provider}): {session_id}")
            openai_messages = self.get_history_for_ollama(session_id)

        with propagate_attributes(session_id=str(session_id)):
            # Add system prompt as first message (OpenAI format)
            # Insert at position 0 so it's always first
            openai_messages.insert(0, {
                "role": "system",
                "content": system_prompt
            })

            # Add user message
            user_openai_msg = {"role": "user", "content": self.req.user_prompt}
            openai_messages.append(user_openai_msg)

            # Save user message to DB in Gemini format
            user_gemini_msg = types.Content(role='user', parts=[types.Part.from_text(text=self.req.user_prompt)])
            self.save_message(session_id, user_gemini_msg)

            # Inject robot status if provided (as assistant + tool message pair)
            if self.req.system_prompt:
                # Assistant message: simulates tool call
                status_call_id = "call_get_robot_status"
                status_assistant_msg = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": status_call_id,
                        "type": "function",
                        "function": {
                            "name": "get_robot_status",
                            "arguments": "{}"
                        }
                    }]
                }
                # Tool response message
                status_tool_msg = {
                    "role": "tool",
                    "tool_call_id": status_call_id,
                    "content": json.dumps({"status": self.req.system_prompt})
                }
                openai_messages.append(status_assistant_msg)
                openai_messages.append(status_tool_msg)

                # Also save to DB in Gemini format
                gemini_call = types.Content(
                    role='model',
                    parts=[types.Part.from_function_call(name='get_robot_status', args={})]
                )
                gemini_resp = types.Content(
                    role='tool',
                    parts=[types.Part.from_function_response(
                        name='get_robot_status',
                        response={"status": self.req.system_prompt}
                    )]
                )
                self.save_message(session_id, gemini_call, showed=False)
                self.save_message(session_id, gemini_resp, showed=False)

            print(f"User: {self.req.user_prompt}\n")

            async with client:
                mcp_tools_raw = await self._fetch_mcp_tools(client)
                openai_tools = convert_mcp_tools_to_ollama(mcp_tools_raw)

                while True:
                    response = await self._call_openai_compatible(openai_messages, openai_tools)

                    choice = response.get("choices", [{}])[0]
                    message = choice.get("message", {})
                    finish_reason = choice.get("finish_reason", "")

                    assistant_content = message.get("content", None)
                    tool_calls = message.get("tool_calls", None)

                    # Guard: empty response
                    if not assistant_content and not tool_calls:
                        print(f"   ⚠️ {provider} returned empty response. Retrying...")
                        openai_messages.append({
                            "role": "user",
                            "content": "[System] Your previous response was empty. Please try again."
                        })
                        continue

                    # Append the full assistant message to conversation (including any tool_calls)
                    assistant_msg_for_history = {"role": "assistant"}
                    if assistant_content:
                        assistant_msg_for_history["content"] = assistant_content
                    else:
                        assistant_msg_for_history["content"] = None
                    if tool_calls:
                        assistant_msg_for_history["tool_calls"] = tool_calls
                    openai_messages.append(assistant_msg_for_history)

                    # Handle tool calls
                    if tool_calls:
                        # Save the assistant message with function_call(s) to DB (Gemini format)
                        gemini_parts = []
                        if assistant_content:
                            gemini_parts.append(types.Part.from_text(text=assistant_content))
                        for tc in tool_calls:
                            func = tc.get("function", {})
                            tc_name = func.get("name", "")
                            tc_args_str = func.get("arguments", "{}")
                            try:
                                tc_args = json.loads(tc_args_str) if isinstance(tc_args_str, str) else tc_args_str
                            except json.JSONDecodeError:
                                tc_args = {}
                            gemini_parts.append(types.Part.from_function_call(name=tc_name, args=tc_args))

                        gemini_assistant_content = types.Content(role='model', parts=gemini_parts)
                        self.save_message(session_id, gemini_assistant_content, showed=False)

                        hit_async_running = False

                        for tc in tool_calls:
                            func = tc.get("function", {})
                            tool_call_id = tc.get("id", "")
                            tool_name = func.get("name", "")
                            tool_args_str = func.get("arguments", "{}")

                            try:
                                tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
                            except json.JSONDecodeError:
                                tool_args = {}

                            tool_name, tool_args, response_payload, runtime_file_injection, is_async_running = await self._process_tool_call(
                                client=client,
                                tool_name=tool_name,
                                tool_args=tool_args,
                                session_id=session_id
                            )

                            # Append tool result in OpenAI format
                            tool_result_msg = {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": json.dumps(response_payload)
                            }
                            openai_messages.append(tool_result_msg)

                            # Save to DB in Gemini format
                            gemini_tool_msg = self._build_gemini_tool_msg(tool_name, response_payload)
                            self.save_message(session_id, gemini_tool_msg, showed=False)

                            # File injection: inject as text description
                            # (OpenAI-compatible API doesn't support raw file bytes like Gemini)
                            if runtime_file_injection:
                                print(f"   📎 Injecting file content to {provider} context (Runtime)...")
                                # Extract text parts from the Gemini file injection
                                file_texts = [p.text for p in runtime_file_injection.parts if p.text]
                                if file_texts:
                                    file_injection_openai = {
                                        "role": "user",
                                        "content": "\n".join(file_texts)
                                    }
                                    openai_messages.append(file_injection_openai)

                            if is_async_running:
                                hit_async_running = True
                                break

                        # Handle async tool
                        if hit_async_running:
                            waiting_msg = assistant_content if assistant_content else "Robot sedang menjalankan perintah..."
                            print(f"   ⏳ Async tool running. Breaking loop ({provider}).")

                            self.generate_session_title(
                                session_id=session_id,
                                user_prompt=self.req.user_prompt,
                                bot_answer=waiting_msg
                            )

                            return {
                                "session_id": session_id,
                                "answer": waiting_msg
                            }

                        # Continue loop to let model process tool results
                        continue

                    else:
                        # No tool calls — this is the final answer
                        final_text = assistant_content or ""
                        if final_text:
                            print(f"\n✨ Final Response ({provider}): {final_text}")

                        # Save final assistant text to DB in Gemini format
                        gemini_final = types.Content(
                            role='model',
                            parts=[types.Part.from_text(text=final_text)]
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