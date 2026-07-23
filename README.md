# Robodog MCP Client (`robodog-mcp-client`)

FastAPI backend service yang bertindak sebagai **MCP Client & LLM Orchestrator** untuk sistem robot quadruped **Robodog**. Client ini menerima prompt dari pengguna (via frontend `robodog-web`), memproses instruksi menggunakan model LLM (Google Gemini, OpenAI GPT-4o, atau Ollama Qwen), serta berkomunikasi dengan `robodog-mcp-server` melalui protokol Model Context Protocol (MCP) untuk eksekusi aksi robot, query spasial database, dan inspeksi visual.

---

## 🚀 Fitur Utama

- **LLM & Agentic Orchestration**: Mengelola interaksi conversational agent menggunakan LangChain, LangGraph, dan FastMCP.
- **Dukungan Multi-Model**:
  - Google Gemini (`gemini-3.1-pro-preview`, `gemini-3-flash-preview`, `gemini-2.5-flash`, dll.)
  - OpenAI (`gpt-4o`, `gpt-4o-mini`)
  - Local/Self-hosted (`qwen3.5:27b` via Ollama)
- **Integrasi MCP Protocol**: Terhubung ke `robodog-mcp-server` melalui transport HTTP untuk memanggil tool navigasi, pergerakan, dan query SOP/database.
- **Integrasi Supabase**: Menyimpan dan mengambil riwayat percakapan (*chat session*), log inspeksi, dan dokumen pendukung.
- **Observabilitas & Tracing**: Dilengkapi integrasi Langfuse untuk memantau performa agent, latency, dan jejak eksekusi tool.
- **High Performance Event Loop**: Memakai `uvloop` secara otomatis saat berjalan di lingkungan Linux.

---

## 📁 Struktur Proyek

```text
robodog-mcp-client/
├── config/
│   ├── config.py           # Pengaturan variabel lingkungan & konfigurasi aplikasi
│   └── logging.py          # Formatter dan logger kustom
├── routes/
│   └── chat_robot.py       # Endpoint API FastAPI (/api/chat_robot)
├── schemas/
│   ├── request.py          # Model Pydantic untuk request chat & daftar model yang didukung
│   └── response.py         # Skema response API
├── services/
│   └── chat_robot.py       # Logika utama LLM Agent, instruksi sistem, dan MCP tool binding
├── utils/
│   ├── prompt.py           # Template prompt & sistem instruksi
│   └── tools_converter.py  # Helper konversi alat MCP ke format LangChain
├── main.py                 # Titik masuk aplikasi FastAPI & server Uvicorn
├── requirements.txt        # Daftar dependensi Python
└── README.md
```

---

## 🛠️ Persyaratan Sistem

- **Python**: `>= 3.10`
- **MCP Server**: `robodog-mcp-server` yang sudah berjalan (default port `8001`)
- **Database**: Supabase Instance

---

## ⚙️ Variabel Lingkungan (`.env`)

Buat file `.env` di root direktori `robodog-mcp-client/` dengan konfigurasi berikut:

```env
# Konfigurasi Server FastAPI
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info

# Connection ke MCP Server
MCP_URL=http://localhost:8001/mcp

# AI Provider Keys
GOOGLE_API_KEY=your_google_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
OLLAMA_HOST=http://localhost:11434

# Supabase Credentials
SUPABASE_URL=https://your-supabase-project.supabase.co
SUPABASE_KEY=your_supabase_anon_or_service_key

# Langfuse Tracing (Opsional)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

---

## 💻 Cara Memulai

### 1. Install Dependensi

Disarankan menggunakan *virtual environment*:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Jalankan Server

Jalankan langsung melalui `main.py`:

```bash
python main.py
```

Atau menggunakan Uvicorn secara manual:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Server akan aktif di `http://localhost:8000`.

---

## 📡 API Endpoint

### `POST /api/chat_robot/`

Endpoint utama untuk berinteraksi dengan AI Agent Robodog.

#### Request Body Example

```json
{
  "user_prompt": "Tolong pergi ke Boiler Room dan periksa tekanan pada pressure tank.",
  "session_id": 12,
  "user_id": "user_123",
  "model_name": "gemini-2.5-flash",
  "files": []
}
```

#### Supported Models

- `gemini-3.1-pro-preview`
- `gemini-3-flash-preview`
- `gemini-3.1-flash-lite`
- `gemini-2.5-flash` *(Default)*
- `gemini-2.5-pro`
- `qwen3.5:27b`
- `gpt-4o`
- `gpt-4o-mini`

---

## 🔗 Lisensi & Kontribusi

Proyek ini merupakan bagian dari ekosistem **Robodog Inspection System**.
