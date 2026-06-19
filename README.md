<div align="center">

# 🥗 NutriGraph AI

### Intelligent Nutrition Tracking with Agentic RAG & LangGraph

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_Workflow-orange)](https://langchain.com/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-teal?logo=fastapi)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-blueviolet)](https://www.trychroma.com/)
[![Groq](https://img.shields.io/badge/Groq-LLM_Inference-black)](https://groq.com)

</div>

---

## 📖 Tentang Proyek

**Dietary Tracker AI** adalah aplikasi pelacak nutrisi cerdas berbasis *Agentic Workflow*. Sistem ini tidak hanya mencatat kalori, tetapi bertindak sebagai konsultan gizi otonom yang mampu menimbang kapan harus menarik data dari API nutrisi global, dan kapan harus menggali literatur medis dari database lokal.

Proyek ini dibangun untuk mendemonstrasikan implementasi **Advanced RAG** (Retrieval-Augmented Generation) dan orkestrasi LLM menggunakan ekosistem Python modern, mengatasi tantangan umum seperti kesenjangan bahasa (*language barrier*) pada API dan halusinasi data.

---

## 🏗️ Arsitektur Sistem (Agentic Workflow)

Sistem menggunakan **LangGraph** untuk mengatur alur kerja cerdas (*conditional routing*). 

### Alur Orkestrasi (LangGraph Router)

```text
Input Pengguna → 🚦 Router Node (Intent Classification)
                      │
          ┌───────────┴───────────┐
          │                       │
     [track_diet]           [general_chat]
          │                       │
 🤖 Extraction Node      💬 General Chat Node
 (Pydantic Output)       (Advanced RAG Pipeline)
          │                       │
   ┌──────┴──────┐                │
   │             │                │
🌐 API Tool   📚 RAG Node         │
(FatSecret/  (Hybrid Search)      │
 USDA)           │                │
   └──────┬──────┘                │
          │                       │
   🧠 Synthesizer Node            │
          │                       │
          └───────────┬───────────┘
                      ↓
               Jawaban Akhir

---

## 🐳 Menjalankan dengan Docker

Proyek ini sudah dilengkapi **Dockerfile multi-stage** (image ramping untuk backend FastAPI) dan `Dockerfile.frontend` khusus Streamlit, serta `docker-compose.yml` untuk orkestrasi keduanya.

### Prasyarat
- [Docker](https://docs.docker.com/get-docker/) ≥ 20.10
- [Docker Compose](https://docs.docker.com/compose/install/) v2
- File `.env` berisi API key (lihat `.env.example`)

### Build & Jalankan
```bash
# 1. Salin template env dan isi API key Anda
cp .env.example .env
# (Edit file .env lalu isi GROQ_API_KEY, USDA_API_KEY, dll.)

# 2. Build image & jalankan semua service
docker compose up -d --build

# 3. Cek log
docker compose logs -f backend
docker compose logs -f frontend
```

Setelah container hidup:
| Service  | URL                                       |
|----------|-------------------------------------------|
| Backend  | http://localhost:8000                     |
| Frontend | http://localhost:8501                     |
| API Docs | http://localhost:8000/docs                |

### Hanya Backend
```bash
docker build -t dietary-tracker-backend .
docker run --rm -p 8000:8000 --env-file .env -v ${PWD}/data:/app/data dietary-tracker-backend
```

### Ingest data (ChromaDB) di dalam container
```bash
docker compose exec backend python script/ingest_data.py
```

### Hentikan & bersihkan
```bash
docker compose down            # stop container
docker compose down -v         # stop + hapus volume (ChromaDB, cache, dll.)
```

### Catatan
- **GPU tidak digunakan** di dalam image; embedding `Qwen3-Embedding-0.6B` berjalan di CPU (`device="cpu"` di fallback `ingest_data.py`). Untuk GPU, install `nvidia-container-toolkit` lalu tambahkan `runtime: nvidia` pada service `backend`.
- Folder `data/chroma_db` dan `data/cache` di-mount sebagai **named volume** agar persist antar restart.
