<div align="center">

# 🥗 NutriGraph AI

### Intelligent Nutrition Tracking with Agentic RAG, LangGraph & VLM

[![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_Workflow-orange)](https://langchain.com/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-teal?logo=fastapi)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-blueviolet)](https://www.trychroma.com/)
[![Groq](https://img.shields.io/badge/Groq-LLM%20%26%20VLM-black)](https://groq.com)
[![Pytest](https://img.shields.io/badge/Pytest-Unit%20Tested-green?logo=pytest)](https://docs.pytest.org)

</div>

---

## 📖 Tentang Proyek

**NutriGraph AI** adalah aplikasi pelacak nutrisi otonom cerdas berbasis *Agentic Workflow*. Berbeda dari kalkulator kalori konvensional, NutriGraph AI bertindak sebagai **konsultan gizi AI mandiri** yang mampu:

1. **Menganalisis Teks & Foto Makanan**: Mengintegrasikan Vision Language Model (VLM) untuk membaca foto makanan atau input bahasa alami dalam Bahasa Indonesia.
2. **Pengambilan Data Nutrisi Otonom**: Mengakses API nutrisi global (USDA FoodData Central) dengan otomatis menangani penerjemahan istilah makanan lokal.
3. **RAG Literatur Medis**: Menggali literatur kesehatan dari database lokal menggunakan teknik *Hybrid Search* (BM25 + Vector) dan *Cross-Encoder Reranking*.
4. **Self-Correction & Klarifikasi Multiturn**: Meminta klarifikasi jika porsi/waktu makan kurang, serta melakukan retry otomatis (*self-correction*) jika API nutrisi gagal.

Proyek ini dibangun sebagai demonstrasi **End-to-End AI Engineering** yang mencakup integrasi LLM/VLM, RAG pipeline, FastAPI backend, Streamlit frontend, unit testing, dan containerization Docker.

---

## ✨ Fitur Utama

- 📸 **Multi-Modal Vision Recognition**: Identifikasi makanan langsung dari foto mengunakan Qwen Vision Language Model.
- 🚦 **Agentic Intent Routing**: Otomatis membedakan apakah pengguna ingin mencatat makanan (*track_diet*) atau bertanya seputar kesehatan umum (*general_chat*).
- 🔍 **Advanced Hybrid Retrieval**: Kombinasi `BM25Retriever` (kata kunci) + `ChromaDB` (vektor spasial) + `BAAI/bge-reranker-v2-m3` untuk hasil RAG paling akurat.
- 🔁 **Self-Correction Mechanism**: Melakukan fallback pencarian istilah alternatif secara otomatis jika data nutrisi awal tidak ditemukan.
- 💬 **Multi-turn Clarification**: Kemampuan bertanya balik jika data waktu makan atau porsi belum lengkap.
- ⚡ **High Performance Caching**: Menggunakan `CacheBackedEmbeddings` dan `SQLiteCache` untuk respons cepat dan efisiensi kuota LLM.
- 🧪 **Unit Tested**: Dilengkapi rangkaian pengujian otomatis dengan `pytest`.

---

## 🏗️ Arsitektur Sistem (LangGraph Workflow)

Seluruh alur percakapan dan pengambilan keputusan diatur secara eksplisit menggunakan **LangGraph StateGraph**.

```text
               Input Pengguna / Foto Makanan
                             │
                             ▼
                 🚦 Router Node (Intent)
                             │
             ┌───────────────┴───────────────┐
             ▼                               ▼
    [track_diet]                        [general_chat]
             │                               │
  🤖 Extraction Node                         │
  (Structured Pydantic)                      │
             │                               │
             ▼                               │
  ❓ Clarification Node                      │
  (Check Porsi/Waktu)                        │
             │                               │
       ┌─────┴─────┐                         │
       ▼           ▼                         │
   [Lengkap]  [Butuh Info]                   │
       │           │                         │
       │           └──► 💬 Tanya User ──► END│
       ▼                                     │
 🌐 API Tool Node (USDA)                     │
       │                                     │
   ┌───┴───┐                                 │
   ▼       ▼                                 │
 [Gagal] [Sukses]                            │
   │       │                                 │
   ▼       │                                 │
🔄 Self-Correction                           │
   │       │                                 │
   └───────┼─────────────────────────────────┤
           ▼                                 ▼
    📚 RAG Node                       📚 Advanced RAG
    (Hybrid Search)                   (MultiQuery + Reranker)
           │                                 │
           └────────────────┬────────────────┘
                            ▼
                  🧠 Synthesizer Node
                  (Analisis Gizi Akhir)
                            │
                            ▼
                           END
```

---

## 🛠️ Tech Stack

| Komponen | Teknologi |
|----------|-----------|
| **Orchestration** | LangGraph, LangChain Classic |
| **LLM & Vision** | Groq API (`openai/gpt-oss-20b`, `qwen/qwen3.8-27b`) |
| **Vector DB** | ChromaDB |
| **Embeddings** | `Qwen/Qwen3-Embedding-0.6B` (HuggingFace) |
| **Reranker** | `BAAI/bge-reranker-v2-m3` |
| **External API** | USDA FoodData Central API |
| **Backend API** | FastAPI, Uvicorn, Pydantic v2 |
| **Frontend UI** | Streamlit |
| **Testing** | Pytest, Unittest.mock |
| **Containerization** | Docker, Docker Compose (Multi-Stage) |

---

## 📂 Struktur Direktori

```text
NutriGraph_AI/
├── backend/
│   └── main.py              # REST API Service (FastAPI)
├── frontend/
│   └── app.py               # User Interface (Streamlit)
├── src/
│   ├── agents/
│   │   ├── graph.py         # Orkestrasi Alur LangGraph
│   │   ├── nodes.py         # Fungsi Utama Setiap Node Agen
│   │   ├── schemas.py       # Validasi Output Pydantic
│   │   ├── state.py         # Definisi State Graph
│   │   └── utils.py         # Helper Parsing Teks & Porsi
│   ├── core/
│   │   └── config.py        # Pengaturan Centralized Envs
│   ├── database/
│   │   └── vector_store.py  # Inisialisasi Hybrid Search & Reranker
│   ├── tools/
│   │   └── nutrition_api.py # Integrasi USDA API
│   └── utils/
│       ├── embeddings.py    # Embedding Caching Engine
│       └── loader.py        # Data Ingestion Loader (.pdf, .csv, web)
├── script/
│   ├── ingest_data.py       # Script Pengisi Vector Database
│   └── evaluate_ragas.py    # Evaluation Script RAGAS Framework
├── tests/
│   ├── test_utils.py        # Unit test fungsi parser & helper
│   ├── test_schemas.py      # Unit test validasi schema Pydantic
│   └── test_nutrition_api.py# Unit test API nutrisi (dengan Mocking)
├── data/                    # Penyimpanan cache & ChromaDB local
├── Dockerfile               # Multi-stage Docker build Backend
├── Dockerfile.frontend      # Docker build Frontend
├── docker-compose.yml       # Orchestration Backend + Frontend
├── requirements.txt         # Daftar dependency Python
└── README.md
```

---

## 🚀 Cara Menjalankan

### Opsi A: Menjalankan dengan Docker (Rekomendasi)

**Prasyarat:** Docker & Docker Compose sudah terinstall.

```bash
# 1. Clone repository
git clone https://github.com/username/NutriGraph_AI.git
cd NutriGraph_AI

# 2. Buat file .env dari template
cp .env.example .env
# Edit file .env dan masukkan GROQ_API_KEY & USDA_API_KEY Anda

# 3. Jalankan container
docker compose up -d --build

# 4. Ingest data awal ke ChromaDB (di dalam container)
docker compose exec backend python script/ingest_data.py
```

Akses Layanan:
- 🎨 **Frontend UI**: [http://localhost:8501](http://localhost:8501)
- ⚡ **Backend API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Opsi B: Menjalankan Secara Lokal (Manual)

**Prasyarat:** Python >= 3.10

```bash
# 1. Buat dan aktifkan virtual environment
python -m venv venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Buat file .env dan isi API key
cp .env.example .env

# 4. Jalankan Ingestion Data ke ChromaDB
python script/ingest_data.py

# 5. Jalankan Backend Server (Terminal 1)
uvicorn backend.main:app --reload --port 8000

# 6. Jalankan Frontend UI (Terminal 2)
streamlit run frontend/app.py
```

---

## 🧪 Pengujian (Unit Testing)

Project ini dilengkapi dengan rangkaian pengujian unit berbasis `pytest` yang mencakup validasi schema, fungsi parsing teks, serta simulasi API (*mocking*).

Jalankan pengujian dengan perintah:

```bash
python -m pytest tests/ -v
```

**Cakupan Test:**
- `tests/test_utils.py`: Normalisasi nama makanan, pembersihan teks, ekstraksi porsi.
- `tests/test_schemas.py`: Validasi tipe data dan struktur schema Pydantic.
- `tests/test_nutrition_api.py`: Simulasi penanganan data API USDA & *error fallback*.

---

## 📊 Evaluasi Framework (RAGAS)

Untuk mengukur kualitas RAG (Retrieval-Augmented Generation), project ini menggunakan framework **RAGAS** (`evaluate_ragas.py`) dengan metrik utama:

- **Faithfulness**: Mengukur apakah jawaban AI sesuai dengan konteks literatur lokal (bebas halusinasi).
- **Answer Relevancy**: Mengukur sejauh mana jawaban menjawab pertanyaan pengguna.
- **Context Precision & Recall**: Mengukur ketepatan dokumen yang ditarik oleh Hybrid Search.

Jalankan evaluasi RAG:
```bash
python script/evaluate_ragas.py
```

---

## 🔌 Dokumentasi REST API

### 1. Check Health Status
- **URL:** `GET /`
- **Response:** `{"message": "Dietary Tracker Agent API berjalan dengan lancar! 🚀"}`

### 2. Analisis Nutrisi (Main Endpoint)
- **URL:** `POST /api/v1/analyze`
- **Request Body:**
  ```json
  {
    "user_input": "Saya makan 2 porsi nasi goreng dan 1 es teh jam 12 siang",
    "session_id": "user_session_123",
    "image_data": null
  }
  ```
- **Response:**
  ```json
  {
    "extracted_items": ["2x nasi goreng", "es teh"],
    "final_analysis": "Analisis Gizi: Total estimasi 550 Kalori...",
    "needs_clarification": false,
    "clarification_question": ""
  }
  ```

---

## 📜 Lisensi & Kontribusi

Dipublikasikan di bawah lisensi **MIT License**. Terbuka untuk kontribusi dan pengembangan lebih lanjut.
