from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from src.agents.state import DietaryTrackerState
from src.core.config import settings
from src.database.vector_store import get_retriever
from src.tools.nutrition_api import fetch_nutrition_data

llm = ChatGroq(
    model=settings.LLM_MODEL,
    temperature=0, 
)

retriever = get_retriever(llm, final_k=3, fetch_k=3)

def extraction_node(state: DietaryTrackerState) -> Dict[str, Any]:
    print("🤖 [Extraction Node] Sedang mengekstrak entitas...")
    prompt = PromptTemplate.from_template(
        "Kamu adalah asisten ahli gizi. Ekstrak nama makanan dan minuman dari teks berikut.\n"
        "Keluarkan HANYA daftar item yang dipisahkan dengan koma, tanpa penjelasan lain.\n"
        "Teks: {user_input}"
    )
    chain = prompt | llm
    response = chain.invoke({"user_input": state["user_input"]})
    items = [item.strip() for item in response.content.split(",")]
    return {"extracted_items": items}

def api_tool_node(state: DietaryTrackerState) -> Dict[str, Any]:
    print("[API Tool Node] Mengambil data makronutrien dari USDA...")
    items = state.get("extracted_items", [])
    if not items:
        return {"nutrition_data": {"summary": "Tidak ada data makanan untuk dianalisis."}}
    
    nutrition_result = fetch_nutrition_data(items)
    print(f"[API Tool Node] Hasil nutrisi: {nutrition_result}    ")
    return {"nutrition_data": nutrition_result}

def rag_node(state: DietaryTrackerState) -> Dict[str, Any]:
    print("[RAG Node] Melakukan Hybrid Search (BM25 + Vector) di database...")
    user_input = state["user_input"]
    
    try:
        docs = retriever.invoke(user_input)
        context = "\n\n".join(doc.page_content for doc in docs)
        
        if not context:
            context = "Tidak ada literatur spesifik yang ditemukan di database."
        
        print(f"[RAG Node] Literatur yang ditemukan:\n{context}\n ================================")
        return {"literature_context": context}
    except Exception as e:
        print(f"[RAG Node] Error: {e}")
        return {"literature_context": "", "error_logs": [f"RAG Error: {str(e)}"]}

def synthesizer_node(state: DietaryTrackerState) -> Dict[str, Any]:
    print("[Synthesizer Node] Merumuskan analisis akhir...")
    prompt = PromptTemplate.from_template(
        "Kamu adalah konsultan kebugaran dan nutrisi berbasis sains.\n\n"
        "Input Asli Pengguna: '{user_input}'\n"
        "Makanan yang diekstrak: {items}\n"
        "Data Nutrisi (Estimasi): {nutrition}\n"
        "Literatur nutrisi Pendukung: {context}\n\n"
        "Tugas Utama:\n"
        "1. Berikan analisis nutrisi per item secara singkat.\n"
        "2. Jelaskan apakah asupannya sudah ideal DENGAN MEMPERHATIKAN KONTEKS WAKTU MAKAN pada 'Input Asli Pengguna'.\n"
        "   - Jika pengguna hanya menyebut 'sarapan', 'makan siang', atau 1 sesi makan, evaluasilah target nutrisinya sebagai porsi satu kali makan (sekitar 1/3 dari kebutuhan harian).\n"
        "   - JANGAN menyebut asupannya 'belum ideal untuk harian' jika dia memang baru makan satu kali.\n"
        "   - Jika pengguna merangkum makanan seharian penuh, barulah evaluasi sebagai total asupan harian.\n\n"
        "Gunakan bahasa Indonesia yang profesional, ramah, dan ringkas. Beri saran pelengkap jika ada nutrisi yang kurang dari sesi makan tersebut."
    )
    chain = prompt | llm
    
    # Kita tambahkan user_input ke dalam dictionary invoke
    response = chain.invoke({
        "user_input": state.get("user_input", ""), # <-- Mengambil konteks asli
        "items": ", ".join(state.get("extracted_items", [])),
        "nutrition": state.get("nutrition_data", {}).get("summary", "Data tidak tersedia"),
        "context": state.get("literature_context", "Tidak ada referensi.")
    })
    return {"final_analysis": response.content}