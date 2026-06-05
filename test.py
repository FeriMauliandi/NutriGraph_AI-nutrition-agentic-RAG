from typing import TypedDict, List
from langgraph.graph import StateGraph, END

# ==========================================
# 1. DEFINISI STATE (Memori Bersama)
# ==========================================
# State ini adalah "buku catatan" yang akan dioper antar agen.
class GraphState(TypedDict):
    user_input: str
    extracted_items: List[str]
    nutrition_data: str
    literature_context: str
    final_analysis: str

# ==========================================
# 2. DEFINISI NODES (Fungsi Agen)
# ==========================================
# Setiap fungsi mewakili satu agen spesifik dalam workflow.

def extraction_node(state: GraphState):
    print("▶️ [Agent] Berpikir: Mengekstrak entitas makanan dari teks...")
    # Di sini nanti kamu masukkan logika LLM prompt (LangChain)
    # Output fungsi harus mengembalikan kamus (dictionary) yang memperbarui State
    return {"extracted_items": ["Ayam Bakar", "Air Kelapa"]}

def api_tool_node(state: GraphState):
    print("▶️ [Agent] Berpikir: Memanggil API Nutrisi eksternal...")
    items = state.get("extracted_items", [])
    # Di sini nanti logika requests ke API (misal: Edamam / OpenFoodFacts)
    return {"nutrition_data": f"Data untuk {items}: 300 Kalori, 20g Protein, 150mg Kalium"}

def rag_node(state: GraphState):
    print("▶️ [Agent] Berpikir: Mencari literatur di ChromaDB...")
    query = state.get("user_input", "")
    # Di sini logika retriever dari ChromaDB
    return {"literature_context": "Jurnal: Air kelapa dan protein tinggi sangat optimal untuk rehidrasi seluler."}

def synthesizer_node(state: GraphState):
    print("▶️ [Agent] Berpikir: Mensintesis data gizi dan jurnal...")
    # Di sini agen LLM terakhir membaca kedua data dari Node API dan RAG
    return {"final_analysis": "✅ Kesimpulan: Asupan Anda sangat baik. Ayam bakar memberikan protein, sedangkan air kelapa mengembalikan elektrolit sesuai standar medis."}

# ==========================================
# 3. MEMBANGUN GRAPH (Orkestrasi)
# ==========================================
# Inisialisasi StateGraph
workflow = StateGraph(GraphState)

# Daftarkan semua nodes
workflow.add_node("extraction", extraction_node)
workflow.add_node("api_tool", api_tool_node)
workflow.add_node("rag", rag_node)
workflow.add_node("synthesizer", synthesizer_node)

# ==========================================
# 4. MENGATUR ALUR (Edges & Routing)
# ==========================================
# Titik mulai
workflow.set_entry_point("extraction")

# Pemrosesan Paralel: Dari Extraction, jalankan API Tool dan RAG secara bersamaan
workflow.add_edge("extraction", "api_tool")
workflow.add_edge("extraction", "rag")

# Setelah API Tool dan RAG selesai, teruskan hasilnya ke Synthesizer
workflow.add_edge("api_tool", "synthesizer")
workflow.add_edge("rag", "synthesizer")

# Akhiri workflow
workflow.add_edge("synthesizer", END)

# ==========================================
# 5. KOMPILASI & EKSEKUSI
# ==========================================
app = workflow.compile()

# --- BLOK PENGUJIAN ---
if __name__ == "__main__":
    print("=== Memulai Simulasi Agentic RAG ===\n")
    
    # Input awal dari pengguna
    inputs = {
        "user_input": "Tadi saya buka puasa makan ayam bakar dan minum air kelapa murni."
    }
    
    # Menjalankan graph
    result = app.invoke(inputs)
    
    print("\n=== HASIL AKHIR ===")
    print(result["final_analysis"])