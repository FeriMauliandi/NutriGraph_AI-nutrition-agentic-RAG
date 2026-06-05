from typing import TypedDict, List, Dict, Any, Optional

class DietaryTrackerState(TypedDict):
    """
    State utama yang akan mengalir melalui seluruh node (agen) di LangGraph.
    """
    # 1. Input asli dari pengguna
    # Contoh: "Habis puasa minum air kelapa campur garam dan makan ayam bakar."
    user_input: str
    
    # 2. Hasil ekstraksi entitas makanan/minuman oleh LLM
    # Contoh: ["air kelapa", "garam", "ayam bakar"]
    extracted_items: List[str]
    
    # 3. Data nutrisi kuantitatif yang didapat dari pemanggilan API Tool
    # Contoh: {"air kelapa": {"kalori": 45, "kalium": 600}, ...}
    nutrition_data: Dict[str, Any]
    
    # 4. Konteks literatur yang ditarik dari ChromaDB (RAG)
    # Contoh: "Penelitian menunjukkan kalium dari air kelapa dan natrium dari garam..."
    literature_context: str
    
    # 5. Kesimpulan akhir yang disintesis oleh agen terakhir
    final_analysis: str
    
    # 6. Variabel kontrol untuk Self-Correction (Sangat Penting untuk Agentic RAG!)
    # Jika API gagal atau jurnal tidak ditemukan, agen akan mencatat error di sini
    error_logs: Optional[List[str]]