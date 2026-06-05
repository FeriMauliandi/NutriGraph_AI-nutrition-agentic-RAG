from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Mengimpor Graph/Workflow LangGraph yang sudah dicompile dari folder src
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.agents.graph import app as ai_agent_app

# ==========================================
# INISIALISASI FASTAPI
# ==========================================
app = FastAPI(
    title="Dietary Tracker Multi-Agent API",
    description="API untuk analisis nutrisi otonom menggunakan LangGraph & Groq",
    version="1.0.0"
)

# ==========================================
# SCHEMAS (VALIDASI DATA DENGAN PYDANTIC)
# ==========================================
class DietRequest(BaseModel):
    user_input: str

class DietResponse(BaseModel):
    extracted_items: List[str]
    final_analysis: str

# ==========================================
# ENDPOINTS
# ==========================================
@app.get("/")
def read_root():
    return {"message": "Dietary Tracker Agent API berjalan dengan lancar! 🚀"}

@app.post("/api/v1/analyze", response_model=DietResponse)
async def analyze_diet(request: DietRequest):
    print(f"📥 Menerima request analisis untuk: {request.user_input}")
    
    # 1. Siapkan initial state untuk LangGraph
    initial_state = {
        "user_input": request.user_input,
        "extracted_items": [],
        "nutrition_data": {},
        "literature_context": "",
        "final_analysis": "",
        "error_logs": []
    }
    
    try:
        # 2. Eksekusi workflow LangGraph
        # .invoke() bersifat sinkron (blocking). 
        # Untuk skalabilitas masa depan, LangGraph juga mendukung .ainvoke() (asynchronous)
        result_state = ai_agent_app.invoke(initial_state)
        
        # 3. Kembalikan response yang sudah difilter sesuai schema
        return DietResponse(
            extracted_items=result_state.get("extracted_items", []),
            final_analysis=result_state.get("final_analysis", "Gagal menghasilkan analisis.")
        )
        
    except Exception as e:
        print(f"❌ Error saat memproses graph: {e}")
        # Kembalikan HTTP 500 jika terjadi fatal error di dalam LangGraph
        raise HTTPException(status_code=500, detail="Terjadi kesalahan internal pada agen AI.")