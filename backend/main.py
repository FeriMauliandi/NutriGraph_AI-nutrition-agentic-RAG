from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.agents.graph import app as ai_agent_app
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

set_llm_cache(SQLiteCache(database_path="data/cache/langchain_cache.db"))


app = FastAPI(
    title="Dietary Tracker Multi-Agent API",
    description="API untuk analisis nutrisi otonom menggunakan LangGraph & Groq",
    version="1.0.0"
)

class DietRequest(BaseModel):
    user_input: str
    session_id: str = "default_user"
    image_data: Optional[str] = None

class DietResponse(BaseModel):
    extracted_items: List[str]
    final_analysis: str
    needs_clarification: bool = False
    clarification_question: str = ""

@app.get("/")
def read_root():
    return {"message": "Dietary Tracker Agent API berjalan dengan lancar! 🚀"}

@app.post("/api/v1/analyze", response_model=DietResponse)
async def analyze_diet(request: DietRequest):
    print(f"📥 Menerima request analisis untuk: {request.user_input} (Session: {request.session_id})")
    
    config = {"configurable": {"thread_id": request.session_id}}
    previous_state = ai_agent_app.get_state(config)
    previous_values = previous_state.values if previous_state else {}
    pending_clarification = previous_values.get("needs_clarification", False)
    
    initial_state = {
        "user_input": request.user_input,
        "image_data": request.image_data,
        "messages": previous_values.get("messages", []) if pending_clarification else [],
        "extracted_items": previous_values.get("extracted_items", []) if pending_clarification else [],
        "needs_clarification": pending_clarification,
        "clarification_question": previous_values.get("clarification_question", "") if pending_clarification else "",
        "clarification_type": previous_values.get("clarification_type", "") if pending_clarification else "",
        "detected_from_image": previous_values.get("detected_from_image", False) if pending_clarification else bool(request.image_data),
        "image_items_confirmed": previous_values.get("image_items_confirmed", False) if pending_clarification else False,
        "nutrition_data": previous_values.get("nutrition_data", {}) if pending_clarification else {},
        "nutrition_sources": previous_values.get("nutrition_sources", []) if pending_clarification else [],
        "literature_context": previous_values.get("literature_context", "") if pending_clarification else "",
        "literature_sources": previous_values.get("literature_sources", []) if pending_clarification else [],
        "final_analysis": "",
        "api_success": previous_values.get("api_success", True) if pending_clarification else True,
        "retry_count": previous_values.get("retry_count", 0) if pending_clarification else 0,
        "error_logs": previous_values.get("error_logs", []) if pending_clarification else []
    }
    
    try:
        # Gunakan config untuk persistent memory
        result_state = ai_agent_app.invoke(initial_state, config=config)
    
        items_list = []
        for item in result_state.get("extracted_items", []):
            if isinstance(item, dict):
                quantity = int(item.get("quantity", 1) or 1)
                item_name = item.get("asli", "")
                items_list.append(f"{quantity}x {item_name}" if quantity > 1 else item_name)
            else:
                items_list.append(item)

        # Kembalikan response
        return DietResponse(
            final_analysis=result_state["final_analysis"],
            extracted_items=items_list,
            needs_clarification=result_state.get("needs_clarification", False),
            clarification_question=result_state.get("clarification_question", "")
        )
        
    except Exception as e:
        print(f"error saat memproses graph: {e}")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan internal pada agen AI.")
