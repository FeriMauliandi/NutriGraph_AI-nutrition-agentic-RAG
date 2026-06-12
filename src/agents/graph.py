from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import os, sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(root_dir)

# Import State dan Nodes yang sudah dibuat
from src.agents.state import DietaryTrackerState
from src.agents.nodes import (
    intent_node,
    general_chat_node,
    extraction_node,
    clarification_node,
    api_tool_node,
    self_correction_node,
    rag_node,
    synthesizer_node
)

def route_intent(state: DietaryTrackerState):
    """Membaca state intent dan menentukan node selanjutnya"""
    if state.get("intent") == "track_diet":
        return "extraction"
    return "general_chat"

def route_clarification(state: DietaryTrackerState):
    """Menentukan apakah workflow perlu berhenti untuk meminta detail porsi."""
    if state.get("needs_clarification"):
        return "clarify"
    return "analyze"

def route_api_check(state: DietaryTrackerState):
    """Mengevaluasi apakah perlu mencoba ulang (Self-Correction) atau lanjut."""
    is_success = state.get("api_success", True)
    retry_count = state.get("retry_count", 0)
    
    # Batasi maksimal 2 kali retry agar tidak infinite loop
    if not is_success and retry_count < 2:
        return "self_correction"
    return "rag"

# Inisialisasi StateGraph
workflow = StateGraph(DietaryTrackerState)

# Daftarkan semua nodes
workflow.add_node("intent_router", intent_node)
workflow.add_node("general_chat", general_chat_node)
workflow.add_node("extraction", extraction_node)
workflow.add_node("clarification", clarification_node)
workflow.add_node("api_tool", api_tool_node)
workflow.add_node("self_correction", self_correction_node)
workflow.add_node("rag", rag_node)
workflow.add_node("synthesizer", synthesizer_node)

workflow.set_entry_point("intent_router")

# 4. Tambahkan Logika Cabang (Conditional Edges)
workflow.add_conditional_edges(
    "intent_router",
    route_intent,
    {
        "extraction": "extraction",
        "general_chat": "general_chat"
    }
)

# Alur Percakapan Umum
workflow.add_edge("general_chat", END)

# Alur Diet Tracking
workflow.add_edge("extraction", "clarification")

workflow.add_conditional_edges(
    "clarification",
    route_clarification,
    {
        "clarify": END,
        "analyze": "api_tool"
    }
)

workflow.add_conditional_edges(
    "api_tool",
    route_api_check,
    {
        "self_correction": "self_correction",
        "rag": "rag"
    }
)

workflow.add_edge("self_correction", "extraction")

workflow.add_edge("api_tool", "rag")
workflow.add_edge("rag", "synthesizer")
workflow.add_edge("synthesizer", END)

# Tambahkan MemorySaver untuk checkpointing
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
