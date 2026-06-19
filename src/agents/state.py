from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from typing import Annotated

from langgraph.graph.message import add_messages

class NutriGraphState(TypedDict):
    messages: Annotated[list, add_messages]
    user_input: str
    image_data: Optional[str]
    intent: str
    extracted_items: List[Dict[str, Any]]
    needs_clarification: bool
    clarification_question: str
    clarification_type: str
    detected_from_image: bool
    image_items_confirmed: bool
    nutrition_data: Dict[str, Any]
    nutrition_sources: List[str]
    api_success: bool
    retry_count: int
    literature_context: str
    literature_sources: List[str]
    final_analysis: str
    error_logs: Optional[List[str]]
