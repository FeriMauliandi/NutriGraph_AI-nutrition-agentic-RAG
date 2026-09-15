from typing import Dict, List, Any
from pydantic import BaseModel, Field

class IntentClassification(BaseModel):
    intent: str = Field(description=(
        "Pilih 'track_diet' JIKA pengguna menyebutkan detail makanan yang mereka konsumsi. "
        "Pilih 'general_chat' JIKA pengguna hanya menyapa, basa-basi, atau bertanya seputar nutrisi secara umum."))

class FoodItem(BaseModel):
    asli: str = Field(description="Nama makanan/minuman dalam bahasa Indonesia")
    english: str = Field(description="Terjemahan bahasa Inggris")

class ExtractionResult(BaseModel):
    items: List[FoodItem] = Field(description="Daftar item yang diekstrak")

class ClarificationDecision(BaseModel):
    needs_clarification: bool
    question: str = Field(description="Pertanyaan klarifikasi singkat. Kosongkan jika tidak perlu.")
