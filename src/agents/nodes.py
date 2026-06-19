import json
import re
from typing import Any, Dict, List

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun

from src.agents.state import NutriGraphState
from src.core.config import settings
from src.database.vector_store import get_advanced_retriever, get_hybrid_retriever
from src.tools.nutrition_api import fetch_combined_nutrition_data


# ---------------------------------------------------------------------------
# LLM, Retrievers & Constants
# ---------------------------------------------------------------------------

llm = ChatGroq(model=settings.LLM_MODEL, temperature=0)
vlm = ChatGroq(model=settings.VLM_MODEL, temperature=0)
diet_retriever = get_hybrid_retriever(final_k=3)
chat_retriever = get_advanced_retriever(llm, final_k=3, fetch_k=5)

COMMON_TRANSLATIONS: Dict[str, str] = {
    "siomay": "steamed fish dumpling",
    "tahu": "tofu",
    "kentang": "potato",
    "kol rebus": "boiled cabbage",
    "kol": "cabbage",
}

RE_PORTION = re.compile(
    r"\b\d+\s*(porsi|piring|mangkuk|potong|gram|g|kg|gelas|buah|bungkus|sendok|sdm|sdt)\b"
    r"|\b(se)?(porsi|piring|mangkuk|gelas|bungkus|buah)\b", re.IGNORECASE)
RE_TIME = re.compile(
    r"\b(pagi|siang|sore|malam|sarapan|breakfast|lunch|dinner|brunch)\b"
    r"|\b(makan\s+)?(pagi|siang|sore|malam)\b"
    r"|\b(jam|pukul)\s*\d{1,2}([:.]\d{2})?\b", re.IGNORECASE)
RE_CLARIFY_NOISE = re.compile(
    r"\b(porsi|utama|waktu|jam|pukul|pagi|siang|sore|malam|sarapan|breakfast|lunch|dinner|brunch)\b",
    re.IGNORECASE)
RE_AFFIRMATIVE = re.compile(
    r"^\s*(ya|iya|y|yes|benar|betul|sudah benar|bener|ok|oke|sesuai)"
    r"(\s*,?\s*(benar|betul|bener|sudah benar|sesuai|kok|aja))?\s*[.!?]*\s*$", re.IGNORECASE)
RE_NEGATIVE = re.compile(
    r"^\s*(tidak|nggak|enggak|bukan|no|salah|belum|kurang tepat)\s*[.!?]*\s*$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _source_labels(docs: List[Any], fallback: str) -> List[str]:
    seen, labels = set(), []
    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        label = meta.get("title") or meta.get("source") or fallback
        if label and label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


def normalize_extracted_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, result = set(), []
    for item in items:
        name = str(item.get("asli", "")).strip().lower()
        if not name or name in seen:
            continue
        seen.add(name)
        result.append({
            "asli": name,
            "english": str(item.get("english", "")).strip().lower() or COMMON_TRANSLATIONS.get(name, name),
            "quantity": int(item.get("quantity", 1) or 1),
        })
    return result


def infer_item_quantities(user_input: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    text = user_input.lower()
    for item in items:
        variants = [re.escape(item["asli"])]
        if item.get("english") and item["english"] != item["asli"]:
            variants.append(re.escape(item["english"]))
        for v in variants:
            for pat in [
                rf"\b(\d+(?:[.,]\d+)?)\s*(?:butir|buah|porsi|sdm|sdt|gram|g|potong|lembar|sendok)?\s*{v}\b",
                rf"\b{v}\s*(?:sebanyak\s*)?(\d+(?:[.,]\d+)?)\b",
            ]:
                m = re.search(pat, text)
                if m:
                    item["quantity"] = max(int(float(m.group(1).replace(",", "."))), 1)
                    break
    return items


def likely_clarification_only(text: str) -> bool:
    t = RE_PORTION.sub(" ", text.lower())
    t = RE_TIME.sub(" ", t)
    t = RE_CLARIFY_NOISE.sub(" ", t)
    t = re.sub(r"\b(makan|minum|saya|aku|tadi|barusan|sudah|pada|waktu|jam|malam|pagi|siang|sore)\b", " ", t)
    return not re.sub(r"[^a-zA-Z\s-]", " ", t).strip()


def fallback_extract_items(user_input: str) -> List[Dict[str, Any]]:
    text = re.sub(r"\b\d+\s*(porsi|piring|mangkuk|potong|gram|gelas|buah|bungkus)\b", "", user_input.lower())
    text = re.sub(r"\b(saya|aku|makan|minum|isinya|isi|dengan|detail|tambahan|porsi)\b", "", text)
    items = []
    for part in re.split(r",|\bdan\b|\+|/", text):
        name = re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z\s-]", " ", part)).strip()
        if len(name) >= 3:
            items.append({"asli": name, "english": COMMON_TRANSLATIONS.get(name, name)})
    return normalize_extracted_items(items)


def format_detected_items(items: List[Dict[str, Any]]) -> str:
    names = [str(i.get("asli", "")).strip() for i in items if i.get("asli")]
    return ", ".join(names) if names else "makanan pada gambar"


def split_partial_item_correction(user_input: str, existing: List[Dict[str, Any]]):
    patterns = [
        r"\bbukan\s+(.+?)\s+(?:tetapi|tapi|melainkan|seharusnya|harusnya|yang benar)\s+(.+)$",
        r"\b(.+?)\s+diganti(?:\s+dengan)?\s+(.+)$",
        r"\bganti\s+(.+?)\s+dengan\s+(.+)$",
    ]
    for pat in patterns:
        m = re.search(pat, user_input.lower().strip(), re.IGNORECASE)
        if m:
            wrong, replacement = m.group(1), m.group(2)
            wrong_names = {
                i["asli"] for i in existing
                if re.search(rf"\b{re.escape(i['asli'])}\b", wrong)
                or (i.get("english") and re.search(rf"\b{re.escape(i['english'])}\b", wrong))
            }
            if wrong_names:
                kept = [i for i in existing if i["asli"] not in wrong_names]
                return kept, replacement, True
    return [], user_input, False


def _update_messages(state, question):
    msgs = state.get("messages", [])
    return msgs + [HumanMessage(content=state.get("user_input", "")), AIMessage(content=question)]


def _combined_user_context(state: NutriGraphState) -> str:
    parts: List[str] = []
    for msg in state.get("messages", []):
        if isinstance(msg, HumanMessage):
            content = str(getattr(msg, "content", "")).strip()
            if content and content not in parts:
                parts.append(content)

    current = str(state.get("user_input", "")).strip()
    if current and current not in parts:
        parts.append(current)

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------

def intent_node(state: NutriGraphState) -> Dict[str, Any]:
    print("[Router Node] Menganalisis intent...")

    if state.get("image_data"):
        return {"intent": "track_diet", "detected_from_image": True}

    clr_type = state.get("clarification_type", "")
    if state.get("needs_clarification") and clr_type in {"item_confirmation", "item_correction", "meal_time"}:
        return {"intent": "track_diet"}

    if state.get("needs_clarification") and state.get("extracted_items"):
        return {"intent": "track_diet", "needs_clarification": False, "clarification_question": ""}

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Tentukan intent: track_diet (ada detail makanan) atau general_chat (sapaan/tanya umum)."),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{user_input}"),
    ])
    response = (prompt | llm.with_structured_output(IntentClassification)).invoke({
        "user_input": state["user_input"], "messages": state.get("messages", [])})
    return {"intent": response.intent}


def general_chat_node(state: NutriGraphState) -> Dict[str, Any]:
    print("[General Chat Node] Menjawab pertanyaan umum...")
    user_input = state.get("user_input", "")
    messages = state.get("messages", [])
    context, sources, used_web = "", [], False

    try:
        docs = chat_retriever.invoke(user_input)
        context = "\n\n".join(d.page_content for d in docs)
        sources = _source_labels(docs, "Database lokal")
    except Exception as e:
        print(f"RAG Error: {e}")

    if not context:
        try:
            context = DuckDuckGoSearchRun().invoke(user_input)
            sources = ["DuckDuckGo Web Search"]
            used_web = True
        except Exception as e:
            print(f"Web Search Error: {e}")
            context = "Gagal mengambil data."

    source_label = "INTERNET (WEB SEARCH)" if used_web else "DATABASE LOKAL (RAG)"
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Asisten gizi. Jawab ringkas & sopan.\nLiteratur: {context}\nSumber: {source_label}"),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{user_input}"),
    ])
    response = (prompt | llm).invoke({"user_input": user_input, "context": context,
                                       "source_label": source_label, "messages": messages})
    return {
        "final_analysis": response.content,
        "literature_context": f"[Sumber: {source_label}]\n{context}",
        "literature_sources": sources,
        "messages": messages + [HumanMessage(content=user_input), AIMessage(content=response.content)],
    }


def extraction_node(state: NutriGraphState) -> Dict[str, Any]:
    print("[Extraction Node] Ekstraksi entitas...")
    user_input = state.get("user_input", "")
    print(f"User input for extraction: {user_input}")
    image_data = state.get("image_data")
    messages = state.get("messages", [])
    existing = state.get("extracted_items", [])
    clr_type = state.get("clarification_type", "")

    # Handle confirmation/correction of image items
    if clr_type == "item_confirmation":
        if RE_AFFIRMATIVE.match(user_input):
            return {"extracted_items": normalize_extracted_items(existing), "needs_clarification": False,
                    "clarification_question": "", "clarification_type": "", "image_items_confirmed": True, "detected_from_image": True}
        if RE_NEGATIVE.match(user_input):
            return {"extracted_items": normalize_extracted_items(existing), "needs_clarification": True,
                    "clarification_question": "Mohon tuliskan item makanan yang benar pada gambar.",
                    "clarification_type": "item_correction",
                    "final_analysis": "Mohon tuliskan item makanan yang benar pada gambar.", "detected_from_image": True}

        kept, replacement, is_partial = split_partial_item_correction(user_input, existing)
        existing = kept if is_partial else []
        user_input = replacement if is_partial else user_input
        image_data = None

    elif clr_type == "item_correction":
        kept, replacement, is_partial = split_partial_item_correction(user_input, existing)
        existing = kept if is_partial else []
        user_input = replacement if is_partial else user_input
        image_data = None

    if existing and likely_clarification_only(user_input):
        return {"extracted_items": infer_item_quantities(user_input, normalize_extracted_items(existing))}

    # Extract items via VLM or LLM
    try:
        if image_data:
            extracted_data = [
                {"asli": i.asli, "english": i.english}
                for i in vlm.with_structured_output(ExtractionResult).invoke([HumanMessage(content=[
                    {"type": "text", "text": f"Ekstrak makanan dari gambar. Teks pengguna: '{user_input}'"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                ])]).items
            ]
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Ekstrak makanan (Indo & English) dari input pengguna."),
                MessagesPlaceholder(variable_name="messages"),
                ("human", "{user_input}"),
            ])
            extracted_data = [
                {"asli": i.asli, "english": i.english}
                for i in (prompt | llm.with_structured_output(ExtractionResult))
                .invoke({"user_input": user_input, "messages": messages}).items
            ]
    except Exception as e:
        print(f"Fallback extraction: {e}")
        extracted_data = fallback_extract_items(user_input)

    # Merge with existing
    if existing:
        existing_names = {i["asli"] for i in existing}
        extracted_data = existing + [i for i in extracted_data if i["asli"] not in existing_names]

    extracted_data = infer_item_quantities(user_input, normalize_extracted_items(extracted_data))
    result = {"extracted_items": extracted_data}

    print(f"Extracted items: {extracted_data}")

    if image_data:
        result.update({"detected_from_image": True, "image_items_confirmed": False, "clarification_type": ""})
    elif clr_type in {"item_confirmation", "item_correction"}:
        result.update({"needs_clarification": False, "clarification_question": "",
                        "clarification_type": "", "image_items_confirmed": True, "detected_from_image": True})
    return result


def clarification_node(state: NutriGraphState) -> Dict[str, Any]:
    print("[Clarification Node] Cek kelengkapan data...")
    items = state.get("extracted_items", [])
    clr_type = state.get("clarification_type", "")

    def _reply(question, clr, extra=None):
        base = {"needs_clarification": True, "clarification_question": question,
                "clarification_type": clr, "final_analysis": question,
                "messages": _update_messages(state, question), "literature_sources": []}
        if extra:
            base.update(extra)
        return base

    if clr_type == "item_correction":
        q = state.get("clarification_question") or "Mohon tuliskan item makanan yang benar pada gambar."
        return _reply(q, "item_correction")

    if not items:
        return _reply("Apa yang Anda konsumsi?", "missing_food")

    if state.get("detected_from_image") and not state.get("image_items_confirmed"):
        detected = format_detected_items(items)
        q = (f"Saya mendeteksi item berikut dari gambar: {detected}. "
             "Apakah item tersebut sudah benar? Jika belum, tuliskan koreksi dengan kata kunci 'diganti'.")
        return _reply(q, "item_confirmation")

    full_text = _combined_user_context(state)
    has_portion = bool(RE_PORTION.search(full_text))
    has_time = bool(RE_TIME.search(full_text))

    if state.get("detected_from_image") and state.get("image_items_confirmed"):
        if has_time:
            return {"needs_clarification": False, "clarification_question": "", "clarification_type": ""}
        return _reply("Kapan waktu makannya? Sebutkan jam atau waktu makan (misal: sarapan, makan siang, jam 12.00).", "meal_time")

    if has_portion and has_time:
        return {"needs_clarification": False, "clarification_question": "", "clarification_type": ""}

    main = items[0].get("asli", "makanan")
    q = (f"Berapa porsi {main} dan kapan waktu makannya?" if not has_portion and not has_time
         else f"Berapa porsi {main}?" if not has_portion else "Kapan waktu makannya?")
    return _reply(q, "portion_time")


def api_tool_node(state: NutriGraphState) -> Dict[str, Any]:
    print("[API Tool Node] Mengambil data nutrisi...")
    items = state.get("extracted_items", [])
    if not items:
        return {"nutrition_data": {"summary": "Tidak ada data."}, "nutrition_sources": []}
    result = fetch_combined_nutrition_data(items)
    return {"nutrition_data": result, "api_success": "0.0 Kalori" not in result.get("summary", ""),
            "nutrition_sources": ["USDA FoodData Central API"]}


def self_correction_node(state: NutriGraphState) -> Dict[str, Any]:
    retry = state.get("retry_count", 0)
    print(f"[Self-Correction Node] Retry ke-{retry + 1}...")
    return {"retry_count": retry + 1, "extracted_items": [], "nutrition_sources": [], "literature_sources": [],
            "messages": [HumanMessage(content=(
                "Sistem API gagal. Ekstrak ulang menggunakan sinonim, bahan dasar mentah, atau istilah Inggris yang lebih umum."))]}


def rag_node(state: NutriGraphState) -> Dict[str, Any]:
    print("[RAG Node] Hybrid Search...")
    items = state.get("extracted_items", [])
    query = " ".join(
        v for i in items for v in ([i["asli"]] + ([i["english"]] if i.get("english") and i["english"] != i["asli"] else []))
    ) or state.get("user_input", "")
    try:
        docs = diet_retriever.invoke(query)
        return {"literature_context": "\n\n".join(d.page_content for d in docs) or "Tidak ada literatur.",
                "literature_sources": _source_labels(docs, "Database lokal")}
    except Exception as e:
        return {"literature_context": "", "literature_sources": [], "error_logs": [str(e)]}


def synthesizer_node(state: NutriGraphState) -> Dict[str, Any]:
    print("[Synthesizer Node] Analisis akhir...")
    user_input = state.get("user_input", "")
    user_context = _combined_user_context(state) or user_input
    messages = state.get("messages", [])
    items = state.get("extracted_items", [])

    items_str = ", ".join(
        f"{i.get('quantity', 1)}x {i['asli']}" if int(i.get("quantity", 1) or 1) > 1 else i["asli"]
        for i in items)

    prompt = PromptTemplate.from_template(
        "Kamu adalah konsultan gizi berbasis sains.\n\n"
        "Input/konteks pengguna: '{user_input}'\nMakanan: {items}\nData Nutrisi: {nutrition}\nLiteratur: {context}\n\n"
        "1. Analisis nutrisi per item (perhatikan jumlah item).\n"
        "2. Evaluasi apakah asupan sudah ideal sesuai waktu makan.\n"
        "Gunakan bahasa Indonesia profesional. Beri saran pelengkap jika perlu.")

    response = (prompt | llm).invoke({
        "user_input": user_context, "items": items_str,
        "nutrition": state.get("nutrition_data", {}).get("summary", "N/A"),
        "context": state.get("literature_context", "N/A"),
    })

    all_sources = [s for lst in (state.get("nutrition_sources", []), state.get("literature_sources", []))
                   for s in lst if s]
    citation = "\n\nSumber:\n" + "\n".join(f"- {s}" for s in dict.fromkeys(all_sources)) if all_sources else ""

    return {"final_analysis": f"{response.content}{citation}",
            "messages": messages + [HumanMessage(content=user_input), AIMessage(content=response.content)]}
