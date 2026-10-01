import re
from typing import Any, Dict, List
from langchain_core.messages import HumanMessage, AIMessage

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


def source_labels(docs: List[Any], fallback: str) -> List[str]:
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
            "quantity": float(item.get("quantity", 1) or 1) if item.get("quantity") is not None else 1.0,
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
                    item["quantity"] = max(float(m.group(1).replace(",", ".")), 1.0)
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
    text = user_input.lower().strip()
    
    # Cek penambahan item (tambah, plus, juga, ditambah)
    add_patterns = [
        r"\b(?:tambah|plus|serta|juga|ditambah)\s+(?:item\s+)?(.+)$",
    ]
    for pat in add_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            addition = m.group(1).strip()
            return existing, addition, "add"

    patterns = [
        r"\bbukan\s+(.+?)\s+(?:tetapi|tapi|melainkan|seharusnya|harusnya|yang benar)\s+(.+)$",
        r"\b(.+?)\s+diganti(?:\s+dengan)?\s+(.+)$",
        r"\bganti\s+(.+?)\s+dengan\s+(.+)$",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
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


def update_messages(state, question):
    msgs = state.get("messages", [])
    return msgs + [HumanMessage(content=state.get("user_input", "")), AIMessage(content=question)]


def combined_user_context(state: Any) -> str:
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
