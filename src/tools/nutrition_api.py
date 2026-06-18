import requests
from typing import List, Dict, Any
from src.core.config import settings


def get_usda_item(item_name: str) -> Dict[str, Any]:
    api_key = settings.USDA_API_KEY
    if not api_key:
        return {"found": False}

    url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {"api_key": api_key, "query": item_name, "pageSize": 1}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if "foods" in data and len(data["foods"]) > 0:
            nutrients = data["foods"][0].get("foodNutrients", [])

            cal = next(
                (n["value"] for n in nutrients if n["nutrientName"] == "Energy" and n["unitName"] == "KCAL"),
                0,
            )
            pro = next((n["value"] for n in nutrients if n["nutrientName"] == "Protein"), 0)
            car = next(
                (n["value"] for n in nutrients if n["nutrientName"] == "Carbohydrate, by difference"),
                0,
            )

            return {"found": True, "cal": cal, "pro": pro, "car": car, "source": "USDA"}
    except Exception:
        pass

    return {"found": False}


def fetch_combined_nutrition_data(items_data: List[Dict[str, str]]) -> Dict[str, Any]:
    print("[Tool] Memulai pencarian nutrisi via USDA...")

    total_calories = 0.0
    total_protein = 0.0
    total_carbs = 0.0
    found_items_log = []

    for item_dict in items_data:
        if isinstance(item_dict, str):
            nama_asli = item_dict
            nama_inggris = item_dict
            quantity = 1
        else:
            nama_asli = item_dict.get("asli", "")
            nama_inggris = item_dict.get("english", "")
            quantity = int(item_dict.get("quantity", 1) or 1)

        query = nama_inggris or nama_asli
        result = get_usda_item(query)

        if result["found"]:
            item_calories = result["cal"] * quantity
            item_protein = result["pro"] * quantity
            item_carbs = result["car"] * quantity

            total_calories += item_calories
            total_protein += item_protein
            total_carbs += item_carbs
            item_label = f"{quantity}x {nama_asli}" if quantity > 1 else nama_asli
            found_items_log.append(f"{item_label} ({item_calories} kkal dari {result['source']})")
        else:
            found_items_log.append(f"{nama_asli} (Tidak ditemukan)")

    summary_text = (
        f"Data diekstraksi untuk {len(items_data)} item. "
        f"Status: {', '.join(found_items_log)}. "
        f"Estimasi Total: {total_calories:.1f} Kalori, {total_protein:.1f}g Protein, {total_carbs:.1f}g Karbohidrat."
    )

    return {"summary": summary_text}
