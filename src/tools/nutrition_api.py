import requests
from typing import List, Dict, Any
from src.core.config import settings

def fetch_nutrition_data(items: List[str]) -> Dict[str, Any]:
    """
    Fungsi ini dipanggil oleh API Tool Agent untuk mencari data nutrisi
    menggunakan USDA FoodData Central API.
    """
    print(f"🌐 [Tool] Melakukan request ke USDA API untuk: {items}")
    
    api_key = settings.USDA_API_KEY
    if not api_key:
        return {"error": "USDA_API_KEY tidak ditemukan di konfigurasi."}

    # Endpoint pencarian USDA
    url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    
    total_calories = 0.0
    total_protein = 0.0
    total_carbs = 0.0
    
    found_items_log = []

    # USDA memproses per item, jadi kita loop dari daftar makanan
    for item in items:
        params = {
            "api_key": api_key,
            "query": item,
            "pageSize": 1 # Kita hanya butuh hasil pencarian paling atas/relevan
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Jika makanan ditemukan di database USDA
            if "foods" in data and len(data["foods"]) > 0:
                food = data["foods"][0]
                nutrients = food.get("foodNutrients", [])
                
                # Ekstrak nutrisi spesifik dari array USDA
                # Catatan: USDA menyimpan nama nutrisi dalam bahasa Inggris
                item_cal = next((n['value'] for n in nutrients if n['nutrientName'] == 'Energy' and n['unitName'] == 'KCAL'), 0)
                item_pro = next((n['value'] for n in nutrients if n['nutrientName'] == 'Protein'), 0)
                item_car = next((n['value'] for n in nutrients if n['nutrientName'] == 'Carbohydrate, by difference'), 0)
                
                # Tambahkan ke total keseluruhan
                total_calories += item_cal
                total_protein += item_pro
                total_carbs += item_car
                
                found_items_log.append(f"{item} ({item_cal} kkal)")
            else:
                found_items_log.append(f"{item} (Tidak ditemukan)")
                
        except Exception as e:
            print(f"⚠️ [Tool] Error mengambil data untuk {item}: {e}")
            found_items_log.append(f"{item} (Error API)")

    # Format hasil akhir agar mudah dibaca oleh agen LLM
    summary_text = (
        f"Data ditarik dari USDA untuk {len(items)} item. "
        f"Status: {', '.join(found_items_log)}. "
        f"Estimasi Total: {total_calories:.1f} Kalori, {total_protein:.1f}g Protein, {total_carbs:.1f}g Karbohidrat."
    )
    
    return {"summary": summary_text}