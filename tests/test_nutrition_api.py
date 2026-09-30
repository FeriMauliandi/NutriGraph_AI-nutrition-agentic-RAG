"""
File ini menguji perhitungan nutrisi tanpa benar-benar memanggil internet (menggunakan Mock sederhana).
"""

from unittest.mock import patch
from src.tools.nutrition_api import fetch_combined_nutrition_data


# TEST 1: Cek perhitungan total kalori dan porsi
@patch("src.tools.nutrition_api.get_usda_item")
def test_kalkulasi_total_nutrisi(mock_usda):
    # Kita pura-pura (mock) bahwa API USDA selalu merespon: 200 kalori, 10g protein, 30g karbo
    mock_usda.return_value = {
        "found": True, "cal": 200, "pro": 10, "car": 30, "source": "USDA"
    }

    # Data makanan yang dikirim: 2 porsi nasi goreng
    makanan = [{"asli": "nasi goreng", "english": "fried rice", "quantity": 2}]

    # Panggil fungsi hitung nutrisi
    hasil = fetch_combined_nutrition_data(makanan)

    # Kalori harusnya: 200 x 2 = 400.0 Kalori
    assert "400.0 Kalori" in hasil["summary"]
    assert "2x nasi goreng" in hasil["summary"]


# TEST 2: Cek jika makanan tidak ditemukan di database API
@patch("src.tools.nutrition_api.get_usda_item")
def test_makanan_tidak_ditemukan(mock_usda):
    # Pura-pura API tidak menemukan makanan (found: False)
    mock_usda.return_value = {"found": False}

    makanan = [{"asli": "makanan_aneh", "english": "unknown_food", "quantity": 1}]

    hasil = fetch_combined_nutrition_data(makanan)

    assert "Tidak ditemukan" in hasil["summary"]
