"""
File ini menguji validasi data (Pydantic Schema).
Hanya mengecek apakah data yang dibuat sudah sesuai aturan.
"""

import pytest
from pydantic import ValidationError
from src.agents.schemas import IntentClassification, FoodItem


# TEST 1: Cek intent valid
def test_intent_valid():
    # Membuat objek intent dengan isi 'track_diet'
    data = IntentClassification(intent="track_diet")

    # Cek apakah isinya sesuai
    assert data.intent == "track_diet"


# TEST 2: Cek error jika intent lupa diisi
def test_intent_error_jika_kosong():
    # Jika tidak mengisi intent, Python harus mengembalikan ValidationError
    with pytest.raises(ValidationError):
        IntentClassification()


# TEST 3: Cek pembuatan data makanan
def test_buat_item_makanan():
    makanan = FoodItem(asli="nasi goreng", english="fried rice")

    assert makanan.asli == "nasi goreng"
    assert makanan.english == "fried rice"
