"""
File ini berisi unit test sederhana untuk fungsi-fungsi helper (utils).
Format test paling simpel:
1. Buat input
2. Panggil fungsi
3. Cek hasil dengan 'assert'
"""

from src.agents.utils import (
    normalize_extracted_items,
    format_detected_items,
    fallback_extract_items,
    infer_item_quantities,
)


# TEST 1: Cek apakah nama makanan otomatis jadi huruf kecil & menghapus duplikat
def test_normalize_makanan():
    # 1. Input: Data mentah dengan huruf besar & duplikat
    data_input = [
        {"asli": "Nasi Goreng", "english": "Fried Rice"},
        {"asli": "nasi goreng", "english": "fried rice"},  # duplikat
    ]

    # 2. Proses: Jalankan fungsi
    hasil = normalize_extracted_items(data_input)

    # 3. Cek: Harus cuma 1 item dan namanya huruf kecil semua ("nasi goreng")
    assert len(hasil) == 1
    assert hasil[0]["asli"] == "nasi goreng"


# TEST 2: Cek penterjemahan otomatis jika bahasa Inggris kosong
def test_terjemahan_otomatis():
    # Input makanan "tahu" tanpa bahasa Inggris
    data_input = [{"asli": "tahu", "english": ""}]

    hasil = normalize_extracted_items(data_input)

    # Harus otomatis terisi "tofu"
    assert hasil[0]["english"] == "tofu"


# TEST 3: Cek fungsi pemformat teks makanan
def test_format_teks_makanan():
    data_input = [
        {"asli": "nasi goreng"},
        {"asli": "es teh"},
    ]

    hasil = format_detected_items(data_input)

    # Hasilnya harus berupa teks gabungan yang dipisah koma
    assert hasil == "nasi goreng, es teh"


# TEST 4: Cek pembacaan jumlah porsi dari teks
def test_baca_jumlah_porsi():
    # Input makanan awal porsi = 1
    makanan = [{"asli": "nasi goreng", "english": "fried rice", "quantity": 1.0}]

    # User mengetik "2 porsi nasi goreng"
    hasil = infer_item_quantities("2 porsi nasi goreng", makanan)

    # Porsi harus otomatis berubah jadi 2
    assert hasil[0]["quantity"] == 2.0


# TEST 5: Cek ekstraksi darurat jika AI gagal (fallback)
def test_ekstrak_darurat():
    teks_user = "saya makan nasi goreng dan teh manis"

    hasil = fallback_extract_items(teks_user)

    # Harus mendapatkan minimal 2 makanan
    assert len(hasil) == 2
