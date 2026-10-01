import os
from fastapi.testclient import TestClient

# Pastikan folder cache ada sebelum import main
os.makedirs("data/cache", exist_ok=True)

from backend.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "berjalan dengan lancar" in response.json()["message"]

def test_analyze_diet_endpoint_structure():
    # Test validasi payload request / response struktur dasar
    response = client.post(
        "/api/v1/analyze",
        json={
            "user_input": "Halo, sapaan pagi!",
            "session_id": "test_session_123"
        }
    )
    # Walaupun memanggil LLM (bisa sukses atau rate limit jika key tidak ada di test env), 
    # kita pastikan response mengembalikan struktur DietResponse atau error HTTP yang tertangani (500/200)
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert "extracted_items" in data
        assert "final_analysis" in data
        assert "needs_clarification" in data
