from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_synthesize_endpoint():
    response = client.post("/api/synthesize", json={"arxiv_input": "2312.00752"})
    assert response.status_code == 200
    data = response.json()
    assert "core_equation" in data
    assert "generated_endpoint" in data

def test_simulate_endpoint():
    response = client.post("/api/simulate", json={
        "arxiv_id": "2312.00752",
        "param_alpha": 32.0,
        "param_beta": 0.5,
        "input_text": "test tensor sequence"
    })
    assert response.status_code == 200
    assert response.json()["status"] == "VERIFIED_OK"
