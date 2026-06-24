import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add backend to path so we can import main
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from main import app

client = TestClient(app)

def test_health_check():
    """Test the root health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "running"

def test_data_summary():
    """Test the data summary endpoint."""
    response = client.get("/data/summary")
    # This might be 200 or 404 depending on if data exists
    assert response.status_code in [200, 404]

def test_segments_list():
    """Test the segments list endpoint."""
    response = client.get("/segments")
    assert response.status_code in [200, 404]

def test_prediction_endpoint():
    """Test the single prediction endpoint with sample data."""
    sample_payload = {
        "customer_id": 10001,
        "recency": 10,
        "frequency": 5,
        "monetary": 500.0,
        "avg_order_value": 100.0,
        "purchase_frequency": 2.0,
        "product_diversity": 0.5,
        "customer_lifetime_days": 100
    }
    response = client.post("/predict", json=sample_payload)
    # Status should be 200 if models are loaded, or 400 if not
    assert response.status_code in [200, 400]
    
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "probability" in data["prediction"]
        assert "segment" in data
