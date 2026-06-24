"""
API Client for communicating with the FastAPI backend.
"""

import requests
import streamlit as st
from typing import Optional, Dict, Any


class APIClient:
    """Client for interacting with the Customer Prediction API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        timeout: int = 60
    ) -> Dict[str, Any]:
        """Make an HTTP request to the API."""
        url = f"{self.base_url}{endpoint}"

        try:
            if method == "GET":
                response = self.session.get(url, params=params, timeout=timeout)
            elif method == "POST":
                response = self.session.post(url, json=data, timeout=timeout)
            else:
                raise ValueError(f"Unsupported method: {method}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to API. Make sure the backend is running on {self.base_url}")
            raise Exception(f"Failed to connect to the backend at {self.base_url}. Please ensure the server is running.")
        except requests.exceptions.Timeout:
            raise Exception(f"Request timed out after {timeout} seconds. The backend might be busy or performing a long-running operation.")
        except requests.exceptions.HTTPError as e:
            error_detail = e.response.json().get("detail", str(e)) if e.response else str(e)
            st.error(f"API Error: {error_detail}")
            raise
        except Exception as e:
            raise Exception(f"API Request failed: {str(e)}")

    # Data endpoints
    def generate_data(self, n_customers: int = 10000, n_transactions: int = 50000) -> Dict:
        """Generate mock data."""
        return self._make_request("POST", "/generate-data", data={
            "n_customers": n_customers,
            "n_transactions": n_transactions
        }, timeout=300)

    def simulate_traffic(self, count: int = 50) -> Dict:
        """Generate random transactions for existing customers to simulate live traffic."""
        return self._make_request("POST", "/simulate-traffic", params={"count": count})

    def get_model_importance(self, model_name: str = "randomforest") -> Dict:
        """Get feature importance for a specific model."""
        return self._make_request("GET", "/models/importance", params={"model_name": model_name})

    def get_product_affinity(self) -> list:
        """Get product category co-occurrence data."""
        return self._make_request("GET", "/data/affinity")

    def get_data_summary(self) -> Dict:
        """Get dataset statistics."""
        return self._make_request("GET", "/data/summary")

    def get_rfm_data(self, limit: int = 100, offset: int = 0) -> Dict:
        """Get RFM data with pagination."""
        return self._make_request("GET", "/data/rfm", params={"limit": limit, "offset": offset})

    def get_transactions(self, limit: int = 100, offset: int = 0) -> Dict:
        """Get transaction data."""
        return self._make_request("GET", "/data/transactions", params={"limit": limit, "offset": offset})

    def get_monthly_revenue(self) -> Dict:
        """Get monthly revenue trends."""
        return self._make_request("GET", "/data/monthly-revenue")

    # Training endpoints
    def train_models(self) -> Dict:
        """Train all ML models."""
        return self._make_request("POST", "/train", timeout=300)

    def get_model_metrics(self) -> Dict:
        """Get model comparison metrics."""
        return self._make_request("GET", "/models/metrics")

    # Prediction endpoints
    def predict_single(self, features: Dict) -> Dict:
        """Predict for a single customer (Alias for predict)."""
        return self._make_request("POST", "/predict", data=features)

    def predict_batch(self, customers: list, model_name: Optional[str] = None) -> Dict:
        """Predict for multiple customers."""
        return self._make_request("POST", "/predict/batch", data={
            "customers": customers,
            "model_name": model_name
        }, timeout=300)

    # Segment endpoints
    def get_segments(self) -> Dict:
        """Get all segments."""
        return self._make_request("GET", "/segments")

    def get_segment_details(self, segment_id: int) -> Dict:
        """Get segment details."""
        return self._make_request("GET", f"/segments/{segment_id}")

    def generate_strategy(self, segment_details: Dict) -> Dict:
        """Generate marketing strategy for a segment using AI."""
        return self._make_request("POST", "/generate-strategy", data=segment_details, timeout=30)

    # Health check
    def health_check(self) -> Dict:
        """Check API health."""
        return self._make_request("GET", "/")

    # Live prediction endpoints
    def get_live_predictions(self, limit: int = 50) -> Dict:
        """Get recent real-time ML predictions."""
        return self._make_request("GET", "/live-predictions", params={"limit": limit})

    def get_live_stats(self) -> Dict:
        """Get aggregate statistics from live predictions."""
        return self._make_request("GET", "/live-stats")


def get_api_client():
    """Get cached API client instance with environment-aware URL."""
    import os
    # Priority: 1. Env Var, 2. Streamlit Secrets, 3. Default (Internal 8000)
    
    # Priority 1: Direct Environment Variable
    base_url = os.environ.get('API_URL')
    
    # Priority 2: Streamlit Secrets (for Streamlit Cloud or Render secrets)
    if not base_url:
        try:
            base_url = st.secrets.get('API_URL')
        except:
            base_url = None
            
    # Priority 3: Smart Fallback
    if not base_url:
        # If running on Render (co-located), use localhost on the internal backend port
        if os.environ.get('RENDER') or os.path.exists('/opt/render'):
            base_url = 'http://127.0.0.1:8000'
        else:
            # Local development default
            base_url = 'http://127.0.0.1:8000'
        
    return APIClient(base_url=base_url)
