import pytest
import sys
import os
import numpy as np

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from predictor import Predictor

def test_predictor_loading():
    """Test if predictor loads models correctly."""
    models_dir = os.path.dirname(os.path.dirname(__file__))
    predictor = Predictor(models_dir)
    
    # We expect models to be loaded if they were trained
    assert hasattr(predictor, "segmentation_scaler")
    assert hasattr(predictor, "classification_scaler")
    assert hasattr(predictor, "kmeans")

def test_dual_scaler_logic():
    """Test that different scalers are used for different steps."""
    models_dir = os.path.dirname(os.path.dirname(__file__))
    predictor = Predictor(models_dir)
    
    if predictor.models_loaded:
        # Check that they are different instances
        assert predictor.segmentation_scaler is not predictor.classification_scaler
        
        # Check feature counts expected by scalers
        # segmentation_scaler should expect 7 features
        # classification_scaler should expect 8 features
        assert predictor.segmentation_scaler.n_features_in_ == 7
        assert predictor.classification_scaler.n_features_in_ == 8

def test_predict_segment_consistency():
    """Test that segment prediction returns expected structure."""
    models_dir = os.path.dirname(os.path.dirname(__file__))
    predictor = Predictor(models_dir)
    
    if predictor.models_loaded:
        test_features = {
            "recency": 10, "frequency": 5, "monetary": 500,
            "avg_order_value": 100, "purchase_frequency": 2,
            "product_diversity": 0.5, "customer_lifetime_days": 100
        }
        result = predictor.predict_segment(test_features)
        assert "segment_name" in result
        assert "segment_id" in result
        assert "confidence" in result
