"""
Prediction Service for Customer Purchase Behavior
Loads trained models and provides real-time predictions.
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
from typing import Optional, Any


class Predictor:
    """Provides real-time predictions using trained models."""

    RECOMMENDATIONS = {
        "Champions": "Send exclusive VIP offers and early access to new products",
        "Loyal": "Implement loyalty rewards program with personalized discounts",
        "At Risk": "Send win-back campaigns with special offers",
        "Hibernating": "Re-engagement campaign with strong incentives",
        "New": "Welcome series with product recommendations",
    }

    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = models_dir or os.path.dirname(__file__)
        self.models_loaded = False
        self.segmentation_scaler: Any = None
        self.classification_scaler: Any = None
        self.kmeans: Any = None
        self.segment_mapping: dict[int, str] = {}
        self.classifiers: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {}
        self.feature_columns = [
            "recency", "frequency", "monetary", "avg_order_value",
            "purchase_frequency", "product_diversity", "customer_lifetime_days"
        ]
        self.default_model_key: Optional[str] = None
        self.load_models()

    @staticmethod
    def _normalize_model_name(model_name: Optional[str]) -> str:
        """Normalize user-facing model names to internal keys."""
        if not model_name:
            return ""
        return model_name.lower().replace(" ", "")

    def load_models(self) -> bool:
        """Load all trained models and artifacts."""
        self.models_loaded = False
        self.classifiers = {}
        try:
            self.segmentation_scaler = joblib.load(os.path.join(self.models_dir, "scaler_segmentation.pkl"))
            self.classification_scaler = joblib.load(os.path.join(self.models_dir, "scaler_classification.pkl"))
            self.kmeans = joblib.load(os.path.join(self.models_dir, "kmeans_segmentation.pkl"))
            self.segment_mapping = joblib.load(os.path.join(self.models_dir, "segment_mapping.pkl"))

            metadata_path = os.path.join(self.models_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    self.metadata = json.load(f)
                self.feature_columns = self.metadata.get("feature_columns", self.feature_columns)
            else:
                self.metadata = {}

            # Load classifiers individually so one bad artifact does not block all predictions.
            for name in ["randomforest", "xgboost", "logisticregression", "gradientboosting"]:
                path = os.path.join(self.models_dir, f"{name}.pkl")
                if os.path.exists(path):
                    try:
                        self.classifiers[name] = joblib.load(path)
                    except Exception as model_error:
                        print(f"Skipping model '{name}' because it could not be loaded: {model_error}")

            if not self.classifiers:
                print("No classifiers could be loaded.")
                return False

            best_model_key = self._normalize_model_name(self.metadata.get("best_model"))
            if best_model_key in self.classifiers:
                self.default_model_key = best_model_key
            else:
                self.default_model_key = next(iter(self.classifiers))

            self.models_loaded = True
            print(f"Loaded models: {list(self.classifiers.keys())}")
            print(f"Default model: {self.default_model_key}")
            return True
        except Exception as e:
            import traceback
            print(f"Error loading models: {e}")
            traceback.print_exc()
            self.models_loaded = False
            return False

    def _validate_input(self, data: dict[str, Any]) -> tuple[bool, str]:
        """Validate input data."""
        required_fields = [
            "recency", "frequency", "monetary", "avg_order_value",
            "purchase_frequency", "product_diversity", "customer_lifetime_days"
        ]

        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
            if not isinstance(data[field], (int, float)):
                try:
                    float(data[field])
                except (ValueError, TypeError):
                    return False, f"Invalid value for {field}: {data[field]}"

        return True, "Valid"

    def predict_segment(self, features: dict[str, Any]) -> dict[str, Any]:
        """Predict customer segment using K-Means."""
        if self.segmentation_scaler is None or self.kmeans is None:
            raise RuntimeError("Segmentation artifacts are not loaded.")

        segment_feature_names = [
            "recency", "frequency", "monetary", "avg_order_value",
            "purchase_frequency", "product_diversity", "customer_lifetime_days",
        ]
        feature_values = [
            features.get("recency", 0),
            features.get("frequency", 0),
            features.get("monetary", 0),
            features.get("avg_order_value", 0),
            features.get("purchase_frequency", 0),
            features.get("product_diversity", 0),
            features.get("customer_lifetime_days", 0),
        ]

        X = pd.DataFrame([feature_values], columns=segment_feature_names)
        X_scaled = self.segmentation_scaler.transform(X)

        segment_id = int(self.kmeans.predict(X_scaled)[0])
        segment_name = self.segment_mapping.get(segment_id, f"Segment {segment_id}")

        # Calculate distance to cluster center for confidence
        distances = self.kmeans.transform(X_scaled)[0]
        confidence = 1 / (1 + distances[segment_id])

        return {
            "segment_id": segment_id,
            "segment_name": segment_name,
            "confidence": round(float(confidence), 4),
        }

    def predict_purchase(
        self,
        features: dict[str, Any],
        model_name: Optional[str] = None
    ) -> dict[str, Any]:
        """Predict purchase probability using specified or best model."""
        if model_name is None:
            model_name = self.metadata.get("best_model") or self.default_model_key or "randomforest"

        # Map model name to key
        model_key = self._normalize_model_name(model_name)
        if not model_key:
            model_key = self.default_model_key or "randomforest"

        if model_key not in self.classifiers:
            return {
                "error": f"Model {model_name} not found. Available: {list(self.classifiers.keys())}"
            }

        model = self.classifiers[model_key]

        # Get segment first
        segment_info = self.predict_segment(features)

        # Prepare features
        feature_values = [
            features.get("recency", 0),
            features.get("frequency", 0),
            features.get("monetary", 0),
            features.get("avg_order_value", 0),
            features.get("purchase_frequency", 0),
            features.get("product_diversity", 0),
            features.get("customer_lifetime_days", 0),
            segment_info["segment_id"],  # Add segment as feature
        ]

        X = pd.DataFrame([feature_values], columns=self.feature_columns)
        
        # Scale for LogisticRegression
        if model_key == "logisticregression":
            X_input = self.classification_scaler.transform(X)
        else:
            X_input = X
            
        # Predict
        will_purchase = bool(model.predict(X_input)[0])
        probabilities = model.predict_proba(X_input)[0]
        probability = float(probabilities[1])  # Probability of class 1 (will purchase)

        # Get recommendation
        recommendation = self.RECOMMENDATIONS.get(
            segment_info["segment_name"],
            "Analyze customer behavior for personalized approach"
        )

        return {
            "segment": segment_info,
            "prediction": {
                "will_purchase": will_purchase,
                "probability": round(probability, 4),
                "model_used": model_name,
            },
            "recommendation": recommendation,
        }

    def get_feature_importance(self, model_name: str = "randomforest") -> dict[str, Any]:
        """Get global feature importance for the specified model."""
        model_key = self._normalize_model_name(model_name)
        if not self.models_loaded or model_key not in self.classifiers:
            return {}

        model = self.classifiers[model_key]
        importance: list[dict[str, Any]] = []

        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based model importance
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear model coefficients (normalized as absolute weights)
                coeffs = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                importances = np.abs(coeffs) / np.sum(np.abs(coeffs))
            else:
                return {}

            # Defensive check for feature column length
            n_features = len(importances)
            n_cols = len(self.feature_columns)
            
            # Use min length to avoid IndexError
            for i in range(min(n_features, n_cols)):
                importance.append({
                    "feature": self.feature_columns[i],
                    "importance": round(float(importances[i]), 4)
                })

            # Sort by importance descending
            importance.sort(key=lambda x: x["importance"], reverse=True)
            return {"model": model_key, "importance": importance}

        except Exception as e:
            print(f"Importance calculation error: {e}")
            return {}

    def predict_batch(self, customers: list[dict[str, Any]], model_name: Optional[str] = None) -> list[dict[str, Any]]:
        """Predict for multiple customers."""
        results = []
        for customer in customers:
            result = self.predict_purchase(customer, model_name)
            result["customer_id"] = customer.get("customer_id", "unknown")
            results.append(result)
        return results

    def get_model_info(self) -> dict[str, Any]:
        """Get information about loaded models."""
        if not self.models_loaded:
            return {"error": "Models not loaded"}

        return {
            "models_available": list(self.classifiers.keys()),
            "best_model": self.metadata.get("best_model"),
            "feature_columns": self.feature_columns,
            "segments": list(self.segment_mapping.values()),
        }


if __name__ == "__main__":
    predictor = Predictor(models_dir="models_current")

    if predictor.models_loaded:
        # Test prediction
        test_features = {
            "recency": 15,
            "frequency": 8,
            "monetary": 1250.50,
            "avg_order_value": 156.31,
            "purchase_frequency": 2.3,
            "product_diversity": 0.65,
            "customer_lifetime_days": 365,
        }

        result = predictor.predict_purchase(test_features)
        print("\nPrediction Result:")
        print(json.dumps(result, indent=2))
    else:
        print("Models not loaded. Please train models first.")
