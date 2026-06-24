"""
Model Training Pipeline for Customer Purchase Behavior Prediction
Includes K-Means segmentation and multiple classifiers for purchase prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, silhouette_score
)
from sklearn.inspection import permutation_importance
import xgboost as xgb
import joblib
import os
import json
from datetime import datetime
from typing import Optional, Any


class ModelTrainer:
    """Trains and evaluates ML models for customer segmentation and purchase prediction."""

    SEGMENT_NAMES = {
        0: "Champions",
        1: "Loyal",
        2: "Potential",
        3: "Promising",
        4: "Customers Needing Attention",
        5: "At Risk",
        6: "Hibernating",
        7: "About to Sleep",
        8: "New",
        9: "One-Time Buyers",
    }

    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = models_dir or os.path.dirname(__file__)
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.segmentation_scaler = StandardScaler()
        self.classification_scaler = StandardScaler()
        self.kmeans_model: Optional[KMeans] = None
        self.classifiers: dict[str, Any] = {}
        self.feature_columns = [
            "recency", "frequency", "monetary", "avg_order_value",
            "purchase_frequency", "product_diversity", "customer_lifetime_days"
        ]

    def _prepare_rfm_features(self, rfm: pd.DataFrame) -> np.ndarray:
        """Standardize RFM features."""
        X_rfm = rfm[self.feature_columns].fillna(0)
        X_rfm = X_rfm.replace([np.inf, -np.inf], 0)
        return self.segmentation_scaler.fit_transform(X_rfm)

    def _find_optimal_k(self, X_scaled: np.ndarray, k_range: range = range(3, 8)) -> tuple[int, dict[str, dict[int, float]]]:
        """Find optimal K using elbow method and silhouette score."""
        inertia_scores: dict[int, float] = {}
        silhouette_scores: dict[int, float] = {}

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            inertia_scores[k] = float(kmeans.inertia_)
            silhouette_scores[k] = float(silhouette_score(X_scaled, labels))

        # Find optimal K (highest silhouette score)
        # Using a more explicit lambda to avoid any key-fetch issues in linters
        optimal_k = max(silhouette_scores, key=lambda k: silhouette_scores[k])

        return optimal_k, {
            "inertia": {k: float(v) for k, v in inertia_scores.items()},
            "silhouette": {k: float(v) for k, v in silhouette_scores.items()}
        }

    def train_segmentation(self, rfm: pd.DataFrame) -> dict:
        """Train K-Means segmentation model."""
        print("Training K-Means segmentation model...")

        X_scaled = self._prepare_rfm_features(rfm)

        # Find optimal K
        optimal_k, eval_metrics = self._find_optimal_k(X_scaled)
        print(f"Optimal K: {optimal_k} (Silhouette: {eval_metrics['silhouette'][optimal_k]:.4f})")

        # Train final model with optimal K
        self.kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        rfm["segment"] = self.kmeans_model.fit_predict(X_scaled)

        # Map segments to meaningful names based on RFM values
        segment_stats = rfm.groupby("segment")[["recency", "frequency", "monetary"]].mean()
        segment_stats["score"] = (
            (1 - segment_stats["recency"] / segment_stats["recency"].max()) * 0.4 +
            segment_stats["frequency"] / segment_stats["frequency"].max() * 0.3 +
            segment_stats["monetary"] / segment_stats["monetary"].max() * 0.3
        )
        sorted_segments = segment_stats.sort_values("score", ascending=False).index.tolist()

        segment_mapping = {}
        names = list(self.SEGMENT_NAMES.values())[:optimal_k]
        for idx, seg in enumerate(sorted_segments):
            segment_mapping[seg] = names[idx] if idx < len(names) else f"Segment {seg}"

        rfm["segment_name"] = rfm["segment"].map(segment_mapping)

        # Save models
        joblib.dump(self.kmeans_model, os.path.join(self.models_dir, "kmeans_segmentation.pkl"))
        joblib.dump(self.segmentation_scaler, os.path.join(self.models_dir, "scaler_segmentation.pkl"))
        joblib.dump(segment_mapping, os.path.join(self.models_dir, "segment_mapping.pkl"))

        return {
            "k_optimal": optimal_k,
            "silhouette_score": round(eval_metrics["silhouette"][optimal_k], 4),
            "inertia": {str(k): round(v, 2) for k, v in eval_metrics["inertia"].items()},
            "silhouette_by_k": {str(k): round(v, 4) for k, v in eval_metrics["silhouette"].items()},
            "segment_mapping": segment_mapping,
            "segment_distribution": rfm["segment_name"].value_counts().to_dict()
        }

    def _create_purchase_target(self, rfm: pd.DataFrame) -> pd.Series:
        """Create binary target: will customer purchase in next 30 days?"""
        # Simulate based on RFM patterns
        # High frequency + recent = likely to purchase
        purchase_prob = (
            (1 / (rfm["recency"] + 1)) * 0.4 +
            (rfm["frequency"] / rfm["frequency"].max()) * 0.3 +
            (rfm["purchase_frequency"] / rfm["purchase_frequency"].max()) * 0.3
        )
        purchase_prob = purchase_prob / purchase_prob.max()
        will_purchase = (purchase_prob > 0.4).astype(int)
        return will_purchase

    def train_classifiers(self, rfm: pd.DataFrame) -> dict:
        """Train multiple classifiers for purchase prediction."""
        print("Training classifiers...")

        # Add segment as feature
        if "segment" not in rfm.columns:
            X_scaled = self.segmentation_scaler.transform(rfm[self.feature_columns].fillna(0))
            rfm["segment"] = self.kmeans_model.predict(X_scaled)

        # Prepare features
        features = self.feature_columns + ["segment"]
        X = rfm[features].copy()
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        y = self._create_purchase_target(rfm)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.classification_scaler.fit_transform(X_train)
        X_test_scaled = self.classification_scaler.transform(X_test)
        
        # Save classification scaler
        joblib.dump(self.classification_scaler, os.path.join(self.models_dir, "scaler_classification.pkl"))

        # Define models
        models = {
            "RandomForest": RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
            "XGBoost": xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=42, use_label_encoder=False, eval_metric="logloss"
            ),
            "LogisticRegression": LogisticRegression(
                max_iter=1000, random_state=42
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=100, max_depth=5, random_state=42
            )
        }

        results = {}
        best_auc = 0
        best_model_name = None

        for name, model in models.items():
            print(f"Training {name}...")
            start_time = datetime.now()

            # Train
            if name == "LogisticRegression":
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]

            train_time = (datetime.now() - start_time).total_seconds()

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)

            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled if name.lower() in ["logisticregression"] else X_train,
                                       y_train, cv=5, scoring="roc_auc")

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred).tolist()

            # ROC curve data
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

            results[name] = {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "auc": round(auc, 4),
                "cv_auc_mean": round(cv_scores.mean(), 4),
                "cv_auc_std": round(cv_scores.std(), 4),
                "train_time_seconds": round(train_time, 2),
                "confusion_matrix": cm,
                "roc_data": roc_data,
            }

            self.classifiers[name] = model

            if auc > best_auc:
                best_auc = auc
                best_model_name = name

        # Feature importance (for tree-based models)
        feature_importance = {}
        for name in ["RandomForest", "XGBoost"]:
            if hasattr(self.classifiers[name], "feature_importances_"):
                importance = self.classifiers[name].feature_importances_
                feature_importance[name] = {
                    feat: round(float(imp), 4)
                    for feat, imp in zip(features, importance)
                }

        # Save models
        for name, model in self.classifiers.items():
            joblib.dump(model, os.path.join(self.models_dir, f"{name.lower()}.pkl"))

        # Save metadata
        metadata = {
            "feature_columns": features,
            "best_model": best_model_name,
            "target_distribution": y.value_counts().to_dict(),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }
        with open(os.path.join(self.models_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        return {
            "models_trained": list(results.keys()),
            "best_model": best_model_name,
            "metrics": results,
            "feature_importance": feature_importance,
            "metadata": metadata,
        }

    def train(self, rfm: pd.DataFrame) -> dict:
        """Complete training pipeline."""
        print("=" * 50)
        print("Starting Model Training Pipeline")
        print("=" * 50)

        # Step 1: Segmentation
        segmentation_results = self.train_segmentation(rfm)

        # Step 2: Classification
        classification_results = self.train_classifiers(rfm)

        # Combine results
        results = {
            "segmentation": segmentation_results,
            "classification": classification_results,
            "timestamp": datetime.now().isoformat(),
        }

        # Save full results
        with open(os.path.join(self.models_dir, "training_results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)

        print("=" * 50)
        print(f"Training Complete! Best Model: {classification_results['best_model']}")
        print("=" * 50)

        return results


if __name__ == "__main__":
    # Load data
    rfm = pd.read_csv("data/rfm_features.csv")

    # Train models
    trainer = ModelTrainer()
    results = trainer.train(rfm)

    print("\nSegmentation Results:")
    print(f"  Optimal K: {results['segmentation']['k_optimal']}")
    print(f"  Silhouette Score: {results['segmentation']['silhouette_score']}")

    print("\nClassification Results:")
    print(f"  Best Model: {results['classification']['best_model']}")
    for name, metrics in results['classification']['metrics'].items():
        print(f"  {name}: AUC={metrics['auc']}, F1={metrics['f1']}")
