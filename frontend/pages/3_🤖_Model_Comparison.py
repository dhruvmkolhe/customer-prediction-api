"""
Model Comparison Page - ML model training and performance comparison
"""

import streamlit as st
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils"))

from api_client import get_api_client
# pyrefly: ignore [missing-import]
from auth import check_auth
# pyrefly: ignore [missing-import]
from ui_components import material_header
# pyrefly: ignore [missing-import]
from charts import (
    create_model_comparison_chart,
    create_roc_curves,
    create_confusion_matrix,
    create_feature_importance_chart,
)

st.set_page_config(page_title="Model Comparison", page_icon=" ", layout="wide")

material_header("insights", "Model Performance & Comparison", "Compare and analyze multiple predictive models for purchase behavior")
check_auth()

api = get_api_client()

# Training Section
st.subheader("Model Training")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    Train multiple classifiers to predict customer purchase behavior:

    | Model | Description |
    |-------|-------------|
    | **Random Forest** | Ensemble of decision trees, robust to overfitting |
    | **XGBoost** | Gradient boosting, high performance |
    | **Logistic Regression** | Linear model, interpretable |
    | **Gradient Boosting** | Sequential ensemble method |
    """)

with col2:
    st.info("**Training Process:**\n1. Data preprocessing\n2. Feature scaling\n3. Model training\n4. Cross-validation\n5. Evaluation")

if st.button("  Train All Models", use_container_width=True, type="primary"):
    with st.spinner("Training models... This may take a minute."):
        try:
            result = api.train_models()
            st.success(f"Training complete! Best model: **{result['results']['classification']['best_model']}**")
            st.rerun()
        except Exception as e:
            st.error(f"Training failed: {str(e)}")

st.divider()

# Check if models are trained
try:
    metrics_response = api.get_model_metrics()
    classification_metrics = metrics_response.get("classification", {})
    segmentation_metrics = metrics_response.get("segmentation", {})
    models_available = True
except Exception:
    models_available = False

if not models_available:
    st.warning("No trained models available. Please train models first.")
    st.stop()

# Model Performance Overview
st.subheader("Model Performance Overview")

best_model = classification_metrics.get("best_model", "Unknown")
st.success(f"  **Best Model:** {best_model}")

# Metrics comparison table
metrics_data = classification_metrics.get("metrics", {})
models = list(metrics_data.keys())

if models:
    metrics_df = pd.DataFrame(metrics_data).T
    metrics_df = metrics_df[["accuracy", "precision", "recall", "f1", "auc", "cv_auc_mean", "cv_auc_std", "train_time_seconds"]]
    metrics_df.columns = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC", "CV AUC (mean)", "CV AUC (std)", "Train Time (s)"]

    st.dataframe(
        metrics_df.style.format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1 Score": "{:.4f}",
            "AUC-ROC": "{:.4f}",
            "CV AUC (mean)": "{:.4f}",
            "CV AUC (std)": "{:.4f}",
            "Train Time (s)": "{:.2f}",
        }).highlight_max(axis=0, color="#2ECC71"),
        use_container_width=True,
    )

st.divider()

# Visualizations
tab1, tab2, tab3, tab4 = st.tabs(["  Metrics Comparison", "  ROC Curves", "  Confusion Matrices", "  Feature Importance"])

with tab1:
    st.subheader("Model Metrics Comparison")
    if metrics_data:
        fig = create_model_comparison_chart(metrics_data)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("ROC Curves")
    if metrics_data:
        fig = create_roc_curves(metrics_data)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Reading the ROC Curve:**
        - **X-axis (FPR):** False Positive Rate - proportion of negative instances incorrectly classified as positive
        - **Y-axis (TPR):** True Positive Rate (Recall) - proportion of positive instances correctly classified
        - **AUC:** Area Under Curve - higher is better (1.0 = perfect, 0.5 = random)
        """)

with tab3:
    st.subheader("Confusion Matrices")

    if metrics_data:
        cols = st.columns(min(len(models), 4))

        for idx, model_name in enumerate(models[:4]):
            with cols[idx]:
                cm = metrics_data[model_name].get("confusion_matrix", [[0, 0], [0, 0]])
                fig = create_confusion_matrix(cm, model_name)
                st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Feature Importance")

    feature_importance = classification_metrics.get("feature_importance", {})

    if feature_importance:
        selected_model = st.selectbox(
            "Select model for feature importance",
            options=list(feature_importance.keys()),
            index=0
        )

        if selected_model in feature_importance:
            fig = create_feature_importance_chart(feature_importance[selected_model])
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            **Feature Importance Interpretation:**
            - Higher values indicate more important features for prediction
            - Based on impurity (tree-based models) or permutation importance
            """)

# Segmentation Results
st.divider()
st.subheader("  Segmentation Results")

if segmentation_metrics:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Optimal Clusters (K)", segmentation_metrics.get("k_optimal", "N/A"))

    with col2:
        st.metric("Silhouette Score", f"{segmentation_metrics.get('silhouette_score', 0):.4f}")

    with col3:
        st.metric("Total Segments", len(segmentation_metrics.get("segment_mapping", {})))

    # Segment mapping
    st.markdown("**Segment Mapping:**")
    segment_mapping = segmentation_metrics.get("segment_mapping", {})
    if segment_mapping:
        seg_df = pd.DataFrame([
            {"Cluster ID": k, "Segment Name": v}
            for k, v in segment_mapping.items()
        ])
        st.dataframe(seg_df, hide_index=True, use_container_width=False)

    # Silhouette scores by K
    silhouette_by_k = segmentation_metrics.get("silhouette_by_k", {})
    if silhouette_by_k:
        st.markdown("**Silhouette Scores by K:**")
        sil_df = pd.DataFrame([
            {"K": k, "Silhouette Score": v}
            for k, v in silhouette_by_k.items()
        ])
        st.line_chart(sil_df.set_index("K"))

# Training metadata
st.divider()
st.subheader("Training Metadata")

metadata = classification_metrics.get("metadata", {})
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Training Samples", f"{metadata.get('train_size', 'N/A'):,}")

with col2:
    st.metric("Test Samples", f"{metadata.get('test_size', 'N/A'):,}")

with col3:
    st.metric("Features Used", len(metadata.get("feature_columns", [])))

# Model selection recommendation
st.divider()
st.subheader("  Model Recommendation")

if metrics_data:
    # Find best model by different metrics
    best_by_auc = max(metrics_data.items(), key=lambda x: x[1].get("auc", 0))
    best_by_f1 = max(metrics_data.items(), key=lambda x: x[1].get("f1", 0))
    fastest = min(metrics_data.items(), key=lambda x: x[1].get("train_time_seconds", float("inf")))

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(f"**Best AUC-ROC:**\n{best_by_auc[0]}\n({best_by_auc[1]['auc']:.4f})")

    with col2:
        st.info(f"**Best F1 Score:**\n{best_by_f1[0]}\n({best_by_f1[1]['f1']:.4f})")

    with col3:
        st.info(f"**Fastest Training:**\n{fastest[0]}\n({fastest[1]['train_time_seconds']:.2f}s)")
