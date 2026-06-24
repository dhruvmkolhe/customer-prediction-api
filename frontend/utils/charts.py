"""
Visualization utilities using Plotly.
Version: 1.1 (Elite Features)
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


# Color scheme
COLORS = {
    "primary": "#FF6B6B",
    "secondary": "#4ECDC4",
    "accent": "#45B7D1",
    "success": "#2ECC71",
    "warning": "#F39C12",
    "danger": "#E74C3C",
    "background": "#0E1117",
    "text": "#FAFAFA",
}

SEGMENT_COLORS = {
    "Champions": "#2ECC71",
    "Loyal": "#3498DB",
    "At Risk": "#F39C12",
    "Hibernating": "#E74C3C",
    "New": "#9B59B6",
}


def create_rfm_scatter_3d(rfm_data: pd.DataFrame) -> go.Figure:
    """Create 3D scatter plot of RFM features."""
    fig = px.scatter_3d(
        rfm_data,
        x="recency",
        y="frequency",
        z="monetary",
        color="segment_name",
        color_discrete_map=SEGMENT_COLORS,
        opacity=0.7,
        title="Customer Segments (RFM Analysis)",
        labels={
            "recency": "Recency (days)",
            "frequency": "Frequency",
            "monetary": "Monetary ($)"
        }
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="Recency (days)",
            yaxis_title="Frequency",
            zaxis_title="Monetary ($)"
        ),
        template="plotly_dark",
        height=500,
    )
    return fig


def create_segment_distribution(segments: list) -> go.Figure:
    """Create pie chart for segment distribution."""
    names = [s["name"] for s in segments]
    counts = [s["count"] for s in segments]
    colors = [SEGMENT_COLORS.get(name, "#888888") for name in names]

    fig = go.Figure(data=[go.Pie(
        labels=names,
        values=counts,
        hole=0.4,
        marker=dict(colors=colors),
        textinfo="label+percent",
        textfont_size=12,
    )])
    fig.update_layout(
        title="Customer Segment Distribution",
        template="plotly_dark",
        height=400,
    )
    return fig


def create_rfm_heatmap(segments: list) -> go.Figure:
    """Create heatmap of average RFM values by segment."""
    segment_names = [s["name"] for s in segments]
    metrics = ["Recency", "Frequency", "Monetary"]
    z_values = [
        [s["avg_recency"] for s in segments],
        [s["avg_frequency"] for s in segments],
        [s["avg_monetary"] for s in segments],
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=segment_names,
        y=metrics,
        colorscale="Viridis",
        text=np.round(z_values, 2),
        texttemplate="%{text}",
        textfont={"size": 12},
    ))
    fig.update_layout(
        title="RFM Metrics by Segment",
        template="plotly_dark",
        height=350,
    )
    return fig


def create_segment_bar_chart(segments: list) -> go.Figure:
    """Create bar chart for segment metrics."""
    names = [s["name"] for s in segments]
    revenue = [s["total_revenue"] for s in segments]
    colors = [SEGMENT_COLORS.get(name, "#888888") for name in names]

    fig = go.Figure(data=[go.Bar(
        x=names,
        y=revenue,
        marker_color=colors,
        text=[f"${r:,.0f}" for r in revenue],
        textposition="outside",
    )])
    fig.update_layout(
        title="Total Revenue by Segment",
        xaxis_title="Segment",
        yaxis_title="Revenue ($)",
        template="plotly_dark",
        height=400,
    )
    return fig


def create_model_comparison_chart(metrics: dict) -> go.Figure:
    """Create bar chart comparing model metrics."""
    models = list(metrics.keys())
    metric_names = ["accuracy", "precision", "recall", "f1", "auc"]

    fig = go.Figure()
    for metric in metric_names:
        values = [metrics[m].get(metric, 0) for m in models]
        fig.add_trace(go.Bar(
            name=metric.upper(),
            x=models,
            y=values,
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
        ))

    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Score",
        barmode="group",
        template="plotly_dark",
        height=450,
    )
    return fig


def create_roc_curves(metrics: dict) -> go.Figure:
    """Create ROC curves for all models."""
    fig = go.Figure()

    for model_name, model_metrics in metrics.items():
        roc_data = model_metrics.get("roc_data", {})
        fpr = roc_data.get("fpr", [])
        tpr = roc_data.get("tpr", [])
        auc = model_metrics.get("auc", 0)

        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            name=f"{model_name} (AUC = {auc:.3f})",
            mode="lines",
        ))

    # Diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        name="Random",
        mode="lines",
        line=dict(dash="dash", color="gray"),
    ))

    fig.update_layout(
        title="ROC Curves",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_dark",
        height=450,
    )
    return fig


def create_confusion_matrix(cm: list, model_name: str) -> go.Figure:
    """Create confusion matrix heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Predicted: No", "Predicted: Yes"],
        y=["Actual: No", "Actual: Yes"],
        colorscale="Blues",
        text=np.array(cm),
        texttemplate="%{text}",
        textfont={"size": 14},
    ))
    fig.update_layout(
        title=f"Confusion Matrix - {model_name}",
        template="plotly_dark",
        height=350,
    )
    return fig


def create_feature_importance_chart(importance: dict) -> go.Figure:
    """Create horizontal bar chart for feature importance."""
    if isinstance(importance, dict) and "importance" in importance:
        raw_items = importance["importance"]
        if raw_items and isinstance(raw_items[0], dict):
            features = [item.get("feature", "") for item in raw_items]
            values = [item.get("importance", 0) for item in raw_items]
        else:
            features = list(importance.keys())
            values = list(importance.values())
    else:
        features = list(importance.keys())
        values = list(importance.values())

    if not features:
        return go.Figure()

    # Sort by importance
    sorted_data = sorted(zip(features, values), key=lambda x: x[1])
    features, values = zip(*sorted_data)

    fig = go.Figure(data=[go.Bar(
        y=features,
        x=values,
        orientation="h",
        marker_color=COLORS["accent"],
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
    )])
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Importance",
        yaxis_title="Feature",
        template="plotly_dark",
        height=400,
    )
    return fig


def create_monthly_revenue_chart(data: list) -> go.Figure:
    """Create line chart for monthly revenue trends."""
    df = pd.DataFrame(data)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["month"],
        y=df["revenue"],
        name="Revenue",
        mode="lines+markers",
        line=dict(color=COLORS["primary"], width=2),
    ))
    fig.update_layout(
        title="Monthly Revenue Trend",
        xaxis_title="Month",
        yaxis_title="Revenue ($)",
        template="plotly_dark",
        height=400,
    )
    return fig


def create_rfm_distributions(rfm_data: pd.DataFrame) -> go.Figure:
    """Create histograms for RFM distributions."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Recency Distribution", "Frequency Distribution", "Monetary Distribution")
    )

    fig.add_trace(go.Histogram(
        x=rfm_data["recency"],
        name="Recency",
        marker_color=COLORS["primary"],
        nbinsx=30,
    ), row=1, col=1)

    fig.add_trace(go.Histogram(
        x=rfm_data["frequency"],
        name="Frequency",
        marker_color=COLORS["secondary"],
        nbinsx=30,
    ), row=1, col=2)

    fig.add_trace(go.Histogram(
        x=rfm_data["monetary"],
        name="Monetary",
        marker_color=COLORS["accent"],
        nbinsx=30,
    ), row=1, col=3)

    fig.update_layout(
        title="RFM Distributions",
        template="plotly_dark",
        height=350,
        showlegend=False,
    )
    return fig


def create_prediction_gauge(probability: float) -> go.Figure:
    """Create gauge chart for purchase probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={"text": "Purchase Probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": COLORS["primary"]},
            "steps": [
                {"range": [0, 30], "color": "#E74C3C"},
                {"range": [30, 70], "color": "#F39C12"},
                {"range": [70, 100], "color": "#2ECC71"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 4},
                "thickness": 0.75,
                "value": probability * 100,
            },
        }
    ))
    fig.update_layout(
        height=250,
        template="plotly_dark",
    )
    return fig


def create_category_distribution(categories: dict) -> go.Figure:
    """Create bar chart for product category distribution."""
    fig = go.Figure(data=[go.Bar(
        x=list(categories.keys()),
        y=list(categories.values()),
        marker_color=COLORS["accent"],
        text=list(categories.values()),
        textposition="outside",
    )])
    fig.update_layout(
        title="Product Category Distribution",
        xaxis_title="Category",
        yaxis_title="Count",
        template="plotly_dark",
        height=350,
    )
    return fig


def create_affinity_heatmap(affinity_data: list) -> go.Figure:
    """Create heatmap of product category co-occurrence."""
    if not affinity_data: return go.Figure()
    
    df = pd.DataFrame(affinity_data)
    if df.empty: return go.Figure()

    # Create pivot table for heatmap
    categories = sorted(list(set(df["category_a"].unique()) | set(df["category_b"].unique())))
    matrix = pd.DataFrame(0, index=categories, columns=categories)
    
    for _, row in df.iterrows():
        matrix.loc[row["category_a"], row["category_b"]] = row["co_occurrence"]
        matrix.loc[row["category_b"], row["category_a"]] = row["co_occurrence"]
        
    fig = go.Figure(data=go.Heatmap(
        z=matrix.values,
        x=matrix.columns,
        y=matrix.index,
        colorscale="Viridis",
        text=matrix.values,
        texttemplate="%{text}",
        hoverinfo="z",
    ))
    
    fig.update_layout(
        title="Product Category Affinity (Market Basket Analysis)",
        xaxis_title="Category",
        yaxis_title="Category",
        template="plotly_dark",
        height=500,
    )
    return fig
