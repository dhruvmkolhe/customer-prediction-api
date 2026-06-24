"""
Live Traffic Simulator Page with Real-Time ML Predictions
"""

import streamlit as st
import pandas as pd
import sys
import os
import time
import random
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils"))

from api_client import get_api_client
from auth import check_auth
from ui_components import material_header

st.set_page_config(page_title="Live Traffic", page_icon="🔄", layout="wide")

material_header("speed", "Live Traffic Simulator", "Simulate real-time customer transactions with live ML predictions")
check_auth()

api = get_api_client()

# Check if data is available
try:
    summary = api.get_data_summary()
    data_available = True
except Exception:
    data_available = False

if not data_available:
    st.warning("No data available. Please generate data and train models first from the Home page.")
    st.info("Navigate to the Data Explorer page to generate initial data.")
    st.stop()

# Initialize session state for the simulator
if 'simulator_running' not in st.session_state:
    st.session_state.simulator_running = False

if 'live_revenue_history' not in st.session_state:
    st.session_state.live_revenue_history = []
    st.session_state.live_revenue_history.append({
        "time": pd.Timestamp.now().strftime("%H:%M:%S"),
        "revenue": summary['revenue']['total']
    })

# Header Controls
col1, col2 = st.columns([1, 3])
with col1:
    if st.session_state.simulator_running:
        if st.button("Stop Simulator", use_container_width=True, type="primary"):
            st.session_state.simulator_running = False
            st.rerun()
    else:
        if st.button("Start Live Traffic", use_container_width=True, type="primary"):
            st.session_state.simulator_running = True
            st.rerun()

with col2:
    if st.session_state.simulator_running:
        st.info("Simulator active — generating transactions with real-time ML predictions...")
    else:
        st.warning("Simulator paused.")

st.divider()

# Layout for dashboards
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

chart_col, stats_col = st.columns([2, 1])

# Fetch latest data
try:
    current_summary = api.get_data_summary()
    live_predictions_data = api.get_live_predictions(limit=50)
    live_stats = api.get_live_stats()
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

total_revenue = current_summary["revenue"]["total"]
total_customers = current_summary["total_customers"]
total_transactions = current_summary["total_transactions"]

# Top Metrics
with metric_col1:
    st.metric("Total Revenue", f"${total_revenue:,.2f}", delta=f"${total_revenue - st.session_state.live_revenue_history[0]['revenue']:,.2f}" if len(st.session_state.live_revenue_history) > 1 else None)
with metric_col2:
    st.metric("Total Transactions", f"{total_transactions:,}")
with metric_col3:
    st.metric("Total Customers", f"{total_customers:,}")
with metric_col4:
    purchase_rate = (live_stats.get("will_purchase_count", 0) / live_stats.get("total_predictions", 1) * 100) if live_stats.get("total_predictions", 0) > 0 else 0
    st.metric("Live Purchase Rate", f"{purchase_rate:.1f}%", delta=f"{live_stats.get('total_predictions', 0)} predictions")

# Record History for charts (keep last 20 points)
if st.session_state.simulator_running:
    st.session_state.live_revenue_history.append({
        "time": pd.Timestamp.now().strftime("%H:%M:%S"),
        "revenue": total_revenue
    })
    if len(st.session_state.live_revenue_history) > 20:
        st.session_state.live_revenue_history.pop(0)

# Live Chart
with chart_col:
    st.subheader("Live Revenue Trend")
    if len(st.session_state.live_revenue_history) > 1:
        df_history = pd.DataFrame(st.session_state.live_revenue_history)
        fig = px.line(
            df_history, 
            x="time", 
            y="revenue", 
            markers=True,
            line_shape="spline",
            title="Real-Time Cumulative Revenue"
        )
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Total Revenue ($)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=30, b=0),
        )
        fig.update_traces(line_color="#10b981", line_width=3, marker=dict(size=8))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Waiting for more data to generate trend...")

# Live ML Prediction Stats
with stats_col:
    st.subheader("Live ML Stats")
    if live_stats.get("total_predictions", 0) > 0:
        st.metric("Will Purchase", live_stats.get("will_purchase_count", 0))
        st.metric("Will Not Purchase", live_stats.get("will_not_purchase_count", 0))
        st.metric("Avg Probability", f"{live_stats.get('avg_probability', 0):.2%}")

        seg_dist = live_stats.get("segment_distribution", {})
        if seg_dist:
            st.caption("Segment Distribution (Live)")
            df_seg = pd.DataFrame(list(seg_dist.items()), columns=["Segment", "Count"])
            fig_seg = px.pie(df_seg, names="Segment", values="Count", hole=0.4)
            fig_seg.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                height=200,
                showlegend=True,
                legend=dict(font=dict(size=10)),
            )
            st.plotly_chart(fig_seg, use_container_width=True)
    else:
        st.info("Start the simulator to see live ML stats.")

# Live Predictions Feed
st.divider()
st.subheader("Live ML Predictions Feed")

predictions = live_predictions_data.get("predictions", [])
if predictions:
    pred_rows = []
    for p in predictions:
        pred_rows.append({
            "Transaction": p.get("transaction_id", ""),
            "Customer": p.get("customer_id", ""),
            "Amount": f"${p.get('amount', 0):,.2f}",
            "Category": p.get("category", ""),
            "Segment": p.get("segment", {}).get("segment_name", "N/A"),
            "Will Purchase": "Yes" if p.get("prediction", {}).get("will_purchase", False) else "No",
            "Probability": f"{p.get('prediction', {}).get('probability', 0):.2%}",
            "Recommendation": p.get("recommendation", ""),
            "Time": p.get("timestamp", ""),
        })
    df_preds = pd.DataFrame(pred_rows)

    # Color-code the purchase prediction
    def highlight_purchase(row):
        if row["Will Purchase"] == "Yes":
            return ["background-color: #0d3320"] * len(row)
        return ["background-color: #3d1111"] * len(row)

    st.dataframe(
        df_preds.style.apply(highlight_purchase, axis=1),
        use_container_width=True,
        hide_index=True,
        height=400,
    )
else:
    st.info("No predictions yet. Start the simulator to see real-time ML predictions.")

# Simulation Loop
if st.session_state.simulator_running:
    # Generate 4-9 transactions per 3 seconds
    tx_count = random.randint(4, 9)
    try:
        api.simulate_traffic(count=tx_count)
    except Exception as e:
        st.error(f"Simulation API error: {e}")
        st.session_state.simulator_running = False
    
    time.sleep(3)
    st.rerun()
