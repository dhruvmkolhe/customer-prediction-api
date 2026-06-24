"""
Segmentation Page - Customer segment analysis and visualization
"""

import streamlit as st
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils"))

# pyrefly: ignore [missing-import]
from api_client import get_api_client
# pyrefly: ignore [missing-import]
from auth import check_auth
# pyrefly: ignore [missing-import]
from ui_components import material_header
# pyrefly: ignore [missing-import]
from charts import (
    create_rfm_scatter_3d,
    create_segment_distribution,
    create_rfm_heatmap,
    create_segment_bar_chart,
)

st.set_page_config(page_title="Customer Segmentation", page_icon=" ", layout="wide")

material_header("group", "Customer Segmentation", "Analyze customer segments based on RFM (Recency, Frequency, Monetary) analysis")
check_auth()

api = get_api_client()

# Check if data is available
try:
    summary = api.get_data_summary()
    data_available = True
except Exception:
    data_available = False

if not data_available:
    st.warning("No data available. Please generate data first.")
    st.stop()

# Get segments
try:
    segments_response = api.get_segments()
    segments = segments_response["segments"]
    total_customers = segments_response["total"]
except Exception as e:
    st.error(f"Error loading segments: {e}")
    st.info("Please train models first to generate segments.")
    st.stop()

# Segment Overview
st.subheader("Segment Overview")

# Summary metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Customers", f"{total_customers:,}")
with col2:
    st.metric("Number of Segments", len(segments))
with col3:
    avg_recency = sum(s["avg_recency"] for s in segments) / len(segments)
    st.metric("Avg Recency", f"{avg_recency:.0f} days")
with col4:
    total_revenue = sum(s["total_revenue"] for s in segments)
    st.metric("Total Revenue", f"${total_revenue:,.0f}")

st.divider()

# Segment Distribution
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Segment Distribution")
    fig = create_segment_distribution(segments)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Revenue by Segment")
    fig = create_segment_bar_chart(segments)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# Segment Details Table
st.subheader("Segment Characteristics")

segment_df = pd.DataFrame(segments)
segment_df = segment_df.rename(columns={
    "name": "Segment",
    "count": "Customers",
    "percentage": "% of Total",
    "avg_recency": "Avg Recency (days)",
    "avg_frequency": "Avg Frequency",
    "avg_monetary": "Avg Monetary ($)",
    "total_revenue": "Total Revenue ($)",
})

st.dataframe(
    segment_df.style.format({
        "% of Total": "{:.1f}%",
        "Avg Recency (days)": "{:.0f}",
        "Avg Frequency": "{:.1f}",
        "Avg Monetary ($)": "${:,.2f}",
        "Total Revenue ($)": "${:,.2f}",
    }),
    use_container_width=True,
    hide_index=True,
)

st.divider()

# RFM Heatmap
st.subheader("RFM Metrics Heatmap")
fig = create_rfm_heatmap(segments)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# 3D Scatter Plot
st.subheader("3D Customer Visualization")
st.markdown("Interactive 3D scatter plot showing customer distribution across RFM dimensions")

try:
    rfm_response = api.get_rfm_data(limit=5000)
    rfm_df = pd.DataFrame(rfm_response["data"])

    if not rfm_df.empty and "segment_name" in rfm_df.columns:
        fig = create_rfm_scatter_3d(rfm_df)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Segment labels not available. Please train models first.")
except Exception as e:
    st.error(f"Error loading RFM data: {e}")

st.divider()

# Segment Deep Dive
st.subheader("Segment Deep Dive")

selected_segment = st.selectbox(
    "Select a segment to explore",
    options=[s["name"] for s in segments],
    index=0,
)

segment_details = next((s for s in segments if s["name"] == selected_segment), None)

if segment_details:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Customers", f"{segment_details['count']:,}")
        st.metric("% of Total", f"{segment_details['percentage']:.1f}%")

    with col2:
        st.metric("Avg Recency", f"{segment_details['avg_recency']:.0f} days")
        st.metric("Avg Frequency", f"{segment_details['avg_frequency']:.1f}")

    with col3:
        st.metric("Avg Monetary", f"${segment_details['avg_monetary']:,.2f}")
        st.metric("Total Revenue", f"${segment_details['total_revenue']:,.0f}")

    # Segment interpretation
    st.divider()
    st.subheader("Segment Interpretation")

    interpretations = {
        "Champions": """
        **Champions** are your best customers. They bought recently, buy often, and spend the most.
        - **Strategy:** Reward them. They can be early adopters for new products and will promote your brand.
        """,
        "Loyal": """
        **Loyal** customers are consistent buyers who respond well to promotions.
        - **Strategy:** Upsell higher-value products. Ask for reviews and engage them in loyalty programs.
        """,
        "At Risk": """
        **At Risk** customers haven't purchased for a while and their frequency is dropping.
        - **Strategy:** Send personalized emails or SMS with big discounts to win them back before they churn.
        """,
        "Hibernating": """
        **Hibernating** customers' last purchase was a long time ago. They are low spenders and low frequency.
        - **Strategy:** Offer standard reactiviation incentives. Don't spend too much on direct marketing here.
        """,
        "New": """
        **New** customers just made their first purchase.
        - **Strategy:** Provide a great onboarding experience. Send a "thank you" note and a coupon for their second purchase.
        """,
        "Potential": """
        **Potential** customers show signs of becoming loyal but need more engagement.
        - **Strategy:** Recommend related products and offer time-limited discounts to encourage a second purchase.
        """,
        "Other": """
        **Other** customers don't clearly fit into main categories but represent diverse behavioral patterns.
        - **Strategy:** Monitor their transition into more defined segments over time.
        """,
    }

    # Get interpretation with a friendly fallback
    default_interpretation = f"""
    ### {selected_segment} Interpretation
    This group represents a specific behavioral cluster identified by the AI.
    - **Characteristics:** {selected_segment} shows unique patterns in purchase frequency and spending.
    - **Strategy:** Monitor this segment's growth and test different engagement offers to see what resonates best.
    """
    interpretation = interpretations.get(selected_segment, default_interpretation)
    st.markdown(interpretation)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🧠 Generate Strategy with RanNeeti AI", key=f"strategy_{selected_segment}", type="primary"):
        with st.spinner("RanNeeti AI is creating a customized strategy..."):
            try:
                # ensure all fields are available
                payload = {
                    "name": segment_details["name"],
                    "count": segment_details["count"],
                    "percentage": segment_details["percentage"],
                    "avg_recency": segment_details["avg_recency"],
                    "avg_frequency": segment_details["avg_frequency"],
                    "avg_monetary": segment_details["avg_monetary"],
                    "total_revenue": segment_details["total_revenue"]
                }
                result = api.generate_strategy(payload)
                st.success("Strategy Generated Successfully!")
                st.markdown("### 📋 Marketing Strategy")
                st.info(result.get("strategy", "No strategy returned."))
            except Exception as e:
                st.error(f"Error calling AI API: {e}")

# Customer Lookup
st.divider()
st.subheader("  Customer Lookup")

customer_id = st.number_input("Enter Customer ID", min_value=10001, max_value=20000, value=10001)

if st.button("Look Up Customer", use_container_width=False):
    try:
        customer_data = rfm_df[rfm_df["customer_id"] == customer_id]
        if not customer_data.empty:
            customer = customer_data.iloc[0]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Recency", f"{customer.get('recency', 'N/A')} days")
            with col2:
                st.metric("Frequency", f"{customer.get('frequency', 'N/A')}")
            with col3:
                st.metric("Monetary", f"${customer.get('monetary', 0):,.2f}")
            with col4:
                st.metric("Segment", customer.get('segment_name', 'N/A'))
        else:
            st.warning(f"Customer {customer_id} not found")
    except Exception as e:
        st.error(f"Error looking up customer: {e}")
