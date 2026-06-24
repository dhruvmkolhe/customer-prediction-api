"""
Data Explorer Page - View and analyze customer data
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
    create_rfm_distributions,
    create_category_distribution,
    create_monthly_revenue_chart,
)

st.set_page_config(page_title="Data Explorer", page_icon=" ", layout="wide")

material_header("analytics", "Data Explorer", "Explore customer transactions and dataset quality")
check_auth()

api = get_api_client()

# Check if data is available
try:
    summary = api.get_data_summary()
    data_available = True
except Exception:
    data_available = False

if not data_available:
    st.warning("No data available. Please generate data first from the Home page.")
    st.info("Click **Generate New Data** in the sidebar or navigate to the Home page.")
    st.stop()

# Data Generation Controls
with st.expander("  Generate New Data", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        n_customers = st.number_input("Number of Customers", 100, 100000, 10000, step=1000)
    with col2:
        n_transactions = st.number_input("Number of Transactions", 500, 500000, 50000, step=5000)
    with col3:
        st.write("")
        st.write("")
        if st.button("  Generate Data", use_container_width=True):
            with st.spinner("Generating synthetic data..."):
                try:
                    result = api.generate_data(n_customers, n_transactions)
                    st.success(f"Generated {result['n_customers']:,} customers and {result['n_transactions']:,} transactions!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to generate data: {str(e)}")

st.divider()

# Dataset Overview
st.subheader("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Customers", f"{summary['total_customers']:,}")
with col2:
    st.metric("Total Transactions", f"{summary['total_transactions']:,}")
with col3:
    st.metric("Total Revenue", f"${summary['revenue']['total']:,.2f}")
with col4:
    st.metric("Avg Order Value", f"${summary['revenue']['average']:,.2f}")

st.divider()

# RFM Statistics
st.subheader("RFM Statistics")

rfm_stats = summary.get("rfm_statistics", {})
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Recency** (days since last purchase)")
    if "recency" in rfm_stats:
        st.write(f"- Mean: {rfm_stats['recency']['mean']:.1f} days")
        st.write(f"- Median: {rfm_stats['recency']['median']:.1f} days")
        st.write(f"- Range: {rfm_stats['recency']['min']} - {rfm_stats['recency']['max']} days")

with col2:
    st.markdown("**Frequency** (number of transactions)")
    if "frequency" in rfm_stats:
        st.write(f"- Mean: {rfm_stats['frequency']['mean']:.1f}")
        st.write(f"- Median: {rfm_stats['frequency']['median']:.1f}")
        st.write(f"- Range: {rfm_stats['frequency']['min']} - {rfm_stats['frequency']['max']}")

with col3:
    st.markdown("**Monetary** (total spend)")
    if "monetary" in rfm_stats:
        st.write(f"- Mean: ${rfm_stats['monetary']['mean']:,.2f}")
        st.write(f"- Median: ${rfm_stats['monetary']['median']:,.2f}")
        st.write(f"- Range: ${rfm_stats['monetary']['min']:,.2f} - ${rfm_stats['monetary']['max']:,.2f}")

st.divider()

# Data Tables
tab1, tab2, tab3 = st.tabs(["  RFM Data", "  Transactions", "  Charts"])

with tab1:
    st.subheader("RFM Feature Data")

    page_size = st.slider("Rows per page", 10, 500, 100, key="rfm_page_size")
    page = st.number_input("Page", 1, 100, 1, key="rfm_page")

    rfm_response = api.get_rfm_data(limit=page_size, offset=(page - 1) * page_size)
    rfm_df = pd.DataFrame(rfm_response["data"])

    if not rfm_df.empty:
        st.dataframe(rfm_df, use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(rfm_df)} of {rfm_response['total']:,} records")
    else:
        st.info("No RFM data available")

with tab2:
    st.subheader("Transaction Data")

    page_size = st.slider("Rows per page", 10, 500, 100, key="trans_page_size")
    page = st.number_input("Page", 1, 100, 1, key="trans_page")

    trans_response = api.get_transactions(limit=page_size, offset=(page - 1) * page_size)
    trans_df = pd.DataFrame(trans_response["data"])

    if not trans_df.empty:
        st.dataframe(trans_df, use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(trans_df)} of {trans_response['total']:,} records")
    else:
        st.info("No transaction data available")

with tab3:
    st.subheader("Data Visualizations")

    # RFM Distributions
    st.markdown("#### RFM Distributions")
    try:
        rfm_all = api.get_rfm_data(limit=10000)
        rfm_df = pd.DataFrame(rfm_all["data"])
        if not rfm_df.empty:
            fig = create_rfm_distributions(rfm_df)
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading RFM data: {e}")

    # Category Distribution
    st.markdown("#### Product Category Distribution")
    categories = summary.get("category_distribution", {})
    if categories:
        fig = create_category_distribution(categories)
        st.plotly_chart(fig, use_container_width=True)

    # Monthly Revenue
    st.markdown("#### Monthly Revenue Trend")
    try:
        monthly_data = api.get_monthly_revenue()
        if monthly_data.get("data"):
            fig = create_monthly_revenue_chart(monthly_data["data"])
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading monthly revenue: {e}")

# Summary insights
st.divider()
st.subheader("  Data Insights")

col1, col2 = st.columns(2)
with col1:
    st.info(f"""
    **Payment Methods:**
    {chr(10).join(f'- {k}: {v:,}' for k, v in summary.get('payment_method_distribution', {}).items())}
    """)

with col2:
    st.info(f"""
    **Top Categories:**
    {chr(10).join(f'- {k}: {v:,}' for k, v in list(summary.get('category_distribution', {}).items())[:5])}
    """)
