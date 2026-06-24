"""
Insights Page - Business insights and recommendations
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
from charts import create_segment_bar_chart, create_affinity_heatmap
# pyrefly: ignore [missing-import]
from ui_components import material_header

st.set_page_config(page_title="Business Insights", page_icon=" ", layout="wide")

material_header("lightbulb", "Business Insights", "Actionable insights and recommendations based on customer analysis")
check_auth()

api = get_api_client()

# Check if data is available
summary = {}
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
except Exception:
    segments = []

st.divider()

# Executive Summary
st.subheader("Executive Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_revenue = summary.get("revenue", {}).get("total", 0)
    st.metric("Total Revenue", f"${total_revenue:,.0f}")

with col2:
    st.metric("Total Customers", f"{summary.get('total_customers', 0):,}")

with col3:
    st.metric("Avg Customer Value", f"${summary.get('revenue', {}).get('average', 0):,.2f}")

with col4:
    avg_frequency = summary.get("rfm_statistics", {}).get("frequency", {}).get("mean", 0)
    st.metric("Avg Purchase Frequency", f"{avg_frequency:.1f}")

st.divider()

# Key Findings
st.subheader("  Key Findings")

if segments:
    # Find best and worst segments
    best_segment = max(segments, key=lambda x: x.get("avg_monetary", 0))
    worst_segment = min(segments, key=lambda x: x.get("avg_monetary", 0))
    largest_segment = max(segments, key=lambda x: x.get("count", 0))
    most_active = min(segments, key=lambda x: x.get("avg_recency", 0))

    col1, col2 = st.columns(2)

    with col1:
        st.info(f"""
        **Highest Value Segment: {best_segment['name']}**
        - Average Monetary: ${best_segment['avg_monetary']:,.2f}
        - Customer Count: {best_segment['count']:,}
        - Revenue Contribution: ${best_segment['total_revenue']:,.0f}
        """)

        st.info(f"""
        **Most Active Segment: {most_active['name']}**
        - Average Recency: {most_active['avg_recency']:.0f} days
        - Customer Count: {most_active['count']:,}
        """)

    with col2:
        st.warning(f"""
        **Largest Segment: {largest_segment['name']}**
        - Customer Count: {largest_segment['count']:,}
        - % of Total: {largest_segment['percentage']:.1f}%
        """)

        st.warning(f"""
        **Lowest Value Segment: {worst_segment['name']}**
        - Average Monetary: ${worst_segment['avg_monetary']:,.2f}
        - Consider: Re-engagement or acquisition focus
        """)

st.divider()

# Revenue Analysis
st.subheader("  Revenue Analysis")

if segments:
    fig = create_segment_bar_chart(segments)
    st.plotly_chart(fig, use_container_width=True)

    total_rev = sum(s["total_revenue"] for s in segments)

    st.markdown("**Revenue Contribution by Segment:**")
    for segment in sorted(segments, key=lambda x: x["total_revenue"], reverse=True):
        rev = segment["total_revenue"]
        pct = (rev / total_rev * 100) if total_rev > 0 else 0
        st.progress(pct / 100, text=f"{segment['name']}: ${rev:,.0f} ({pct:.1f}%)")

st.divider()

# Market Basket Analysis [ELITE FEATURE]
st.subheader("  Market Basket Analysis (Product Affinity)")
st.markdown("Analyze which product categories are frequently purchased together mapping out cross-sell opportunities.")

try:
    affinity_data = api.get_product_affinity()
    if affinity_data:
        fig = create_affinity_heatmap(affinity_data)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Insufficient transaction variety to calculate affinity. Generate more diverse data.")
except Exception as e:
    st.error(f"Error calculating affinity: {str(e)}")

st.divider()

# Customer Recommendations
st.subheader("  Customer Strategy Recommendations")

segment_strategies = {
    "Champions": {
        "icon": " ",
        "strategy": "Retention & Advocacy",
        "actions": [
            "Implement VIP loyalty program with exclusive perks",
            "Offer early access to new products and sales",
            "Create referral rewards program",
            "Provide premium customer service",
            "Send personalized thank-you messages",
        ],
        "expected_impact": "Increase lifetime value by 20-30%",
    },
    "Loyal": {
        "icon": " ",
        "strategy": "Upsell & Cross-sell",
        "actions": [
            "Recommend complementary products",
            "Offer bundle deals and volume discounts",
            "Implement tiered loyalty program",
            "Send personalized product recommendations",
            "Create exclusive member events",
        ],
        "expected_impact": "Increase purchase frequency by 15-25%",
    },
    "At Risk": {
        "icon": " ",
        "strategy": "Re-engagement",
        "actions": [
            "Send win-back email campaigns",
            "Offer special comeback discounts",
            "Survey to understand why they're slipping",
            "Create urgency with limited-time offers",
            "Assign dedicated account manager",
        ],
        "expected_impact": "Recover 30-40% of at-risk customers",
    },
    "Hibernating": {
        "icon": " ",
        "strategy": "Reactivation",
        "actions": [
            "Send strong reactivation offers (20-30% off)",
            "Highlight what's new since their last purchase",
            "Consider if worth the reactivation cost",
            "Run win-back ad campaigns",
            "Offer free shipping incentive",
        ],
        "expected_impact": "Reactivate 10-20% of hibernating customers",
    },
    "New": {
        "icon": " ",
        "strategy": "Nurture & Onboard",
        "actions": [
            "Send welcome email series",
            "Provide product education content",
            "Offer first-purchase follow-up discount",
            "Create onboarding tutorials",
            "Request feedback early",
        ],
        "expected_impact": "Increase second purchase rate by 25-35%",
    },
}

for segment in segments:
    seg_name = segment["name"]
    if seg_name in segment_strategies:
        strategy = segment_strategies[seg_name]

        expander_label = f"{strategy['icon']} {seg_name} - {strategy['strategy']}"
        with st.expander(str(expander_label), expanded=False):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**Recommended Actions:**")
                for action in strategy["actions"]:
                    st.write(f"- {action}")

            with col2:
                st.metric("Customers", f"{segment['count']:,}")
                st.metric("% of Total", f"{segment['percentage']:.1f}%")
                st.metric("Avg Value", f"${segment['avg_monetary']:,.0f}")
                st.info(f"**Expected Impact:**\n{strategy['expected_impact']}")

# Strategy Generator [ELITE FEATURE]
st.divider()
st.subheader("  Interactive Marketing Strategy Optimizer")
st.markdown("Select a segment to generate a targeted marketing campaign in real-time.")

if segments:
    target_seg = st.selectbox("Select Target Segment", [s["name"] for s in segments])
    
    if st.button("🚀 Optimize Campaign Strategy"):
        with st.spinner("Analyzing segment behavior..."):
            # Rules-based dynamic strategy generation
            if target_seg == "Champions":
                st.success("### 💎 VIP Exclusive Strategy")
                st.markdown("""
                **Recommended Campaign:** 'The Elite Circle Invitation'
                
                **Campaign Logic:** 
                These customers have high Frequency and high Monetary. They don't need discounts; they need **Recognition**.
                
                **Generated Action Plan:**
                - Send personalized invitation to 'Premier Rewards' tier.
                - Assign a Dedicated Support concierge.
                - Gift: Limited edition product during their birthday month.
                
                **Potential ROI:** Very High (Retaining these customers protects 40% of total revenue).
                """)
            elif target_seg == "At Risk":
                st.error("### 🛡️ Retention & Shielding Strategy")
                st.markdown("""
                **Recommended Campaign:** 'The Personalized Comeback Offer'
                
                **Campaign Logic:** 
                Recency is increasing dangerously. We must intercept with a **Stop-Loss** offer.
                
                **Generated Action Plan:**
                - 'We miss you' email with a 30% discount on their last-viewed category.
                - Free Shipping for the next 48 hours.
                - Customer Satisfaction Survey call.
                """)
            else:
                st.info(f"### 📈 Growth Strategy for {target_seg}")
                st.markdown(f"""
                **Recommended Campaign:** '{target_seg} Growth Initiative'
                
                **Focus:** Moving this segment to 'Loyal' or 'Champions' through consistent value.
                
                **Action Plan:**
                - Tiered loyalty points for every dollar spent.
                - Educational content about product benefits.
                """)

st.divider()

# Marketing Campaign Ideas
st.subheader("  Marketing Campaign Ideas")

campaigns = [
    {
        "name": "VIP Appreciation Week",
        "target": "Champions",
        "description": "Exclusive week-long event with special discounts, early access, and free gifts for top customers.",
        "channels": ["Email", "In-app", "SMS"],
        "expected_roi": "3-5x",
    },
    {
        "name": "Win-Back Flash Sale",
        "target": "At Risk",
        "description": "48-hour flash sale with steep discounts targeted at customers who haven't purchased in 60+ days.",
        "channels": ["Email", "Retargeting Ads", "Push Notifications"],
        "expected_roi": "2-3x",
    },
    {
        "name": "New Customer Welcome Journey",
        "target": "New",
        "description": "Automated 5-email welcome series with product education, tips, and a second-purchase incentive.",
        "channels": ["Email", "In-app"],
        "expected_roi": "4-6x",
    },
    {
        "name": "Cross-Sell Recommendation Engine",
        "target": "Loyal",
        "description": "AI-powered product recommendations based on purchase history and similar customer behavior.",
        "channels": ["Website", "Email", "App"],
        "expected_roi": "5-8x",
    },
]

for campaign in campaigns:
    with st.expander(str(campaign["name"]), expanded=False):
        col1, col2 = st.columns([3, 1])

        with col1:
            st.write(f"**Target Segment:** {campaign['target']}")
            st.write(f"**Description:** {campaign['description']}")
            st.write(f"**Channels:** {', '.join(campaign['channels'])}")

        with col2:
            st.metric("Expected ROI", str(campaign["expected_roi"]))

st.divider()

# RFM Segment Guide
st.subheader("  RFM Segmentation Guide")

st.markdown("""
| Segment | Recency | Frequency | Monetary | Action |
|---------|---------|-----------|----------|--------|
| **Champions** | Recent | High | High | Reward, upsell premium products |
| **Loyal** | Recent | High | Medium | Upsell, loyalty program |
| **At Risk** | Old | Low | Medium | Win-back campaigns |
| **Hibernating** | Very Old | Low | Low | Strong re-engagement or sunset |
| **New** | Recent | Low | Low | Onboarding, nurture sequence |
""")

# Business Metrics
st.divider()
st.subheader("  Business Health Metrics")

if segments:
    col1, col2, col3 = st.columns(3)

    with col1:
        # Customer concentration
        largest_pct = max(s["percentage"] for s in segments)
        st.metric("Customer Concentration", f"{largest_pct:.1f}%")
        st.caption("Highest % in single segment")

    with col2:
        # Revenue concentration
        total_rev = sum(s["total_revenue"] for s in segments)
        rev_concentration = max(s["total_revenue"] for s in segments) / total_rev * 100 if total_rev > 0 else 0
        st.metric("Revenue Concentration", f"{rev_concentration:.1f}%")
        st.caption("Highest % revenue from one segment")

    with col3:
        # Segment balance
        segment_counts = [s["count"] for s in segments]
        balance = min(segment_counts) / max(segment_counts) * 100 if max(segment_counts) > 0 else 0
        st.metric("Segment Balance", f"{balance:.1f}%")
        st.caption("Smallest/Largest segment ratio")

# Final recommendations
st.divider()
st.subheader("  Action Plan Summary")

st.markdown("""
### Immediate Actions (This Week):
1. **Champions** - Send appreciation email and VIP program invite
2. **At Risk** - Launch win-back campaign with 15% discount
3. **New** - Ensure welcome series is active

### Short-term (This Month):
1. Implement product recommendation engine for Loyal segment
2. Create re-engagement ad campaign for Hibernating customers
3. Set up automated RFM recalculation pipeline

### Long-term (This Quarter):
1. Build predictive churn model using At Risk + Hibernating data
2. Develop personalized pricing strategy per segment
3. Create segment-specific landing pages and experiences
""")
