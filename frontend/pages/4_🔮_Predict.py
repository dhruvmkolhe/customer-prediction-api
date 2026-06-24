"""
Real-Time Customer Purchase Behavior Prediction
Predict Page - Individual & Batch Predictions with XAI and What-If Simulator
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from io import BytesIO

from openpyxl import Workbook

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils"))

# pyrefly: ignore [missing-import]
from api_client import get_api_client
# pyrefly: ignore [missing-import]
from auth import check_auth
# pyrefly: ignore [missing-import]
from ui_components import material_header
# pyrefly: ignore [missing-import]
from charts import create_prediction_gauge, create_feature_importance_chart

st.set_page_config(page_title="Predict", page_icon="🔮", layout="wide")

material_header("online_prediction", "Purchase Prediction", "Real-time behavioral prediction and AI-driven recommendations")
check_auth()

api = get_api_client()


def _normalize_batch_row(row: dict, row_index: int) -> dict:
    """Map common spreadsheet column names to the API contract."""
    lowered = {str(key).strip().lower(): value for key, value in row.items()}

    recency = lowered.get("recency", lowered.get("recency (days since last purchase)", 0))
    frequency = lowered.get("frequency", lowered.get("total_purchases", lowered.get("purchases", 0)))
    monetary = lowered.get("monetary", lowered.get("total_spend", lowered.get("spend", 0.0)))
    account_age = lowered.get("customer_lifetime_days", lowered.get("account_age", lowered.get("age", 0)))

    try:
        frequency_value = int(float(frequency))
    except (TypeError, ValueError):
        frequency_value = 0

    try:
        monetary_value = float(monetary)
    except (TypeError, ValueError):
        monetary_value = 0.0

    try:
        account_age_value = int(float(account_age))
    except (TypeError, ValueError):
        account_age_value = 0

    return {
        "customer_id": int(lowered.get("customer_id", row_index + 1)),
        "recency": float(recency or 0),
        "frequency": frequency_value,
        "monetary": monetary_value,
        "avg_order_value": float(lowered.get("avg_order_value", monetary_value / max(1, frequency_value))),
        "purchase_frequency": float(lowered.get(
            "purchase_frequency",
            frequency_value / max(1, account_age_value / 30)
        )),
        "product_diversity": float(lowered.get("product_diversity", 0.5)),
        "customer_lifetime_days": account_age_value,
    }


def _build_xlsx_template() -> bytes:
    """Create an Excel template that matches the API contract."""
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Predictions"
    headers = [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "avg_order_value",
        "purchase_frequency",
        "product_diversity",
        "customer_lifetime_days",
    ]
    examples = [10001, 30, 5, 500, 100, 1.5, 0.5, 365]
    sheet.append(headers)
    sheet.append(examples)

    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()

def main():
    # Tabs for different prediction modes
    tab1, tab2 = st.tabs(["🎯 Individual Prediction & What-If", "📥 Batch Prediction"])

    with tab1:
        st.subheader("Interactive What-If Simulator")
        st.info("Adjust the behavioral parameters to see how the model's prediction changes dynamically.")

        col1, col2 = st.columns([1, 1])

        with col1:
            with st.form("prediction_form"):
                st.markdown("### Customer Metrics")
                recency = st.slider("Recency (Days since last purchase)", 0, 365, 30)
                frequency = st.slider("Frequency (Total number of purchases)", 1, 100, 5)
                monetary = st.number_input("Monetary (Total spend $)", 0.0, 100000.0, 500.0)
                
                st.divider()
                st.markdown("### Demographics")
                age = st.slider("Customer Age", 18, 100, 35)
                account_age = st.slider("Account Age (Days)", 1, 3650, 365)
                
                st.divider()
                model_name = st.selectbox(
                    "Select ML Model",
                    ["randomforest", "xgboost", "logisticregression", "gradientboosting"],
                    index=0
                )
                
                predict_btn = st.form_submit_button("🚀 Run Prediction", use_container_width=True)

        if predict_btn:
            with st.spinner("Analyzing data..."):
                try:
                    # Map UI sliders to model features and calculate derived metrics
                    payload = {
                        "customer_id": 9999, # Dummy ID for simulator
                        "recency": float(recency),
                        "frequency": int(frequency),
                        "monetary": float(monetary),
                        "avg_order_value": float(monetary / max(1, frequency)),
                        "purchase_frequency": float(frequency / max(1, account_age / 30)),
                        "product_diversity": 0.5, # Default for manual entry
                        "customer_lifetime_days": int(account_age),
                        "model_name": model_name
                    }
                    result = api.predict_single(payload)
                    importance = api.get_model_importance(model_name)

                    with col2:
                        pred_data = result.get("prediction", {})
                        prob = pred_data.get("probability", 0.0)
                        pred = pred_data.get("will_purchase", False)
                        
                        st.markdown(f"""
                        <div style='text-align: center; padding: 1.5rem; background: #1a1a2e; border-radius: 15px; border: 1px solid #4ECDC4; margin-bottom: 2rem;'>
                            <h3 style='margin: 0; color: #888;'>Likelihood of Purchase</h3>
                            <h1 style='font-size: 3.5rem; margin: 0.5rem 0; color: {"#4ECDC4" if pred else "#FF6B6B"};'>{"WILL BUY" if pred else "NO PURCHASE"}</h1>
                            <p style='font-size: 1.4rem; font-weight: bold;'>{prob:.1%} Accuracy Probability</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        fig = create_prediction_gauge(prob)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # XAI Section
                        st.subheader("🔍 The 'Why' Factor (XAI)")
                        st.markdown(f"**{model_name.upper()}** decision drivers for this result:")
                        imp_fig = create_feature_importance_chart(importance)
                        st.plotly_chart(imp_fig, use_container_width=True)
                        
                        if pred:
                            st.success("✅ **Positive Signal**: High engagement and recent activity suggest a high conversion probability.")
                        else:
                            st.warning("⚠️ **Churn Risk**: Low frequency or high recency indicates a possible loss of engagement.")

                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")

    with tab2:
        st.subheader("Batch Assessments")
        st.markdown("Process large customer datasets simultaneously. Results include predicted purchase probability and binary classification.")
        
        # Download templates section
        st.write("### 📎 Download Templates")
        t_col1, t_col2, t_col3 = st.columns(3)
        with t_col1:
            st.download_button(
                "CSV Template",
                "customer_id,recency,frequency,monetary,avg_order_value,purchase_frequency,product_diversity,customer_lifetime_days\n10001,30,5,500,100,1.5,0.5,365",
                "template.csv",
                "text/csv",
                use_container_width=True,
            )
        with t_col2:
            st.download_button(
                "XLSX Template",
                _build_xlsx_template(),
                "template.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with t_col3:
            st.download_button(
                "JSON Template",
                '{"customer_id":10001,"recency":30,"frequency":5,"monetary":500,"avg_order_value":100,"purchase_frequency":1.5,"product_diversity":0.5,"customer_lifetime_days":365}',
                "template.json",
                "application/json",
                use_container_width=True,
            )

        st.divider()

        uploaded_file = st.file_uploader("Upload file for batch processing", type=["csv", "xlsx", "json"], key="batch_upload")

        if uploaded_file is not None:
            try:
                file_ext = uploaded_file.name.split('.')[-1].lower()
                if file_ext == 'csv':
                    batch_df = pd.read_csv(uploaded_file)
                elif file_ext == 'xlsx':
                    batch_df = pd.read_excel(uploaded_file)
                elif file_ext == 'json':
                    batch_df = pd.read_json(uploaded_file)
                
                st.write(f"### 📄 Dataset Preview ({file_ext.upper()})")
                st.dataframe(batch_df.head(), use_container_width=True, hide_index=True)

                if st.button("📊 Run Batch Prediction", use_container_width=True):
                    with st.spinner("Processing large-scale inference..."):
                        normalized_customers = [
                            _normalize_batch_row(row, idx)
                            for idx, row in enumerate(batch_df.to_dict(orient="records"))
                        ]
                        results = api.predict_batch(normalized_customers)
                        results_df = pd.DataFrame(results.get("predictions", []))
                        
                        st.subheader("📊 Batch Results")
                        st.success(f"Successfully processed {len(results_df)} records")
                        
                        # Add some visual distinction
                        results_df['Decision'] = results_df['prediction'].apply(
                            lambda x: "BUY" if isinstance(x, dict) and x.get("will_purchase") else "PASS"
                        )
                        results_df['Probability'] = results_df['prediction'].apply(
                            lambda x: x.get("probability", 0.0) if isinstance(x, dict) else 0.0
                        )
                        st.dataframe(results_df, use_container_width=True, hide_index=True)
                        
                        # Download results
                        csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Download Full Prediction Results",
                            csv,
                            "prediction_results.csv",
                            "text/csv",
                            use_container_width=True
                        )
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        Major Project: Real-Time Customer Behavior Intelligence Suite © 2026
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
