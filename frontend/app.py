"""
Real-Time Customer Purchase Behavior Prediction
Main Streamlit Application
"""

import streamlit as st
import sys
import os

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))

# pyrefly: ignore [missing-import]
from api_client import get_api_client
# pyrefly: ignore [missing-import]
from auth import check_auth, logout
from ui_components import material_header, inject_material_icons

# Page configuration
st.set_page_config(
    page_title="Customer Purchase Prediction",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4ECDC4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #4ECDC4;
    }
    .stMetric {
        background-color: #1a1a2e;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application."""
    # Check authentication first
    check_auth()
    
    inject_material_icons()
    material_header("hub", "Customer Intelligence Suite", "Advanced real-time purchase behavior prediction & segmentation")

    # Initialize API client
    api = get_api_client()

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/robot.png", width=80)
        st.title("Control Panel")

        # Connection status
        try:
            health = api.health_check()
            st.success("  API Connected")
            st.info(f"Data Loaded: {health.get('data_loaded', False)}")
            st.info(f"Models Loaded: {health.get('models_loaded', False)}")
        except Exception:
            st.error("  API Not Connected")
            st.warning("Please start the backend server")

        st.divider()

        # Quick actions
        st.subheader("Quick Actions")
        if st.button("  Generate New Data", use_container_width=True):
            with st.spinner("Generating data..."):
                try:
                    result = api.generate_data()
                    st.success(f"Generated {result['n_customers']} customers")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        if st.button("  Train Models", use_container_width=True):
            with st.spinner("Training models..."):
                try:
                    result = api.train_models()
                    st.success(f"Best model: {result['results']['classification']['best_model']}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        st.divider()

        st.markdown("""
        Use the **Pages** sidebar to navigate:
        -   **Data Explorer** - View and analyze data
        -   **Segmentation** - Customer segments
        -   **Model Comparison** - ML model metrics
        -   **Predict** - Make predictions
        -   **Insights** - Business insights
        -   **Live Traffic** - Real-time traffic simulation
        """)

        st.divider()
        if st.sidebar.button("  Log Out", key="logout_btn"):
            logout()

    # Main content
    st.divider()

    # System overview
    st.subheader("System Overview")

    col1, col2, col3, col4 = st.columns(4)

    try:
        summary = api.get_data_summary()

        with col1:
            st.metric(
                "Total Customers",
                f"{summary.get('total_customers', 0):,}"
            )
        with col2:
            st.metric(
                "Total Transactions",
                f"{summary.get('total_transactions', 0):,}"
            )
        with col3:
            st.metric(
                "Total Revenue",
                f"${summary.get('revenue', {}).get('total', 0):,.2f}"
            )
        with col4:
            st.metric(
                "Avg Customer Value",
                f"${summary.get('revenue', {}).get('average', 0):,.2f}"
            )

    except Exception:
        col1.metric("Total Customers", "N/A")
        col2.metric("Total Transactions", "N/A")
        col3.metric("Total Revenue", "N/A")
        col4.metric("Avg Customer Value", "N/A")

    st.divider()

    # Project description
    st.subheader("About This Project")

    st.markdown("""
    This system uses **Machine Learning** to predict customer purchase behavior and segment customers
    based on their purchasing patterns. The key features include:

    | Feature | Description |
    |---------|-------------|
    |   **Customer Segmentation** | K-Means clustering based on RFM (Recency, Frequency, Monetary) analysis |
    |   **Purchase Prediction** | Multiple ML classifiers (Random Forest, XGBoost, Logistic Regression) |
    |   **Real-Time Predictions** | Instant predictions via FastAPI backend |
    |   **Interactive Dashboard** | Streamlit-based visualization and analysis |

    ### How to Use:
    1. **Generate Data** - Click the button in the sidebar to create sample data
    2. **Train Models** - Train ML models on the generated data
    3. **Explore** - Navigate through pages to analyze segments and model performance
    4. **Predict** - Use the Predict page to make real-time predictions
    """)

    # Tech stack
    st.subheader("Tech Stack")
    tech_cols = st.columns(4)
    with tech_cols[0]:
        st.info("**Backend**\n- FastAPI\n- Scikit-learn\n- XGBoost")
    with tech_cols[1]:
        st.info("**Frontend**\n- Streamlit\n- Plotly\n- Pandas")
    with tech_cols[2]:
        st.info("**ML Models**\n- K-Means\n- Random Forest\n- XGBoost")
    with tech_cols[3]:
        st.info("**Features**\n- RFM Analysis\n- Real-time API\n- Interactive Viz")


if __name__ == "__main__":
    main()
