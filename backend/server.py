"""
FastAPI Backend for Customer Purchase Behavior Prediction
Provides endpoints for data generation, training, and real-time predictions.
"""

from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
from typing import Optional, List
import pandas as pd
import numpy as np
import os
import json
import sqlite3
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure backend directory is in path for relative imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("server.log")
    ]
)
logger = logging.getLogger("CustomerAPI")

from data_generator import DataGenerator
from model_trainer import ModelTrainer
from predictor import Predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for data and models."""
    global transactions_df, rfm_df, predictor

    # Load data if exists
    if os.path.exists(DB_PATH):
        logger.info(f"Loading data from SQLite at {DB_PATH}...")
        conn = sqlite3.connect(DB_PATH)
        try:
            transactions_df = pd.read_sql("SELECT * FROM transactions", conn)
            rfm_df = pd.read_sql("SELECT * FROM rfm_features", conn)
            logger.info(f"Successfully loaded {len(rfm_df)} RFM records from SQL")
        except Exception as e:
            logger.error(f"Error loading from DB: {e}. Falling back to CSV.")
            # Fallback to CSV
            transactions_path = os.path.join(DATA_DIR, "transactions.csv")
            rfm_path = os.path.join(DATA_DIR, "rfm_features.csv")
            if os.path.exists(rfm_path):
                transactions_df = pd.read_csv(transactions_path)
                rfm_df = pd.read_csv(rfm_path)
        finally:
            conn.close()
    else:
        logger.warning(f"Database not found at {DB_PATH}")

    # Load models if exist
    predictor = Predictor(MODELS_DIR)
    if predictor.load_models():
        logger.info("Models successfully loaded during startup")
    else:
        logger.warning("Models could not be loaded during startup")

    yield

    # Cleanup
    print("Shutting down API...")


# Initialize FastAPI app
app = FastAPI(
    title="Customer Purchase Prediction API",
    description="Real-time customer purchase behavior prediction and segmentation",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
BACKEND_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BACKEND_DIR, "data")
MODELS_DIR = BACKEND_DIR
DB_PATH = os.path.join(DATA_DIR, "customer_data.db")

# Add explicit type annotations for global variables
transactions_df: Optional[pd.DataFrame] = None
rfm_df: Optional[pd.DataFrame] = None
predictor: Optional[Predictor] = None

# Real-time prediction log (stores recent predictions for streaming)
live_predictions: list[dict] = []
MAX_LIVE_PREDICTIONS = 200


# Pydantic models
RFM_FEATURES = [
    "recency", "frequency", "monetary", "avg_order_value",
    "purchase_frequency", "product_diversity", "customer_lifetime_days"
]

class PredictionRequest(BaseModel):
    customer_id: int = Field(..., description="Customer ID")
    recency: float = Field(..., ge=0, description="Days since last purchase")
    frequency: int = Field(..., ge=0, description="Number of transactions")
    monetary: float = Field(..., ge=0, description="Total spend amount")
    avg_order_value: float = Field(..., ge=0, description="Average order value")
    purchase_frequency: float = Field(..., ge=0, description="Purchases per month")
    product_diversity: float = Field(..., ge=0, le=1, description="Product diversity score")
    customer_lifetime_days: int = Field(..., ge=0, description="Days as customer")
    model_name: Optional[str] = Field(None, description="Specific model to use for prediction")


class BatchPredictionRequest(BaseModel):
    customers: list[PredictionRequest]
    model_name: Optional[str] = None


class TrainingConfig(BaseModel):
    n_customers: int = Field(5000, ge=100, le=50000)
    n_transactions: int = Field(20000, ge=500, le=200000)

class SegmentDetailsRequest(BaseModel):
    name: str
    count: int
    percentage: float
    avg_recency: float
    avg_frequency: float
    avg_monetary: float
    total_revenue: float





@app.get("/")
def home():
    """API health check."""
    return {
        "message": "Customer Purchase Prediction API",
        "version": "1.0.0",
        "status": "running",
        "data_loaded": rfm_df is not None,
        "models_loaded": predictor is not None and predictor.models_loaded,
    }


@app.post("/generate-data")
def generate_data(config: TrainingConfig = TrainingConfig()):
    """Generate mock transaction data."""
    global transactions_df, rfm_df

    generator = DataGenerator(
        n_customers=config.n_customers,
        n_transactions=config.n_transactions
    )
    transactions_df, rfm_df = generator.save(DATA_DIR, use_db=True)

    return {
        "message": "Data generated successfully",
        "n_customers": config.n_customers,
        "n_transactions": config.n_transactions,
        "rfm_records": len(rfm_df) if rfm_df is not None else 0,
    }


@app.post("/simulate-traffic")
def simulate_traffic(count: int = Query(50, ge=1, le=500)):
    """Simulate live incoming traffic by generating new random transactions with real-time ML predictions."""
    try:
        global transactions_df, rfm_df, live_predictions
        
        if transactions_df is None or rfm_df is None:
            raise HTTPException(status_code=400, detail="Initialize data first using /generate-data.")

        # Randomly select existing customers
        customers = rfm_df["customer_id"].sample(n=count, replace=True).values
        
        # Generate random mock transactions
        import random
        from datetime import datetime
        
        categories = list(transactions_df["product_category"].unique()) if not transactions_df.empty else ["Electronics", "Clothing", "Home", "Beauty", "Sports"]
        payment_methods = list(transactions_df["payment_method"].unique()) if not transactions_df.empty else ["Credit Card", "PayPal", "Debit Card", "Store Credit"]
        
        new_tx = []
        new_predictions = []
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Check max transaction id
        max_id = 0
        if not transactions_df.empty and "transaction_id" in transactions_df.columns:
            try:
                coords = transactions_df["transaction_id"].astype(str).str.extract(r'(\d+)').astype(float)
                max_id = int(coords.max().fillna(0).iloc[0])
            except Exception:
                max_id = len(transactions_df)

        for i in range(count):
            cust_id = int(customers[i])
            category = random.choice(categories)
            payment = random.choice(payment_methods)
            
            # Determine amount based on category loosely
            base_amt = {"Electronics": 200, "Clothing": 50, "Home": 100, "Beauty": 30, "Sports": 80}.get(category, 50)
            amount = round(random.uniform(base_amt * 0.5, base_amt * 2.5), 2)
            
            new_tx.append({
                "transaction_id": f"TXN-{max_id + i + 1}",
                "customer_id": cust_id,
                "transaction_date": current_time,
                "amount": amount,
                "total_amount": amount,
                "product_category": category,
                "payment_method": payment,
                "status": "completed"
            })
            
            # Find customer in RFM to update their stats
            idx_list = rfm_df.index[rfm_df["customer_id"].astype(int) == cust_id].tolist()
            idx = idx_list[0] if idx_list else None
            
            if idx is not None:
                # pyrefly: ignore [bad-argument-type]
                rfm_df.at[idx, "frequency"] = int(rfm_df.at[idx, "frequency"]) + 1
                # pyrefly: ignore [bad-argument-type]
                rfm_df.at[idx, "monetary"] = float(rfm_df.at[idx, "monetary"]) + amount
                rfm_df.at[idx, "recency"] = 0 # they just bought

            # Run real-time ML prediction for this transaction
            if predictor is not None and predictor.models_loaded and idx:
                try:
                    # Add slight noise to features to ensure prediction variety for demonstration
                    recency_val = float(random.choice([0, 0, 5, 10, 20])) # Occasionally simulate non-immediate recency
                    
                    customer_features = {
                        "recency": recency_val,
                        "frequency": int(rfm_df.at[idx, "frequency"]),
                        "monetary": float(rfm_df.at[idx, "monetary"]),
                        "avg_order_value": float(rfm_df.at[idx, "avg_order_value"]) if "avg_order_value" in rfm_df.columns else float(rfm_df.at[idx, "monetary"]) / max(int(rfm_df.at[idx, "frequency"]), 1),
                        "purchase_frequency": float(rfm_df.at[idx, "purchase_frequency"]) if "purchase_frequency" in rfm_df.columns else 1.0,
                        "product_diversity": float(rfm_df.at[idx, "product_diversity"]) if "product_diversity" in rfm_df.columns else 0.5,
                        "customer_lifetime_days": int(rfm_df.at[idx, "customer_lifetime_days"]) if "customer_lifetime_days" in rfm_df.columns else 30,
                    }
                    ml_result = predictor.predict_purchase(customer_features)

                    prediction_record = {
                        "transaction_id": f"TXN-{max_id + i + 1}",
                        "customer_id": cust_id,
                        "amount": amount,
                        "category": category,
                        "payment_method": payment,
                        "timestamp": current_time,
                        "segment": ml_result.get("segment", {}),
                        "prediction": ml_result.get("prediction", {}),
                        "recommendation": ml_result.get("recommendation", ""),
                    }
                    new_predictions.append(prediction_record)
                except Exception as pred_err:
                    logger.warning(f"Prediction failed for customer {cust_id}: {pred_err}")

        # Append to dataframe
        new_df = pd.DataFrame(new_tx)
        transactions_df = pd.concat([transactions_df, new_df], ignore_index=True)

        # Store predictions in the live_predictions buffer
        live_predictions.extend(new_predictions)
        if len(live_predictions) > MAX_LIVE_PREDICTIONS:
            live_predictions = live_predictions[-MAX_LIVE_PREDICTIONS:]
        
        return {
            "message": f"{count} transactions simulated successfully",
            "new_transactions": new_tx,
            "predictions": new_predictions,
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.post("/generate-strategy")
def generate_strategy(segment: SegmentDetailsRequest):
    """Generate marketing strategy using Groq API (RanNeeti AI)."""
    import os
    import requests
    
    # Secure API Key handling via Environment Variable
    api_key = os.environ.get("GROQ_API_KEY")
    
    prompt = f"""
    You are RanNeeti AI, a world-class marketing strategist AI.
    Please analyze the following customer segment and provide a specific, personalized email copy and a discount strategy.
    
    Segment Name: {segment.name}
    Number of Customers: {segment.count} ({segment.percentage}%)
    Average Recency: {segment.avg_recency:.1f} days
    Average Frequency: {segment.avg_frequency:.1f} purchases
    Average Spend: ${segment.avg_monetary:.2f}
    Total Revenue: ${segment.total_revenue:.2f}
    
    Structure your response using Markdown:
    1. **RanNeeti Strategic Analysis**: Briefly explain what these metrics mean.
    2. **Recommended Discount Strategy**: Specific promotional offer.
    3. **Email Copy Draft**: A ready-to-use personalized email template.
    
    Keep it professional, actionable, and engaging.
    """
    
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 800
            },
            timeout=15
        )
        response.raise_for_status()
        result = response.json()
        strategy_text = result["choices"][0]["message"]["content"]
        return {"strategy": strategy_text}
    except Exception as e:
        return {"strategy": f"**RanNeeti AI Encountered an Error:**\\nCould not generate strategy. Make sure the API key is active. Details: {str(e)}"}
@app.get("/models/importance")
def get_model_importance(model_name: str = Query("randomforest")):
    """Get feature importance for a specific model."""
    if predictor is None:
        raise HTTPException(status_code=404, detail="Predictor not initialized")
    
    importance = predictor.get_feature_importance(model_name)
    if not importance:
        raise HTTPException(status_code=404, detail=f"No importance data for model: {model_name}")
        
    return importance


@app.get("/data/affinity")
def get_product_affinity():
    """Calculate product category co-occurrence (simplified Market Basket Analysis)."""
    if not os.path.exists(DB_PATH):
        # Fallback to sample data if DB missing
        return {"Affinity": 0.5, "Categories": ["Electronics", "Clothing"]}
        
    conn = sqlite3.connect(DB_PATH)
    try:
        # Cross join to find category co-occurrences by customer_id
        query = """
        SELECT t1.product_category as category_a, t2.product_category as category_b, COUNT(DISTINCT t1.customer_id) as co_occurrence
        FROM transactions t1
        JOIN transactions t2 ON t1.customer_id = t2.customer_id
        WHERE t1.product_category < t2.product_category
        GROUP BY t1.product_category, t2.product_category
        ORDER BY co_occurrence DESC
        LIMIT 20
        """
        affinity_df = pd.read_sql(query, conn)
        return affinity_df.to_dict(orient="records")
    finally:
        conn.close()


@app.get("/data/summary")
def get_data_summary():
    """Get dataset statistics."""
    if rfm_df is None or transactions_df is None:
        raise HTTPException(status_code=404, detail="No data available. Generate data first.")

    # Convert to datetime to avoid str vs Timestamp comparison errors
    date_col = pd.to_datetime(transactions_df["transaction_date"], format='mixed')
    
    summary = {
        "total_customers": len(rfm_df),
        "total_transactions": len(transactions_df),
        "date_range": {
            "start": str(date_col.min()),
            "end": str(date_col.max()),
        },
        "revenue": {
            "total": round(rfm_df["monetary"].sum(), 2),
            "average": round(rfm_df["monetary"].mean(), 2),
            "median": round(rfm_df["monetary"].median(), 2),
        },
        "rfm_statistics": {
            "recency": {
                "mean": round(rfm_df["recency"].mean(), 2),
                "median": round(rfm_df["recency"].median(), 2),
                "min": int(rfm_df["recency"].min()),
                "max": int(rfm_df["recency"].max()),
            },
            "frequency": {
                "mean": round(rfm_df["frequency"].mean(), 2),
                "median": round(rfm_df["frequency"].median(), 2),
                "min": int(rfm_df["frequency"].min()),
                "max": int(rfm_df["frequency"].max()),
            },
            "monetary": {
                "mean": round(rfm_df["monetary"].mean(), 2),
                "median": round(rfm_df["monetary"].median(), 2),
                "min": round(rfm_df["monetary"].min(), 2),
                "max": round(rfm_df["monetary"].max(), 2),
            },
        },
    }

    summary["category_distribution"] = {str(k): v for k, v in transactions_df["product_category"].value_counts().to_dict().items()}
    summary["payment_method_distribution"] = {str(k): v for k, v in transactions_df["payment_method"].value_counts().to_dict().items()}

    return summary


@app.get("/data/rfm")
def get_rfm_data(
    limit: int = Query(100, ge=1, le=10000),
    offset: int = Query(0, ge=0)
):
    """Get RFM data with pagination. Uses in-memory data if available for up-to-date segments."""
    if rfm_df is None:
        raise HTTPException(status_code=404, detail="No data available. Generate data first.")

    # Slice the in-memory DataFrame
    total = len(rfm_df)
    data = rfm_df.iloc[offset:offset + limit].replace({np.nan: None}).to_dict(orient="records")

    return {
        "data": data,
        "total": int(total),
        "limit": limit,
        "offset": offset,
    }


@app.post("/train")
def train_models(config: Optional[TrainingConfig] = None):
    """Train all ML models."""
    global predictor, rfm_df

    if rfm_df is None:
        raise HTTPException(status_code=400, detail="No data available. Generate data first.")

    trainer = ModelTrainer(MODELS_DIR)
    results = trainer.train(rfm_df)

    # Reload predictor
    predictor = Predictor(MODELS_DIR)

    return {
        "message": "Models trained successfully",
        "results": results,
    }


@app.get("/models/metrics")
def get_model_metrics():
    """Get model comparison metrics."""
    results_path = os.path.join(MODELS_DIR, "training_results.json")

    if not os.path.exists(results_path):
        raise HTTPException(status_code=404, detail="No trained models. Train models first.")

    with open(results_path, "r") as f:
        results = json.load(f)

    return results


@app.post("/predict")
def predict(request: PredictionRequest):
    """Predict purchase behavior for a single customer."""
    if predictor is None or not predictor.models_loaded:
        raise HTTPException(status_code=400, detail="Models not loaded. Train models first.")

    features = request.model_dump()
    model_name = features.pop("model_name", None)
    result = predictor.predict_purchase(features, model_name=model_name)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return {
        "customer_id": request.customer_id,
        **result,
    }


@app.post("/predict/batch")
def predict_batch(request: BatchPredictionRequest):
    """Predict for multiple customers."""
    if predictor is None or not predictor.models_loaded:
        raise HTTPException(status_code=400, detail="Models not loaded.")

    customers = [c.model_dump() for c in request.customers]
    results = predictor.predict_batch(customers, request.model_name)

    return {
        "predictions": results,
        "total": len(results),
    }


@app.get("/segments")
def get_segments():
    """Get all segments with statistics."""
    if rfm_df is None:
        raise HTTPException(status_code=404, detail="No data available.")

    if "segment_name" not in rfm_df.columns:
        # Run segmentation
        if predictor is None or not predictor.models_loaded:
            raise HTTPException(status_code=400, detail="Models not trained.")
        
        try:
            # Use only base RFM features for segmentation
            X_scaled = predictor.segmentation_scaler.transform(
                rfm_df[RFM_FEATURES].fillna(0)
            )
            rfm_df["segment"] = predictor.kmeans.predict(X_scaled)
            segment_mapping = predictor.segment_mapping
            # Ensure segment mapping keys are integers
            mapping = {int(k): v for k, v in segment_mapping.items()}
            rfm_df["segment_name"] = rfm_df["segment"].map(mapping)
        except Exception as e:
            print(f"Segmentation error: {e}")
            raise HTTPException(status_code=500, detail=f"Segmentation calculation failed: {str(e)}")

    segments = []
    # Use dropna=False to see potential issues
    for segment_name in rfm_df["segment_name"].unique():
        if pd.isna(segment_name): continue
        segment_data = rfm_df[rfm_df["segment_name"] == segment_name]
        segments.append({
            "name": str(segment_name),
            "count": len(segment_data),
            "percentage": round(len(segment_data) / len(rfm_df) * 100, 2),
            "avg_recency": round(segment_data["recency"].mean(), 2),
            "avg_frequency": round(segment_data["frequency"].mean(), 2),
            "avg_monetary": round(segment_data["monetary"].mean(), 2),
            "total_revenue": round(segment_data["monetary"].sum(), 2),
        })

    return {"segments": segments, "total": len(rfm_df)}


@app.get("/segments/{segment_id}")
def get_segment_details(segment_id: int):
    """Get details of a specific segment."""
    if rfm_df is None or "segment" not in rfm_df.columns:
        raise HTTPException(status_code=404, detail="Segment data not available.")

    segment_data = rfm_df[rfm_df["segment"] == segment_id]

    if len(segment_data) == 0:
        raise HTTPException(status_code=404, detail=f"Segment {segment_id} not found.")

    return {
        "segment_id": segment_id,
        "segment_name": segment_data["segment_name"].iloc[0],
        "count": len(segment_data),
        "statistics": {
            "recency": {
                "mean": round(segment_data["recency"].mean(), 2),
                "std": round(segment_data["recency"].std(), 2),
            },
            "frequency": {
                "mean": round(segment_data["frequency"].mean(), 2),
                "std": round(segment_data["frequency"].std(), 2),
            },
            "monetary": {
                "mean": round(segment_data["monetary"].mean(), 2),
                "std": round(segment_data["monetary"].std(), 2),
            },
        },
        "customers": segment_data.head(10).to_dict(orient="records"),
    }


@app.get("/data/transactions")
def get_transactions(
    limit: int = Query(100, ge=1, le=10000),
    offset: int = Query(0, ge=0)
):
    """Get transaction data with pagination using in-memory dataframe to include live traffic."""
    if transactions_df is None:
        raise HTTPException(status_code=404, detail="No transaction data available.")
        
    # Sort descending to get latest first for the live traffic feed
    df_sorted = transactions_df.sort_index(ascending=False)
    # Use where/notnull to handle NaN for JSON compliance
    data = df_sorted.iloc[offset:offset + limit].replace({np.nan: None}).to_dict(orient="records")
    total = len(transactions_df)

    return {
        "data": data,
        "total": int(total),
        "limit": limit,
        "offset": offset,
    }


@app.get("/data/monthly-revenue")
def get_monthly_revenue():
    """Get monthly revenue trends."""
    if transactions_df is None:
        raise HTTPException(status_code=404, detail="No data available.")

    df = transactions_df.copy()
    # Ensure transaction_date is a datetime Series to allow .dt accessor
    datetimes = pd.to_datetime(df["transaction_date"])
    df["month"] = datetimes.dt.to_period("M").astype(str)

    monthly = df.groupby("month").agg(
        revenue=("total_amount", "sum"),
        transactions=("transaction_id", "count"),
        customers=("customer_id", "nunique")
    ).reset_index()

    monthly["month"] = monthly["month"].astype(str)

    return {
        "data": monthly.to_dict(orient="records"),
    }


@app.get("/live-predictions")
def get_live_predictions(limit: int = Query(50, ge=1, le=200)):
    """Get recent real-time ML predictions from the live traffic simulator."""
    recent = live_predictions[-limit:] if live_predictions else []
    return {
        "predictions": list(reversed(recent)),
        "total": len(live_predictions),
    }


@app.get("/live-stats")
def get_live_stats():
    """Get aggregate statistics from live predictions."""
    if not live_predictions:
        return {
            "total_predictions": 0,
            "will_purchase_count": 0,
            "will_not_purchase_count": 0,
            "avg_probability": 0,
            "segment_distribution": {},
        }

    will_purchase = sum(1 for p in live_predictions if p.get("prediction", {}).get("will_purchase", False))
    total = len(live_predictions)
    avg_prob = sum(p.get("prediction", {}).get("probability", 0) for p in live_predictions) / total if total > 0 else 0

    seg_dist = {}
    for p in live_predictions:
        seg_name = p.get("segment", {}).get("segment_name", "Unknown")
        seg_dist[seg_name] = seg_dist.get(seg_name, 0) + 1

    return {
        "total_predictions": total,
        "will_purchase_count": will_purchase,
        "will_not_purchase_count": total - will_purchase,
        "avg_probability": round(avg_prob, 4),
        "segment_distribution": seg_dist,
    }


if __name__ == "__main__":
    import uvicorn
    # Use 127.0.0.1 for local development to avoid resolution issues
    uvicorn.run("server:app", host="127.0.0.1", port=8001, reload=False)
