# Customer Prediction System

Real-time customer purchase prediction and segmentation with a FastAPI backend
and a Streamlit dashboard.

## What it does

- Generates synthetic transaction data
- Builds RFM features for segmentation
- Trains multiple purchase-prediction models
- Serves real-time predictions through an API
- Simulates live traffic with real-time ML predictions per transaction
- Provides an interactive Streamlit dashboard for exploration and what-if
  analysis

## Project Structure

- `backend/` FastAPI app, model training, prediction service, tests, and saved
  artifacts
- `frontend/` Streamlit dashboard and visualizations
- `backend/data/` generated CSV and SQLite data
- `backend/` trained models and metadata

## Core Features

- Customer segmentation with K-Means
- Purchase prediction with Random Forest, XGBoost, Logistic Regression, and
  Gradient Boosting
- Real-time ML predictions on simulated live traffic
- Live prediction feed with segment classification and purchase probability
- Feature importance and model comparison views
- Batch prediction support
- Downloadable CSV, JSON, and Excel templates

## Requirements

- Python 3.11+ recommended
- Packages from `backend/requirements.txt`
- Packages from `frontend/requirements.txt`

## Setup

1. Create and activate a virtual environment.
2. Install backend dependencies.
3. Install frontend dependencies.

Example:

```powershell
python -m pip install -r backend/requirements.txt
python -m pip install -r frontend/requirements.txt
```

## Run the Backend

From `backend/`:

```powershell
uvicorn server:app --host 127.0.0.1 --port 8001
```

This matches the Streamlit client configuration.

## Run the Frontend

From `frontend/`:

```powershell
streamlit run app.py
```

## Recommended Workflow

1. Open the dashboard.
2. Generate data.
3. Train models.
4. Explore the data and segment analysis pages.
5. Use the prediction page for single or batch inference.
6. Start the Live Traffic simulator to see real-time ML predictions.

## API Endpoints

- `GET /` health check
- `POST /generate-data` create synthetic data
- `POST /train` train all models
- `GET /data/summary` dataset summary
- `GET /models/metrics` model comparison results
- `GET /models/importance` feature importance
- `POST /predict` single prediction
- `POST /predict/batch` batch prediction
- `GET /segments` segment overview
- `GET /segments/{segment_id}` segment details
- `POST /simulate-traffic` simulate live transactions with real-time ML predictions
- `GET /live-predictions` retrieve recent real-time predictions
- `GET /live-stats` aggregate live prediction statistics

## Notes

- The app is designed to keep working even if one saved model artifact is
  incompatible.
- If you retrain models in a different library version, regenerate the files in
  `backend/`.
- The Excel template on the prediction page matches the API field names.

## Troubleshooting

- If the frontend says the API is unavailable, make sure the backend is running
  on `127.0.0.1:8001`.
- If models are not loaded, generate data and retrain from the dashboard.
- If you see warnings about serialized models from older library versions,
  retraining usually fixes them.
