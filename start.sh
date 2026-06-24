#!/bin/bash

# Port configuration
# Render provides $PORT for the main entry point (Streamlit)
# We'll run the backend on a fixed internal port
BACKEND_PORT=8000
FRONTEND_PORT=${PORT:-8501}

echo "🚀 Starting Customer Prediction System..."

# Add backend to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/backend

# Start FastAPI backend in the background
echo "📡 Starting FastAPI backend on port $BACKEND_PORT..."
# Using --app-dir to ensure backend is treated as the root for the app
uvicorn server:app --app-dir backend --host 0.0.0.0 --port $BACKEND_PORT &

# Wait for backend to be ready
echo "⏳ Waiting for backend to start..."
MAX_RETRIES=30
RETRY_COUNT=0
while ! curl -s http://127.0.0.1:$BACKEND_PORT/ > /dev/null; do
    RETRY_COUNT=$((RETRY_COUNT+1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "❌ Backend failed to start in time. Proceeding anyway..."
        break
    fi
    sleep 1
done

echo "✅ Backend is up!"

# Start Streamlit frontend
echo "🎨 Starting Streamlit frontend on port $FRONTEND_PORT..."
streamlit run frontend/app.py \
    --server.port $FRONTEND_PORT \
    --server.address 0.0.0.0 \
    --browser.gatherUsageStats false \
    --server.headless true
