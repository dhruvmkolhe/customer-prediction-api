#!/bin/bash
# Increase reliability: Add backend to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/backend

# Start FastAPI backend in the background on port 8001
uvicorn backend.server:app --host 127.0.0.1 --port 8001 &

# Start Streamlit frontend on the port Render provides ($PORT)
streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0
