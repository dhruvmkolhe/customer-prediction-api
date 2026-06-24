"""
Backward-compatible backend entrypoint.

The canonical FastAPI app lives in `server.py`. This module keeps older
imports and test references working while standardizing startup on a single app.
"""

# pyrefly: ignore [missing-import]
from server import app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8001, reload=False)
