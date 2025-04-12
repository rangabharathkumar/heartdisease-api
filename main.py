from app.main import app

# This file re-exports the FastAPI app from app/main.py
# This ensures compatibility with both local development and deployment

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10006) 