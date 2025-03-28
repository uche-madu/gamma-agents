import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.routers import analyze_stock

# Create the reports directory if it doesn't exist
REPORT_DIR = "static/reports"
os.makedirs(REPORT_DIR, exist_ok=True)

app = FastAPI(
    title="Stock Analysis API",
    version="1.0.0",
    description=(
        "The Stock Analysis API provides advanced stock analysis and forecasting "
        "capabilities by integrating machine learning models with a human-in-the-loop "
        "review process. Analysts can submit stock queries, review AI-generated reports in PDF, "
        "and provide feedback to refine insights. "
    )
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Mount the static folder for reports
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(analyze_stock.router)

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Stock Analysis API!",
        "project": "Stock Analysis API",
        "version": app.version,
        "description": app.description,
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        }
    }
