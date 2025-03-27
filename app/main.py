from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import analyze_stock

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
