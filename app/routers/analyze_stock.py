from fastapi import APIRouter, HTTPException
from app.services.analyze_stock import analyze_stock
from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse

router = APIRouter(prefix="/analyze", tags=["Analyze Stock"])

@router.post("/", response_model=AnalyzeResponse)
async def analyze_stock_endpoint(request: AnalyzeRequest):
    try:
        result = await analyze_stock(request.query, request.edited_insights)
        if not result or "pdf_report" not in result:
            raise HTTPException(status_code=400, detail="Failed to generate report.")
        return AnalyzeResponse(pdf_report=result["pdf_report"], insights=result["insights"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
