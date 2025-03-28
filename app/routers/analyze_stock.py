from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

from app.services.analyze_stock import analyze_stock
from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse

router = APIRouter(prefix="/analyze", tags=["Analyze Stock"])

# Define the reports directory
REPORT_DIR = Path("static/reports")

@router.post("/", response_model=AnalyzeResponse)
async def analyze_stock_endpoint(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    Analyzes stock data and returns a report and insights.
    """
    result = await analyze_stock(request.query)  

    if not result.get("pdf_report"):
        raise HTTPException(status_code=500, detail="Failed to generate report")

    return result  # Return AI-generated insights and report.


@router.get("/download/{filename}")
async def download_pdf(filename: str):
    """
    Serves the generated PDF file for download.
    Example: if the generated filename from the post request is TSLA_financial_report.pdf as in \n\n - "pdf_report": "/static/reports/TSLA_financial_report.pdf", \n\n then it can be downloaded by passing in TSLA_financial_report.pdf as the filename.
    """
    file_path = REPORT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")

    return FileResponse(file_path, filename=filename, media_type="application/pdf")






# @router.post("/", response_model=AnalyzeResponse)
# async def analyze_stock_endpoint(request: AnalyzeRequest, background_tasks: BackgroundTasks):
#     """
#     Analyzes stock data and returns a report.
#     """
#     result = await analyze_stock(request.query, request.edited_insights)

#     if not result.get("pdf_report"):
#         raise HTTPException(status_code=500, detail="Failed to generate report")

#     return result


# @router.post("/review", response_model=AnalyzeResponse)
# async def review_insights(request: ReviewRequest):
#     """
#     Step 2: Accepts human-edited insights and regenerates the report.
#     """
#     final_report = await analyze_stock(None, request.edited_insights)  # Now we use edited insights

#     if not final_report.get("pdf_report"):
#         raise HTTPException(status_code=500, detail="Failed to generate report")

#     return final_report
