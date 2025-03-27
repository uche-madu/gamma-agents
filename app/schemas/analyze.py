from pydantic import BaseModel

class AnalyzeRequest(BaseModel):
    query: str
    edited_insights: str

class AnalyzeResponse(BaseModel):
    pdf_report: str
    insights: str