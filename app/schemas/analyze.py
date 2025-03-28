from pydantic import BaseModel

class AnalyzeRequest(BaseModel):
    query: str

class AnalyzeResponse(BaseModel):
    pdf_report: str
    insights: str

class ReviewRequest(BaseModel):
    edited_insights: str # This is the human-edited insights that will be used to regenerate the report



# class AnalyzeRequest(BaseModel):
#     query: str
#     edited_insights: str

# class AnalyzeResponse(BaseModel):
#     pdf_report: str
#     insights: str