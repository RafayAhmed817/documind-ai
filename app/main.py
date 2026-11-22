from functools import lru_cache
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import get_settings
from app.rag import RAGService

app = FastAPI(title="PDF AI", version="1.0.0")


@lru_cache
def get_rag() -> RAGService:
    return RAGService()


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_id: str = Field(min_length=64, max_length=64)
    k: int = Field(default=4, ge=1, le=10)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    limit = get_settings().max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail="PDF exceeds upload limit")

    try:
        return get_rag().ingest(data, file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/ask")
def ask(request: QuestionRequest):
    try:
        return get_rag().ask(request.question, request.document_id, request.k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc