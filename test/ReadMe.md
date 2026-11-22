# DocuChat AI — LangChain + Gemini + FastAPI + Docker

A learning-first, end-to-end PDF RAG application. Uses LangChain integrations, local HuggingFace embeddings, persistent Chroma, Google Gemini, FastAPI, Streamlit, and Docker Compose.

## Requirements

- Docker Desktop / Docker Engine with Compose
- A Google AI Studio API key with Gemini API access (free-tier access and quotas depend on region/account)
- Internet on first run for Python dependencies and the embedding model; Gemini requires an internet connection for questions
- About 4 GB of free RAM recommended for containers and local embeddings

## Run

```bash
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY

docker compose up --build -d
```

- UI: http://localhost:8501
- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

First upload can take longer while the embedding model downloads. View logs with `docker compose logs -f api`. Stop with `docker compose down` (volumes remain); `docker compose down -v` removes stored documents and model cache.

## API

`POST /documents` accepts multipart field `file` (PDF, up to 15 MB) and returns `document_id`, `chunks`, and `status`.

`POST /ask` accepts:

```json
{"document_id": "<id from upload>", "question": "What is this document about?", "k": 4}
```

The answer includes `sources` with page numbers and excerpts. Source excerpts are retrieved evidence, not proof that every generated claim is correct.

## Tests

With a local Python 3.11 environment:

```bash
pip install -r requirements.txt
pytest -q
```

Or after building:

```bash
docker compose run --rm api pytest -q
```

## Study order

1. `app/rag.py` — `extract_pdf`: page-wise PDF extraction and metadata.
2. `make_chunks`: recursive splitting, overlap and source preservation.
3. `RAGService.ingest`: SHA-256 deduplication, local embeddings and Chroma persistence.
4. `RAGService.ask`: similarity search, document-specific filtering, prompt and Gemini chain.
5. `app/main.py`: validation, API contracts and errors.
6. `frontend.py`: Streamlit → FastAPI communication.
7. `compose.yaml`: API/UI services and persistent Docker volumes.
8. `tests/test_rag.py`: fast tests that don't call Gemini.

## Known limitations and future exercises

- Only text-based PDFs; scanned PDFs need OCR.
- No login or tenant isolation; this is a local learning project, **not** safe to expose publicly as-is.
- No streaming, retries, async job queue, or production rate limiting.
- Deduplication uses document SHA-256; filenames are not separate document identities.
- Prompting helps limit hallucinations but does not guarantee factual correctness.
- Retrieved page citations should be checked against the PDF.
- Provider API may return rate-limit, quota, model-unavailable, or billing errors. If Gemini 2.5 Flash is unavailable, change `GEMINI_MODEL` in `.env` to an available model shown in your AI Studio account.
- Pin dependency versions and run full integration tests before production use.

## Practice tasks

1. Add tests for empty scanned PDFs and duplicate uploads.
2. Compare chunk sizes 500, 900 and 1400 on the same PDF (clear collection between runs).
3. Build a 10-question evaluation set with expected page references and measure retrieval hit rate.
4. Add a `DELETE /documents/{id}` endpoint.
5. Add structured error handling for Gemini quota errors.
