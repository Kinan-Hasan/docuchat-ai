"""Application entrypoint: wires together routers, middleware, rate limiting,
structured logging and error handling. Run with:

    uvicorn app.main:app --reload
"""
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.routes import chat, documents, health
from app.api.routes.chat import limiter
from app.config import get_settings
from app.core.logging import get_logger, new_request_id, setup_logging

settings = get_settings()
setup_logging(json_logs=settings.environment != "development")
log = get_logger(__name__)

app = FastAPI(
    title=settings.app_name,
    description="Production-style RAG assistant for document Q&A, running on a local LLM via Ollama.",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
    new_request_id()
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    log.info("request_handled", path=request.url.path, method=request.method,
              status_code=response.status_code, duration_ms=round(duration_ms, 2))
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log.error("unhandled_exception", path=request.url.path, error=str(exc))
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.include_router(health.router)
app.include_router(chat.router)
app.include_router(documents.router)


@app.get("/", tags=["health"])
async def root():
    return {"service": settings.app_name, "docs": "/docs"}
