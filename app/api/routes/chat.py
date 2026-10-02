from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.api.deps import get_agent, get_pipeline
from app.config import Settings, get_settings
from app.core.logging import Timer, get_logger
from app.core.security import verify_api_key
from app.models.schemas import AgentChatResponse, AgentStep, ChatRequest, ChatResponse, Source
from app.rag.agent import Agent
from app.rag.pipeline import RAGPipeline

router = APIRouter(prefix="/chat", tags=["chat"])
limiter = Limiter(key_func=get_remote_address)
log = get_logger(__name__)


@router.post("", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit(lambda: get_settings().rate_limit)
async def chat(
    request: Request,
    body: ChatRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
    settings: Settings = Depends(get_settings),
):
    with Timer() as t:
        result = await pipeline.answer(body.query, top_k=body.top_k)
    log.info("chat_completed", query=body.query, latency_ms=t.elapsed_ms, n_sources=len(result["sources"]))
    sources = [
        Source(content=s["content"], source=s["metadata"].get("source", "unknown"), chunk_id=s["id"], score=s["score"])
        for s in result["sources"]
    ]
    return ChatResponse(answer=result["answer"], sources=sources, latency_ms=t.elapsed_ms, model=settings.ollama_model)


@router.post("/stream", dependencies=[Depends(verify_api_key)])
@limiter.limit(lambda: get_settings().rate_limit)
async def chat_stream(request: Request, body: ChatRequest, pipeline: RAGPipeline = Depends(get_pipeline)):
    async def event_generator():
        async for token in pipeline.answer_stream(body.query, top_k=body.top_k):
            yield f"data: {token}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/agent", response_model=AgentChatResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit(lambda: get_settings().rate_limit)
async def chat_agent(request: Request, body: ChatRequest, agent: Agent = Depends(get_agent)):
    with Timer() as t:
        answer, steps = await agent.run(body.query)
    log.info("agent_completed", query=body.query, latency_ms=t.elapsed_ms, n_steps=len(steps))
    return AgentChatResponse(
        answer=answer,
        steps=[AgentStep(thought=s.thought, action=s.action, action_input=s.action_input, observation=s.observation) for s in steps],
        latency_ms=t.elapsed_ms,
    )
