import time
import traceback
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from engine import InferenceEngine


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Dict[str, int]


from fastapi.responses import StreamingResponse
import json


def create_app(model_name: str) -> FastAPI:
    """
    Create FastAPI application with OpenAI-compatible endpoints

    Args:
        model_name: Name of the model to load

    Returns:
        FastAPI application instance
    """
    app = FastAPI(title="YAIE API", version="0.1.0")

    # Load model when the server starts
    engine = InferenceEngine(model_name)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        try:
            # Convert Pydantic models to dicts for the engine
            messages_dicts = [
                {"role": msg.role, "content": msg.content}
                for msg in request.messages
            ]

            # Call engine
            # We pass relevant parameters from request
            kwargs = {}
            if request.max_tokens is not None:
                kwargs["max_tokens"] = request.max_tokens
            if request.temperature is not None:
                kwargs["temperature"] = request.temperature
            if request.top_p is not None:
                kwargs["top_p"] = request.top_p

            if request.stream:
                # Return streaming response
                def generate_stream():
                    # Use the engine's streaming method directly
                    for chunk in engine.chat_completion_stream(messages_dicts, **kwargs):
                        yield f"data: {json.dumps(chunk)}\n\n"

                    yield "data: [DONE]\n\n"

                return StreamingResponse(generate_stream(), media_type="text/event-stream")
            else:
                # Return non-streaming response
                response = engine.chat_completion(messages_dicts, **kwargs)
                return response

        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v1/models")
    async def list_models():
        # Return the loaded model information
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "owned_by": "user",
                    "created": int(time.time()),
                }
            ],
        }

    @app.get("/health")
    async def health_check():
        # Simple health check endpoint
        return {"status": "healthy", "model": model_name}

    return app
