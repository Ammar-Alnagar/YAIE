import traceback
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..engine import InferenceEngine


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
            # TODO: Implement the chat completion logic using the inference engine
            # This should handle continuous batching and generate responses appropriately
            # 1. Convert incoming request to the format expected by the engine
            # 2. Pass to the engine for processing
            # 3. Format the response in OpenAI-compatible format
            # 4. Handle streaming if requested

            raise NotImplementedError(
                "Chat completions endpoint not yet implemented - this is an exercise for the learner"
            )

        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v1/models")
    async def list_models():
        # Return the loaded model information
        return {
            "object": "list",
            "data": [{"id": model_name, "object": "model", "owned_by": "user"}],
        }

    return app
