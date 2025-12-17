# API & Serving: OpenAI-Compatible Server

## Overview

The API server in Mini-YAIE implements an OpenAI-compatible interface, allowing the engine to be used with existing applications and tools designed for OpenAI's API. The server uses FastAPI to provide RESTful endpoints with proper request/response handling, streaming support, and health monitoring.

## API Design Philosophy

The server follows OpenAI's API specification to ensure compatibility with existing tools and applications while leveraging the advanced features of the SGLang-style inference engine. This approach allows users to:

- Use existing OpenAI clients without modification
- Take advantage of Mini-YAIE's performance optimizations
- Integrate with tools built for OpenAI's API format
- Maintain familiar request/response patterns

## Core Architecture

### FastAPI Application

The server is built using FastAPI for high-performance web serving:

```python
def create_app(model_name: str) -> FastAPI:
    app = FastAPI(title="YAIE API", version="0.1.0")
    engine = InferenceEngine(model_name)
    return app
```

### Main Components

1. **Inference Engine Integration**: Connects API endpoints to the core inference engine
2. **Request Validation**: Pydantic models for request/response validation
3. **Streaming Support**: Server-sent events for real-time token streaming
4. **Error Handling**: Proper HTTP error codes and message formatting
5. **Health Monitoring**: Endpoints for system status and availability

## API Endpoints

### 1. Chat Completions Endpoint

The main endpoint follows OpenAI's `/v1/chat/completions` specification:

```python
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # Implementation handles both streaming and non-streaming responses
```

#### Request Schema

```python
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
```

#### Response Schema

```python
class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Dict[str, int]
```

#### Supported Parameters

- **model**: Model identifier (passed during server startup)
- **messages**: List of message objects with role and content
- **temperature**: Sampling temperature (0.0-2.0 recommended)
- **top_p**: Nucleus sampling threshold (0.0-1.0)
- **max_tokens**: Maximum tokens to generate
- **stream**: Whether to stream responses (true/false)

### 2. Model Listing Endpoint

Lists the available model:

```python
@app.get("/v1/models")
async def list_models():
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
```

### 3. Health Check Endpoint

Simple health monitoring:

```python
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": model_name}
```

## Streaming Implementation

### Streaming vs Non-Streaming

The server supports both response formats using the same endpoint:

```python
if request.stream:
    # Return streaming response
    return StreamingResponse(generate_stream(), media_type="text/event-stream")
else:
    # Return non-streaming response
    response = engine.chat_completion(messages_dicts, **kwargs)
    return response
```

### Streaming Response Format

The streaming implementation generates Server-Sent Events (SSE):

```python
def generate_stream():
    for chunk in engine.chat_completion_stream(messages_dicts, **kwargs):
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"
```

Each chunk follows OpenAI's streaming format:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion.chunk",
  "created": 1234567890,
  "model": "model-name",
  "choices": [{
    "index": 0,
    "delta": {"content": "token"},
    "finish_reason": null
  }]
}
```

## Integration with Inference Engine

### Request Processing Flow

1. **API Request**: Received through FastAPI endpoints
2. **Validation**: Pydantic models validate request format
3. **Parameter Extraction**: Convert API parameters to engine format
4. **Engine Processing**: Call appropriate engine methods
5. **Response Formatting**: Convert engine output to API format
6. **API Response**: Return properly formatted responses

### Parameter Mapping

API parameters are mapped to engine capabilities:

```python
kwargs = {}
if request.max_tokens is not None:
    kwargs["max_tokens"] = request.max_tokens
if request.temperature is not None:
    kwargs["temperature"] = request.temperature
if request.top_p is not None:
    kwargs["top_p"] = request.top_p
```

### Message Formatting

The server handles OpenAI-style messages by converting them to a format the engine understands:

```python
messages_dicts = [
    {"role": msg.role, "content": msg.content}
    for msg in request.messages
]

# Apply chat template if available
if hasattr(self.tokenizer, 'apply_chat_template'):
    formatted_prompt = self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
else:
    # Fallback formatting
    formatted_prompt = ""
    for message in messages:
        formatted_prompt += f"{message['role'].capitalize()}: {message['content']}\n"
    formatted_prompt += "\nAssistant:"
```

## Error Handling

### HTTP Error Codes

The server properly handles various error conditions:

- **400 Bad Request**: Invalid request parameters
- **429 Too Many Requests**: Rate limiting (not implemented in basic version)
- **500 Internal Server Error**: Server-side errors during processing

### Error Response Format

Standard OpenAI-compatible error format:

```python
{
  "error": {
    "message": "Error description",
    "type": "server_error",
    "param": null,
    "code": null
  }
}
```

### Exception Handling

The server wraps all processing in try-catch blocks:

```python
try:
    # Process request
    response = engine.chat_completion(messages_dicts, **kwargs)
    return response
except Exception as e:
    traceback.print_exc()
    raise HTTPException(status_code=500, detail=str(e))
```

## Performance Considerations

### Request Batching

The API integrates with the engine's batching system:
- Multiple API requests can be batched together in the engine
- Continuous batching maintains high throughput
- Batch size limited by engine configuration

### Memory Management

The server shares memory with the inference engine:
- KV-cache is shared across API requests
- Efficient memory reuse through paged cache system
- Memory limits enforced by engine configuration

### Concurrency

FastAPI provides automatic concurrency handling:
- Async request processing
- Connection pooling
- Efficient handling of multiple simultaneous requests

## Security Considerations

### Input Validation

- Pydantic models validate all request parameters
- Type checking prevents injection attacks
- Length limits prevent excessive resource consumption

### Rate Limiting

While not implemented in the basic version, can be added:
- Per-IP rate limiting
- Request quota management
- Usage monitoring

## Deployment Configuration

### Server Startup

The server can be started with a specific model:

```bash
uvicorn server.api:app --host 0.0.0.0 --port 8000
```

### Environment Configuration

The server supports environment-based configuration:
- Model name via environment variables
- Port and host configuration
- Resource limits

## SGLang-Style Features Integration

### Continuous Batching

The API benefits from the engine's continuous batching:
- Requests are automatically batched
- High throughput maintained
- Low latency for individual requests

### Prefix Sharing

API requests with similar prefixes benefit from:
- Shared computation in radial attention
- Reduced memory usage
- Improved efficiency

### Multi-Step Processing

The API leverages the engine's multi-step capabilities:
- Efficient prefill and decode phases
- Optimized request scheduling
- Memory-aware processing

## Usage Examples

### Basic Chat Request

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model-name",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "temperature": 0.7
  }'
```

### Streaming Request

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model-name",
    "messages": [
      {"role": "user", "content": "Write a short story"}
    ],
    "stream": true
  }'
```

### Programmatic Usage

```python
import openai

# Configure client to use local server
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # Required but ignored
)

# Use standard OpenAI SDK
response = client.chat.completions.create(
    model="model-name",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7
)
```

## Monitoring and Logging

### Health Endpoints

The health endpoint can be used for monitoring:

```bash
curl http://localhost:8000/health
```

### Performance Metrics

The server can be extended with:
- Response time tracking
- Request volume monitoring
- Error rate monitoring
- Resource utilization metrics

## Advanced Features

### Model Loading

The server handles model loading automatically:
- Lazy loading when first accessed
- Caching for subsequent requests
- HuggingFace model integration

### Response Caching

The system supports response caching for:
- Repeated identical requests
- Common prompt patterns
- Improved response times for cached content

### Logging and Debugging

Comprehensive logging can be added:
- Request/response logging
- Performance metrics
- Error tracing
- Usage analytics

This OpenAI-compatible API server enables Mini-YAIE to be integrated into existing ecosystems while providing the performance benefits of SGLang-style inference optimization.