# API Module (`api.py`)

The `api.py` module defines the HTTP API for the Mini-YAIE kernel services, providing OpenAI-compatible endpoints. It is built using FastAPI and leverages Pydantic for request and response model validation.

## Pydantic Models

### `ChatMessage`
Represents a single message in a chat conversation.

*   `role` (str): The role of the author of this message (e.g., "user", "assistant", "system").
*   `content` (str): The content of the message.

### `ChatCompletionRequest`
Defines the structure for a chat completion request.

*   `model` (str): The name of the model to use for completion.
*   `messages` (List[ChatMessage]): A list of messages comprising the conversation so far.
*   `temperature` (Optional[float], default: 1.0): Controls the randomness of the output. Higher values mean more random.
*   `top_p` (Optional[float], default: 1.0): Controls the diversity of the output. A value of 0.1 means only the tokens comprising the top 10% probability mass are considered.
*   `max_tokens` (Optional[int]): The maximum number of tokens to generate in the completion.
*   `stream` (Optional[bool], default: False): If true, partial message deltas will be sent, like in ChatGPT.

### `Choice`
Represents a single completion choice returned by the API.

*   `index` (int): The index of the choice in the list of choices.
*   `message` (ChatMessage): The generated chat message.
*   `finish_reason` (Optional[str]): The reason the model stopped generating tokens (e.g., "stop", "length").

### `ChatCompletionResponse`
The full response structure for a non-streaming chat completion request.

*   `id` (str): A unique identifier for the completion.
*   `object` (str): The object type, always "chat.completion".
*   `created` (int): The Unix timestamp (in seconds) of when the completion was created.
*   `model` (str): The model used for the completion.
*   `choices` (List[Choice]): A list of completion choices.
*   `usage` (Dict[str, int]): Information about the API usage (e.g., prompt tokens, completion tokens).

## Functions

### `create_app(model_name: str) -> FastAPI`
This function initializes and configures the FastAPI application.

*   **Args**:
    *   `model_name` (str): The identifier for the language model to be loaded and served by the API.
*   **Returns**:
    *   `FastAPI`: An instance of the FastAPI application.

## Endpoints

### `POST /v1/chat/completions`
Handles requests for chat completions, supporting both streaming and non-streaming responses.

*   **Request Body**: `ChatCompletionRequest`
*   **Responses**:
    *   **Non-streaming (if `stream=False`)**: Returns a `ChatCompletionResponse` JSON object.
    *   **Streaming (if `stream=True`)**: Returns `text/event-stream` with server-sent events, where each event is a JSON chunk representing partial message deltas.
*   **Error Handling**: Catches exceptions during processing and returns an `HTTPException` with a 500 status code.

### `GET /v1/models`
Returns a list of models currently available and served by the API.

*   **Response**: A JSON object containing a list of model details.
    ```json
    {
      "object": "list",
      "data": [
        {
          "id": "model_name_here",
          "object": "model",
          "owned_by": "user",
          "created": 1678886400 // Example timestamp
        }
      ]
    }
    ```

### `GET /health`
A simple endpoint to check the health status of the API and the loaded model.

*   **Response**: A JSON object indicating the status and the model name.
    ```json
    {
      "status": "healthy",
      "model": "model_name_here"
    }
    ```
