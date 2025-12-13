# API Endpoints (`server/api.py`)

## OpenAI Compatibility

Mini-YAIE strives to be drop-in compatible with OpenAI's API format.

### POST `/v1/chat/completions`

**Request Body**:

```json
{
  "model": "gpt2",
  "messages": [{ "role": "user", "content": "Hello!" }],
  "temperature": 0.7,
  "max_tokens": 100
}
```

**Response**:

```json
{
  "id": "chat-123",
  "object": "chat.completion",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hi there!"
      },
      "finish_reason": "stop"
    }
  ]
}
```
