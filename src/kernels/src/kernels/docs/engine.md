# Engine Module (`engine.py`)

The `engine.py` module implements the core `InferenceEngine` for Mini-YAIE, designed with an SGLang-style approach to optimize large language model (LLM) inference. This includes features like continuous batching, radial attention, and prefix sharing for enhanced performance and efficiency.

## `InferenceEngine` Class

The `InferenceEngine` class is the central component responsible for managing the LLM inference process, from model loading and request scheduling to token generation and response formatting.

### `__init__(self, model_name: str)`

Initializes the inference engine with the specified model.

*   **Parameters**:
    *   `model_name` (str): The identifier for the language model to be loaded (e.g., from HuggingFace or a local path).
*   **Initialization**:
    *   Loads the `tokenizer` and the `model` using the `ModelLoader`.
    *   Initializes an `SGLangScheduler` for managing continuous batching and prefix grouping.
    *   Sets up `RadixAttentionWithPagedKVCache` to handle the radial attention mechanism, leveraging paged KV-cache.
    *   Configures `KVCacheManager` for efficient memory management of the KV cache.
    *   Connects the scheduler to the KV cache manager for SGLang-specific optimizations.
*   **TODOs**: The initializer also outlines future enhancements, such as integrating a Radix tree for prefix matching, multi-step attention processors, advanced memory pooling, request preemption, and chunked prefill for very long prompts.

### `_load_tokenizer(self) -> PreTrainedTokenizer`

A private helper method to load the appropriate tokenizer for the specified model using `ModelLoader`.

*   **Returns**:
    *   `PreTrainedTokenizer`: An instance of the HuggingFace tokenizer.

### `_load_model(self)`

A private helper method to load the LLM model using `ModelLoader`.

*   **Returns**:
    *   The loaded LLM model instance.

### `generate(self, prompts: List[str], **kwargs) -> List[str]`

The primary method for generating text responses based on a list of prompts. It orchestrates the process by adding requests to the scheduler and then running the generation loop.

*   **Parameters**:
    *   `prompts` (List[str]): A list of input text prompts for which to generate responses.
    *   `**kwargs`: Additional generation parameters, such as `max_tokens`, `temperature`, and `top_p`.
*   **Returns**:
    *   `List[str]`: A list of generated text responses, corresponding to the input prompts.

### `_run_generation_loop(self, request_ids: List[str]) -> List[str]`

A simplified internal method that simulates the SGLang-style continuous batching loop. It processes requests one by one, tokenizes prompts, performs model forward passes, samples the next token using `SamplingKernel`, and decodes the generated tokens.

*   **Parameters**:
    *   `request_ids` (List[str]): A list of request identifiers managed by the scheduler.
*   **Returns**:
    *   `List[str]`: A list of generated text segments for each request.

### `chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]`

Generates a single chat completion response formatted to be compatible with the OpenAI API specification. It handles the formatting of chat messages using the tokenizer's chat template (if available) or a simple fallback.

*   **Parameters**:
    *   `messages` (List[Dict[str, str]]): A list of message dictionaries, where each dictionary contains `"role"` and `"content"` keys.
    *   `**kwargs`: Additional generation parameters (e.g., `temperature`, `top_p`, `max_tokens`).
*   **Returns**:
    *   `Dict[str, Any]`: A dictionary representing the chat completion response, adhering to the OpenAI API format.

### `chat_completion_stream(self, messages: List[Dict[str, str]], **kwargs)`

Generates a streaming chat completion response, yielding token chunks compatible with the OpenAI streaming API. This method allows for real-time updates as tokens are generated.

*   **Parameters**:
    *   `messages` (List[Dict[str, str]]): A list of message dictionaries, each with `"role"` and `"content"` keys.
    *   `**kwargs`: Additional generation parameters (e.g., `temperature`, `top_p`, `max_tokens`).
*   **Yields**:
    *   `Dict[str, Any]`: Dictionary chunks representing partial chat completion responses, formatted as server-sent events.
