# Chat Module (`chat.py`)

The `chat.py` module provides an interactive command-line interface for engaging with Large Language Models (LLMs) powered by the Mini-YAIE `InferenceEngine`. It handles conversation flow, user input, model response display, and persistence of the last-used model configuration.

## Configuration Management

The `chat.py` module includes utility functions for managing a user-specific configuration file (`.yaie_config.json`) located in the user's home directory. This allows the application to remember the last model used across sessions.

### `get_config_path()`

*   **Purpose**: Determines the file path for the user's Mini-YAIE configuration.
*   **Returns**: `Path`: A `pathlib.Path` object pointing to `~/.yaie_config.json`.

### `save_model_to_config(model_name: str)`

*   **Purpose**: Stores the name of the currently used model into the configuration file. This ensures that the application can remember it for subsequent sessions.
*   **Parameters**:
    *   `model_name` (str): The name of the model to save.

### `load_last_used_model()`

*   **Purpose**: Retrieves the name of the last used model from the configuration file.
*   **Returns**: `Optional[str]`: The `model_name` as a string if found, otherwise `None`.

## `run_chat(model_name: str)` Function

This is the main entry point for starting an interactive chat session.

*   **Purpose**: Initiates and manages a continuous chat dialogue between the user and the LLM.
*   **Parameters**:
    *   `model_name` (str): The name of the model to be loaded and used for the chat session.
*   **Process**:
    1.  **Configuration Update**: Calls `save_model_to_config()` to store the `model_name` for future sessions.
    2.  **Engine Initialization**: Creates an instance of the `InferenceEngine` with the specified `model_name`. This loads the model and sets up the inference pipeline.
    3.  **Conversation Loop**: Enters an infinite loop to facilitate continuous interaction:
        *   **User Input**: Prompts the user for input.
        *   **Exit Commands**: Checks for `quit`, `exit`, or `q` to gracefully terminate the session.
        *   **History Management**: Appends the user's message to `conversation_history`.
        *   **Model Inference**: Calls `engine.chat_completion()` with the `conversation_history` to get a response from the LLM.
        *   **Response Display**: Extracts and prints the assistant's response to the console.
        *   **History Update**: Appends the assistant's response to `conversation_history` to maintain context for subsequent turns.
    4.  **Error Handling**: Includes `try-except` blocks to catch:
        *   `KeyboardInterrupt`: Allows the user to exit cleanly by pressing `Ctrl+C`.
        *   General `Exception`: Catches other unexpected errors during the chat process and provides informative messages.
