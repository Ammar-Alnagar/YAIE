# Main CLI Entry Point (`main.py`)

The `main.py` module serves as the command-line interface (CLI) entry point for the Mini-YAIE application. It utilizes the `click` library to define a primary command group (`yaie`) with subcommands for starting an inference server and initiating an interactive chat session.

## `main()` Function

This function defines the main `click` group for the application. When you run `python -m src.kernels.main`, this is the entry point.

*   **Purpose**: Acts as the root command for the YAIE CLI, providing a top-level description for the tool.
*   **Usage**: Invoked directly by `click` when the `main()` group is called.

## `serve` Command

This subcommand is used to start the LLM inference server, which exposes an OpenAI-compatible API.

*   **Purpose**: Launches a FastAPI server that makes the `InferenceEngine` accessible via HTTP endpoints.
*   **Usage**: `yaie serve <MODEL_NAME> [OPTIONS]`
*   **Arguments**:
    *   `model_name` (str): **Required**. The name or path of the language model to load and serve.
*   **Options**:
    *   `--host` (str, default: `"localhost"`): The host address to bind the server to.
    *   `--port` (int, default: `8000`): The port number to bind the server to.
*   **Delegation**: This command internally calls the `run_server()` function (from `server.py`, though not explicitly documented here as it's outside `kernels`) with the provided `model_name`, `host`, and `port`.

## `chat` Command

This subcommand allows users to start an interactive text-based chat session with a loaded LLM.

*   **Purpose**: Provides a conversational interface to interact with a language model directly from the terminal.
*   **Usage**: `yaie chat [MODEL_NAME]`
*   **Arguments**:
    *   `model_name` (str, optional): The name or path of the language model to use for the chat.
*   **Logic**:
    *   **Model Auto-detection**: If `model_name` is *not* provided, the command attempts to load the `last_used_model` from the user's configuration file (`~/.yaie_config.json`).
    *   **Default Model**: If no `model_name` is provided and no `last_used_model` is found in the config, it defaults to `"microsoft/DialoGPT-small"` for testing purposes.
    *   Informative messages are printed to the console indicating which model is being used.
*   **Delegation**: This command internally calls the `run_chat()` function (from `chat.py`) with the resolved `model_name`.

## Main Execution Block

The standard Python `if __name__ == "__main__":` block ensures that the `main()` click group is invoked when the script is run directly, making the CLI commands available.

```python
if __name__ == "__main__":
    main()
```
This is how the command-line arguments are parsed and dispatched to the respective subcommands (`serve` or `chat`).