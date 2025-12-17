import click

from .chat import run_chat, load_last_used_model
from .server import run_server


@click.group()
def main():
    """YAIE: Educational LLM Inference Engine"""
    pass


@main.command()
@click.argument("model_name")
@click.option("--host", default="localhost", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
def serve(model_name, host, port):
    """Start the inference server"""
    run_server(model_name, host, port)


@main.command()
@click.argument("model_name", required=False)
def chat(model_name):
    """Start an interactive chat session"""
    if model_name is None:
        # If no model name is provided, try to get the last used model from config
        last_used_model = load_last_used_model()
        if last_used_model:
            model_name = last_used_model
            print(f"Using last used model: {model_name}")
        else:
            # Default to a common small model for testing purposes
            model_name = "microsoft/DialoGPT-small"  # Default model
            print(f"No model specified and no previous model found, using default: {model_name}")

    run_chat(model_name)


if __name__ == "__main__":
    main()
