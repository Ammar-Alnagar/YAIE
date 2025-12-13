import click

from .chat import run_chat
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
@click.argument("model_name")
def chat(model_name):
    """Start an interactive chat session"""
    run_chat(model_name)


if __name__ == "__main__":
    main()
