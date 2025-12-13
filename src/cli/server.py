import uvicorn
from ..server.api import create_app


def run_server(model_name: str, host: str = 'localhost', port: int = 8000):
    """
    Run the inference server with OpenAI-compatible API

    Args:
        model_name: Name of the model to load
        host: Host to bind to
        port: Port to bind to
    """
    app = create_app(model_name)
    uvicorn.run(app, host=host, port=port)
