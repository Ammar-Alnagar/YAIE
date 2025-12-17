# CLI Usage: Interactive and Server Modes

## Overview

Mini-YAIE provides a comprehensive command-line interface (CLI) that serves as the primary entry point for users. The CLI supports both interactive chat mode and server mode, making it suitable for both direct interaction and production deployment scenarios.

## CLI Architecture

### Entry Point Structure

The CLI is organized around different command verbs:

```
yaie <command> [options] [arguments]

Commands:
- serve: Start an OpenAI-compatible API server
- chat: Start an interactive chat session
```

### Core Components

1. **Argument Parsing**: Uses argparse for command-line option handling
2. **Model Integration**: Connects CLI commands to the inference engine
3. **Interactive Interface**: Provides user-friendly chat experience
4. **Server Integration**: Launches API server with proper configuration

## Server Mode

### Basic Server Usage

Start the API server with a specific model:

```bash
yaie serve microsoft/DialoGPT-medium --host localhost --port 8000
```

### Server Options

#### Model Selection
```bash
--model MODEL_NAME          # Specify the model to use (required)
```

#### Network Configuration
```bash
--host HOST                 # Server host (default: localhost)
--port PORT                 # Server port (default: 8000)
--workers WORKERS           # Number of server workers
```

#### Performance Options
```bash
--max-batch-size N          # Maximum batch size
--max-prefill-batch-size N  # Maximum prefill batch size
--max-decode-batch-size N   # Maximum decode batch size
--num-blocks N              # Number of KV-cache blocks
--block-size N              # Size of each cache block
```

### Server Startup Process

1. **Model Loading**: Download and load model from HuggingFace if not cached
2. **Engine Initialization**: Create inference engine with specified parameters
3. **API Server Creation**: Initialize FastAPI application with engine
4. **Server Launch**: Start the web server on specified host/port

### Example Server Commands

#### Basic Server
```bash
yaie serve microsoft/DialoGPT-medium
```

#### Production Server
```bash
yaie serve microsoft/DialoGPT-medium --host 0.0.0.0 --port 8000 --max-batch-size 16
```

#### Resource-Constrained Server
```bash
yaie serve microsoft/DialoGPT-medium --num-blocks 1000 --max-batch-size 4
```

## Chat Mode

### Basic Chat Usage

Start an interactive chat session:

```bash
yaie chat microsoft/DialoGPT-medium
```

### Chat Options

#### Generation Parameters
```bash
--temperature TEMP          # Sampling temperature (default: 1.0)
--top-p TOP_P               # Nucleus sampling threshold (default: 1.0)
--max-tokens N              # Maximum tokens to generate (default: 512)
--stream                    # Stream responses in real-time (default: true)
```

#### Model Configuration
```bash
--model MODEL_NAME          # Specify the model to use (required)
```

### Interactive Chat Experience

#### Session Flow
1. **Model Loading**: Model is loaded if not cached
2. **Chat Initialization**: Engine and tokenizer are set up
3. **Conversation Loop**: User inputs are processed and responses generated
4. **Session Termination**: Exit with Ctrl+C or quit command

#### User Interaction

The chat interface provides a conversational experience:

```
$ yaie chat microsoft/DialoGPT-medium
Model loaded successfully!
Starting chat session (press Ctrl+C to exit)...

User: Hello, how are you?
AI: I'm doing well, thank you for asking!

User: What can you help me with?
AI: I can have conversations, answer questions, and assist with various tasks.
```

### Example Chat Commands

#### Basic Chat
```bash
yaie chat microsoft/DialoGPT-medium
```

#### Creative Chat
```bash
yaie chat microsoft/DialoGPT-medium --temperature 1.2 --top-p 0.9
```

#### Focused Chat
```bash
yaie chat microsoft/DialoGPT-medium --temperature 0.7 --max-tokens 128
```

## Model Selection

### Supported Model Formats

The CLI supports any HuggingFace-compatible model:

#### Pre-trained Models
```bash
yaie serve microsoft/DialoGPT-medium
yaie serve gpt2
yaie serve facebook/opt-1.3b
```

#### Local Models
```bash
yaie serve /path/to/local/model
yaie serve ./models/my-custom-model
```

### Model Caching

Models are automatically downloaded and cached:

- First run: Download from HuggingFace Hub
- Subsequent runs: Use cached version
- Cache location: Standard HuggingFace cache directory

## Performance Tuning

### Memory Configuration

Adjust memory settings based on available GPU memory:

```bash
# For 24GB+ GPU
yaie serve model --num-blocks 4000 --max-batch-size 32

# For 8-16GB GPU  
yaie serve model --num-blocks 1500 --max-batch-size 8

# For 4-8GB GPU
yaie serve model --num-blocks 800 --max-batch-size 4
```

### Batch Size Optimization

Tune batch sizes for optimal throughput:

```bash
# High throughput (more memory)
yaie serve model --max-batch-size 32 --max-prefill-batch-size 64

# Memory efficient (lower batch sizes)
yaie serve model --max-batch-size 4 --max-prefill-batch-size 8
```

## Error Handling and Troubleshooting

### Common Errors

#### Model Loading Errors
```bash
# If model name is invalid
Error: Model not found on HuggingFace Hub

# If network is unavailable during first load
Error: Failed to download model
```

#### Memory Errors
```bash
# If not enough GPU memory
CUDA out of memory error

# If KV-cache is too large
Memory allocation failed
```

### Debugging Options

#### Verbose Output
```bash
yaie serve model --verbose  # Show detailed startup information
```

#### Configuration Validation
```bash
yaie serve model --debug    # Enable debugging features
```

## Advanced CLI Features

### Configuration Files

The CLI supports configuration files for complex setups:

```bash
yaie serve --config config.yaml model
```

### Environment Variables

Several environment variables can customize behavior:

```bash
# Set default host
export YAIE_HOST=0.0.0.0

# Set default port  
export YAIE_PORT=9000

# Set memory limits
export YAIE_MAX_BLOCKS=2000
```

### Logging Configuration

Control logging verbosity and output:

```bash
# Enable detailed logging
yaie serve model --log-level DEBUG

# Log to file
yaie serve model --log-file server.log
```

## Integration with SGLang Features

### Batching Optimization

The CLI exposes SGLang batching parameters:

```bash
yaie serve model \
  --max-prefill-batch-size 16 \
  --max-decode-batch-size 256
```

### Prefix Sharing Control

Parameters that affect prefix sharing efficiency:

```bash
yaie serve model \
  --max-seq-len 2048 \
  --block-size 16
```

## Production Deployment

### Server Management

#### Process Control
```bash
# Start server in background
nohup yaie serve model > server.log 2>&1 &

# Kill server process
pkill -f "yaie serve"
```

#### Process Monitoring
```bash
# Monitor server with systemd
systemctl start yaie-server

# Monitor with supervisor
supervisorctl start yaie-server
```

### Health Checks

The server provides health status:

```bash
# Check server status
curl http://localhost:8000/health

# Integrate with monitoring tools
# Health check interval and thresholds
```

## Examples and Use Cases

### Development Usage
```bash
# Quick test with small model
yaie chat gpt2

# Interactive development with verbose output
yaie serve gpt2 --port 8000 --verbose
```

### Production Usage
```bash
# High-performance server for production
yaie serve microsoft/DialoGPT-medium \
  --host 0.0.0.0 \
  --port 8000 \
  --max-batch-size 16 \
  --num-blocks 2000

# Low-resource server for edge deployment
yaie serve gpt2 \
  --max-batch-size 2 \
  --num-blocks 500
```

### Testing and Evaluation
```bash
# Test with various parameters
yaie chat model --temperature 0.5 --top-p 0.9

# Evaluate different models
yaie serve model1 --port 8001 &
yaie serve model2 --port 8002 &
```

The CLI provides a comprehensive interface to access all of Mini-YAIE's features, from simple interactive chat to high-performance API serving with SGLang-style optimizations.