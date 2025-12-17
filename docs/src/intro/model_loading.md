# Model Loading

YAIE supports loading models from HuggingFace Hub with automatic caching and local model support.

## ModelLoader Class

The `ModelLoader` class in `src/models/loader.py` handles all model and tokenizer loading operations.

### Initialization

```python
from src.models.loader import ModelLoader

# Load from HuggingFace Hub
loader = ModelLoader("microsoft/DialoGPT-medium")

# Load from local path
loader = ModelLoader("/path/to/local/model")
```

### Loading Models

```python
# Load the model
model = loader.load_model()

# Load the tokenizer
tokenizer = loader.load_tokenizer()
```

## Supported Model Sources

### HuggingFace Hub Models

YAIE can load any compatible model from HuggingFace Hub:

```python
# Popular conversational models
loader = ModelLoader("microsoft/DialoGPT-medium")
loader = ModelLoader("microsoft/DialoGPT-large")

# Code generation models
loader = ModelLoader("Salesforce/codegen-350M-mono")

# General purpose models
loader = ModelLoader("gpt2")
loader = ModelLoader("gpt2-medium")
```

### Local Models

You can also load models from local directories:

```python
# Load from local path
loader = ModelLoader("./models/my-custom-model")
```

## Caching Behavior

### Automatic Caching

Models are automatically cached in the standard HuggingFace cache directory:

- **Linux/macOS**: `~/.cache/huggingface/`
- **Windows**: `C:\Users\<username>\.cache\huggingface\`

### Cache Structure

```
~/.cache/huggingface/
├── hub/
│   └── models--microsoft--DialoGPT-medium/
│       ├── blobs/
│       ├── refs/
│       └── snapshots/
│           └── abc123.../
│               ├── config.json
│               ├── pytorch_model.bin
│               └── tokenizer.json
```

### Cache Management

The loader automatically:

1. Checks for existing models in cache
2. Downloads missing models from HuggingFace Hub
3. Uses cached models for subsequent loads

## Model Configuration

### Data Types

Models are loaded with optimized data types:

```python
# Models are loaded with float16 by default for efficiency
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # Half precision
    device_map="auto"          # Automatic device placement
)
```

### Device Placement

- **Single GPU**: Model is loaded directly to GPU
- **Multi-GPU**: Automatically distributed across available GPUs
- **CPU**: Falls back to CPU if no GPU available

## Tokenizer Configuration

### Automatic Pad Token

The loader ensures tokenizers have a pad token:

```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

This is important for batch processing where sequences need to be padded to the same length.

## Memory Optimization

### Lazy Loading

Models are loaded on-demand, not at import time:

```python
# Model is not loaded here
loader = ModelLoader("gpt2")

# Model is loaded here when requested
model = loader.load_model()
```

### Memory Mapping

Large models use memory mapping to reduce RAM usage during loading.

## Error Handling

### Network Issues

If downloading fails, the loader will retry and provide clear error messages.

### Incompatible Models

Models must be compatible with `AutoModelForCausalLM`. Incompatible models will raise clear errors.

### Disk Space

Large models require significant disk space. The loader shows download progress and estimated sizes.

## Performance Tips

### Pre-download Models

For production deployments, pre-download models:

```bash
# This will cache the model
python -c "from src.models.loader import ModelLoader; loader = ModelLoader('microsoft/DialoGPT-medium'); loader.load_model()"
```

### Cache Location

You can customize the cache location by setting the `HF_HOME` environment variable:

```bash
export HF_HOME=/path/to/custom/cache
```

### Model Selection

Choose appropriate model sizes for your hardware:

- **Small models** (< 1GB): `gpt2`, `DialoGPT-small`
- **Medium models** (1-5GB): `gpt2-medium`, `DialoGPT-medium`
- **Large models** (> 5GB): `gpt2-large`, `DialoGPT-large`

## Troubleshooting

### Common Issues

**"Model not found" errors:**
- Check model name spelling
- Verify model exists on HuggingFace Hub
- Ensure internet connection for downloads

**Out of memory errors:**
- Try smaller models
- Reduce batch sizes in configuration
- Use CPU-only mode if GPU memory is insufficient

**Tokenizer issues:**
- Some models may require special token handling
- Check the model's documentation on HuggingFace Hub</content>
<parameter name="filePath">docs/src/intro/model_loading.md