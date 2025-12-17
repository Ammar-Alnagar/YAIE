# Model Loader (`models/loader.py`)

The ModelLoader class handles loading of models with caching and fallback to HuggingFace Hub.

## Overview

The ModelLoader provides a unified interface for loading models from various sources:

- Local directories
- HuggingFace Hub (with caching)
- Automatic model downloading

## Key Features

1. **Caching**: Uses HuggingFace cache to avoid repeated downloads
2. **Fallback**: Automatically downloads models if not found locally
3. **Flexibility**: Supports both local paths and HuggingFace model names
4. **Efficiency**: Uses half-precision (float16) and automatic device mapping

## Implementation

### Initialization

```python
class ModelLoader:
    def __init__(self, model_name: str):
        """
        Initialize the model loader
        
        Args:
            model_name: Name of the model (e.g., "microsoft/DialoGPT-medium", or local path)
        """
        self.model_name = model_name
        self.cache_dir = os.path.expanduser("~/.cache/huggingface")
```

### Model Loading

```python
def load_model(self) -> PreTrainedModel:
    """
    Load a model from cache or download it if not present
    
    Returns:
        Loaded PyTorch model
    """
    # 1. Check if it's a local path
    if os.path.isdir(self.model_name):
        model_path = self.model_name
    else:
        # 2. Try to find in cache
        model_path = self._get_model_path_from_cache()
        if not model_path:
            # 3. Download the model
            model_path = self._download_model()
    
    # 4. Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # Use half precision for efficiency
        device_map="auto"  # Automatically distribute across available devices
    )
    
    return model
```

### Tokenizer Loading

```python
def load_tokenizer(self) -> PreTrainedTokenizer:
    """
    Load a tokenizer from cache or download it if not present
    
    Returns:
        Loaded tokenizer
    """
    # Similar path resolution logic as model loading
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Ensure tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer
```

### Cache Management

```python
def _get_model_path_from_cache(self) -> Optional[str]:
    """
    Check if model exists in HuggingFace cache
    
    Returns:
        Path to model if found in cache, None otherwise
    """
    # Standard HuggingFace cache structure
    repo_path = os.path.join(
        self.cache_dir, 
        "hub", 
        f"models--{self.model_name.replace('/', '--')}"
    )
    
    if os.path.exists(repo_path):
        # Look for the snapshots directory
        snapshots_dir = os.path.join(repo_path, "snapshots")
        if os.path.exists(snapshots_dir):
            # Get the most recent snapshot
            snapshots = [f for f in os.listdir(snapshots_dir) 
                        if os.path.isdir(os.path.join(snapshots_dir, f))]
            if snapshots:
                return os.path.join(snapshots_dir, snapshots[0])
    
    return None
```

### Model Download

```python
def _download_model(self) -> str:
    """
    Download model from HuggingFace Hub to cache
    
    Returns:
        Path to downloaded model
    """
    print(f"Downloading {self.model_name} from HuggingFace Hub...")
    
    # Download the model to cache
    model_path = snapshot_download(
        repo_id=self.model_name,
        cache_dir=self.cache_dir
    )
    
    print(f"Downloaded {self.model_name} to {model_path}")
    return model_path
```

## Usage Examples

### Loading a Model

```python
from models.loader import ModelLoader

# Load a HuggingFace model
loader = ModelLoader("microsoft/DialoGPT-medium")
model = loader.load_model()
tokenizer = loader.load_tokenizer()

# Load a local model
loader = ModelLoader("./path/to/local/model")
model = loader.load_model()
tokenizer = loader.load_tokenizer()
```

### Cache Behavior

```python
# First run - downloads model
loader1 = ModelLoader("gpt2")
model1 = loader1.load_model()  # Downloads from HuggingFace

# Second run - uses cached model
loader2 = ModelLoader("gpt2")
model2 = loader2.load_model()  # Loads from cache
```

## Implementation Details

### Cache Structure

```
~/.cache/huggingface/
└── hub/
    └── models--gpt2/
        └── snapshots/
            └── abc123.../
                ├── config.json
                ├── pytorch_model.bin
                ├── tokenizer.json
                └── ...
```

### Performance Optimizations

1. **Half Precision**: Uses `torch.float16` for memory efficiency
2. **Automatic Device Mapping**: Distributes model across available devices
3. **Caching**: Avoids repeated downloads of the same model
4. **Tokenizer Configuration**: Ensures proper padding token configuration

### Error Handling

The loader handles various scenarios:

- Missing cache directories
- Invalid model names
- Network issues during download
- Corrupted cache entries

## Educational Focus

The ModelLoader demonstrates:

1. **HuggingFace Integration**: How to work with HuggingFace Hub
2. **Caching Strategies**: Efficient model caching patterns
3. **Resource Management**: Memory-efficient model loading
4. **Error Handling**: Robust handling of various edge cases

## Future Enhancements

The current implementation provides a foundation for:

- Model quantization support
- Multiple model versions
- Cache management utilities
- Model validation and verification
- Advanced device placement strategies

The ModelLoader is a critical component that handles the foundational task of loading models efficiently, making it accessible for educational exploration of more advanced inference concepts.