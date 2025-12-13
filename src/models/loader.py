from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from huggingface_hub import snapshot_download
import os
import torch


class ModelLoader:
    """
    Handles loading of models with caching and fallback to HuggingFace Hub
    """

    def __init__(self, model_name: str):
        """
        Initialize the model loader

        Args:
            model_name: Name of the model (e.g., "microsoft/DialoGPT-medium", or local path)
        """
        self.model_name = model_name
        self.cache_dir = os.path.expanduser("~/.cache/huggingface")

    def load_model(self) -> PreTrainedModel:
        """
        Load a model from cache or download it if not present

        Returns:
            Loaded PyTorch model
        """
        # Check if it's a local path first
        if os.path.isdir(self.model_name):
            model_path = self.model_name
        else:
            # Try to find in cache first
            model_path = self._get_model_path_from_cache()
            if not model_path:
                # Download the model
                model_path = self._download_model()

        # Load the model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # Use half precision for efficiency
            device_map="auto"  # Automatically distribute across available devices
        )

        return model

    def load_tokenizer(self) -> PreTrainedTokenizer:
        """
        Load a tokenizer from cache or download it if not present

        Returns:
            Loaded tokenizer
        """
        # Check if it's a local path first
        if os.path.isdir(self.model_name):
            tokenizer_path = self.model_name
        else:
            # Try to find in cache first
            model_path = self._get_model_path_from_cache()
            if not model_path:
                # Download the model (tokenizer is part of it)
                model_path = self._download_model()

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Ensure tokenizer has a pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def _get_model_path_from_cache(self) -> Optional[str]:
        """
        Check if model exists in HuggingFace cache

        Returns:
            Path to model if found in cache, None otherwise
        """
        # Standard HuggingFace cache structure
        repo_path = os.path.join(self.cache_dir, "hub", f"models--{self.model_name.replace('/', '--')}")

        if os.path.exists(repo_path):
            # Look for the snapshots directory which contains the actual model
            snapshots_dir = os.path.join(repo_path, "snapshots")
            if os.path.exists(snapshots_dir):
                # Get the most recent snapshot (usually there's only one)
                snapshots = [f for f in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, f))]
                if snapshots:
                    # Return the first (most recent) snapshot
                    return os.path.join(snapshots_dir, snapshots[0])

        return None

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
