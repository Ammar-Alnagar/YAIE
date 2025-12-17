"""
SGLang-style inference engine implementation
Implements SGLang's approach with radial attention, prefix sharing, and continuous batching
"""

import time
import uuid
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

from core.sglang_scheduler import SGLangScheduler
from kernels.kv_cache import KVCacheManager
from kernels.radix_attention import RadixAttentionWithPagedKVCache
from models.loader import ModelLoader
from kernels.sampling import SamplingKernel


class InferenceEngine:
    """
    Main inference engine for LLMs with SGLang-style continuous batching,
    radial attention and prefix sharing
    """

    def __init__(self, model_name: str):
        """
        Initialize the SGLang-style inference engine

        Args:
            model_name: Name of the model to load (from HuggingFace or local path)
        """
        self.model_name = model_name

        # Load the model and tokenizer
        self.tokenizer: PreTrainedTokenizer = self._load_tokenizer()
        self.model = self._load_model()

        # Initialize SGLang-style scheduler with prefix grouping
        self.scheduler = SGLangScheduler(
            max_batch_size=8, max_prefill_batch_size=16, max_decode_batch_size=256
        )

        # Initialize SGLang-style radial attention with paged KV-cache
        self.radix_attention = RadixAttentionWithPagedKVCache(
            num_layers=self.model.config.num_hidden_layers,
            num_heads=self.model.config.num_attention_heads,
            head_dim=self.model.config.hidden_size
            // self.model.config.num_attention_heads,
        )

        # Initialize KV-cache manager for SGLang-style memory management
        self.kv_cache_manager = KVCacheManager(
            num_blocks=2000,  # Configurable based on available memory
            block_size=16,  # Standard block size
            num_heads=self.model.config.num_attention_heads,
            head_dim=self.model.config.hidden_size
            // self.model.config.num_attention_heads,
            dtype=torch.float16,
        )

        # Connect scheduler to memory manager for SGLang optimization
        self.scheduler.connect_memory_manager(self.kv_cache_manager)

        # TODO: Initialize other SGLang-specific engine components:
        # 1. Radix tree for prefix matching and sharing
        # 2. Multi-step attention processors (prefill/decode)
        # 3. Advanced memory pooling strategies
        # 4. Request preemption and re-scheduling mechanisms
        # 5. Chunked prefill for very long prompts

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """
        Load the tokenizer from HuggingFace or local cache
        """
        loader = ModelLoader(self.model_name)
        return loader.load_tokenizer()

    def _load_model(self):
        """
        Load the model from HuggingFace or local cache
        """
        loader = ModelLoader(self.model_name)
        return loader.load_model()

    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate responses for a batch of prompts using SGLang-style processing

        Args:
            prompts: List of input prompts
            **kwargs: Additional generation parameters

        Returns:
            List of generated responses
        """
        # 1. Add requests to scheduler
        request_ids = []
        for prompt in prompts:
            req_id = self.scheduler.add_request(prompt, **kwargs)
            request_ids.append(req_id)

        # 2. Process requests in the main generation loop
        # This simulates the SGLang-style continuous batching loop
        responses = self._run_generation_loop(request_ids)

        return responses

    def _run_generation_loop(self, request_ids: List[str]) -> List[str]:
        """
        Main generation loop - simplified version that directly uses the model
        """
        responses = []

        for req_id in request_ids:
            # Get the request from scheduler
            request = self.scheduler.request_lookup.get(req_id)
            if not request:
                responses.append("Request not found")
                continue

            # Tokenize the prompt
            input_ids = self.tokenizer(request.prompt, return_tensors="pt").input_ids.to(self.model.device)

            # Generate based on max_tokens
            max_new_tokens = request.max_tokens
            current_ids = input_ids

            generated_tokens = []
            for i in range(max_new_tokens):
                # Forward pass through the model
                with torch.no_grad():
                    outputs = self.model(current_ids)
                    next_token_logits = outputs.logits[:, -1, :]  # Get last token logits

                # Use our sampling kernel
                sampling_kernel = SamplingKernel()
                next_token_id = sampling_kernel.sample(
                    next_token_logits,
                    temperature=request.temperature,
                    top_p=request.top_p
                )

                # Add to generated sequence
                generated_tokens.append(next_token_id.item())

                # next_token_id from sampling is typically a 1D tensor with shape [1],
                # but current_ids has shape [batch, seq_len] - typically [1, seq_len]
                # For concatenation along sequence dimension (dim=-1), both must have same number of dimensions
                if next_token_id.dim() == 1:
                    # If next_token_id is 1D, reshape it to match [batch, 1] format
                    next_token_tensor = next_token_id.unsqueeze(0)  # Shape becomes [1, 1]
                else:
                    # If already correct dimension, keep as is
                    next_token_tensor = next_token_id

                # Now concatenate along the sequence dimension
                current_ids = torch.cat([current_ids, next_token_tensor], dim=-1)

                # Check for EOS token
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

            # Decode the full sequence
            # Move generated_tokens tensor to the same device as input_ids
            generated_tokens_tensor = torch.tensor(generated_tokens, device=input_ids.device)
            full_sequence = torch.cat([input_ids.squeeze(0), generated_tokens_tensor], dim=0)

            # Extract just the generated part (not the original prompt)
            prompt_length = len(self.tokenizer.encode(request.prompt))
            generated_part = full_sequence[prompt_length:]
            if len(generated_part) > 0:
                response_text = self.tokenizer.decode(generated_part, skip_special_tokens=True)
            else:
                response_text = ""

            responses.append(response_text)

        return responses

    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Generate a chat completion response from a list of messages.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional generation parameters (temperature, top_p, max_tokens, etc.)

        Returns:
            Dictionary containing the chat completion response in OpenAI format
        """
        import time

        # Format messages into a single prompt using chat template if available
        # Otherwise, use a simple format
        if hasattr(self.tokenizer, 'apply_chat_template'):
            # Use the tokenizer's chat template if available
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback to simple formatting
            formatted_prompt = ""
            for message in messages:
                formatted_prompt += f"{message['role'].capitalize()}: {message['content']}\n"
            formatted_prompt += "\nAssistant:"

        # Generate response using the existing generate method
        responses = self.generate([formatted_prompt], **kwargs)
        generated_text = responses[0] if responses else ""

        # Create OpenAI-compatible response format
        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        response_dict = {
            "id": response_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text
                    },
                    "finish_reason": "stop"  # Simplified assumption
                }
            ],
            "usage": {
                "prompt_tokens": len(self.tokenizer.encode(formatted_prompt)),
                "completion_tokens": len(self.tokenizer.encode(generated_text)),
                "total_tokens": len(self.tokenizer.encode(formatted_prompt)) + len(self.tokenizer.encode(generated_text))
            }
        }

        return response_dict

    def chat_completion_stream(self, messages: List[Dict[str, str]], **kwargs):
        """
        Generate a streaming chat completion response from a list of messages.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional generation parameters (temperature, top_p, max_tokens, etc.)

        Yields:
            Dictionary containing the streaming chat completion chunks in OpenAI format
        """
        import time
        import uuid

        # Format messages into a single prompt
        if hasattr(self.tokenizer, 'apply_chat_template'):
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = ""
            for message in messages:
                formatted_prompt += f"{message['role'].capitalize()}: {message['content']}\n"
            formatted_prompt += "\nAssistant:"

        # Tokenize the prompt
        input_ids = self.tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(self.model.device)

        # Use generation parameters
        max_new_tokens = kwargs.get('max_tokens', 512)
        temperature = kwargs.get('temperature', 1.0)
        top_p = kwargs.get('top_p', 1.0)

        current_ids = input_ids
        generated_tokens = []

        # Generate token by token
        for i in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(current_ids)
                next_token_logits = outputs.logits[:, -1, :]  # Get last token logits

            # Use our sampling kernel
            sampling_kernel = SamplingKernel()
            next_token_id = sampling_kernel.sample(
                next_token_logits,
                temperature=temperature,
                top_p=top_p
            )

            # Check for EOS token
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

            # Add to generated sequence
            generated_tokens.append(next_token_id.item())

            # Decode and yield the token
            new_token_text = self.tokenizer.decode([next_token_id.item()], skip_special_tokens=True)

            # Create streaming chunk
            chunk = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": self.model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": new_token_text
                        },
                        "finish_reason": None
                    }
                ]
            }

            yield chunk

            # Update current_ids for next iteration
            if next_token_id.dim() == 1:
                next_token_tensor = next_token_id.unsqueeze(0)  # Shape becomes [1, 1]
            else:
                next_token_tensor = next_token_id
            current_ids = torch.cat([current_ids, next_token_tensor], dim=-1)

        # Send final chunk with finish_reason
        final_chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }
            ]
        }

        yield final_chunk
