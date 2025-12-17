"""
KV-Cache management implementation
Implements the paged KV-cache system for efficient memory utilization
"""

from typing import List, Optional, Tuple

import torch


class KVCacheBlock:
    """Represents a single block in the paged KV-cache"""

    def __init__(
        self,
        block_id: int,
        size: int,
        num_heads: int,
        head_dim: int,
        dtype=torch.float16,
    ):
        self.block_id = block_id
        self.size = size  # Number of tokens this block can hold
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype

        # Initialize GPU memory block for keys and values
        # This should allocate GPU memory for storing key and value tensors
        self.keys = None  # [size, num_heads, head_dim]
        self.values = None  # [size, num_heads, head_dim]

    def allocate(self):
        """Allocate GPU memory for the block"""
        # Use torch.cuda for GPU allocation if available, else CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.keys = torch.zeros(
            self.size, self.num_heads, self.head_dim, dtype=self.dtype, device=device
        )
        self.values = torch.zeros(
            self.size, self.num_heads, self.head_dim, dtype=self.dtype, device=device
        )


class KVCacheManager:
    """Manages the paged KV-cache system"""

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        dtype=torch.float16,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype

        # Initialize the blocks
        self.blocks: List[KVCacheBlock] = []
        for i in range(num_blocks):
            block = KVCacheBlock(i, block_size, num_heads, head_dim, dtype)
            self.blocks.append(block)

        # Track free blocks
        self.free_block_list: List[int] = list(range(num_blocks))

        # Track which blocks are assigned to which request
        self.block_tables: dict = {}  # request_id -> [block_ids]

    def allocate_blocks(self, request_id: str, num_tokens: int) -> List[int]:
        """
        Allocate blocks for a request

        Args:
            request_id: ID of the request
            num_tokens: Number of tokens needed

        Returns:
            List of allocated block IDs
        """
        # Calculate how many blocks are needed and allocate from free_blocks
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        if len(self.free_block_list) < num_blocks_needed:
            raise RuntimeError(
                f"Not enough free blocks. Need {num_blocks_needed}, have {len(self.free_block_list)}"
            )

        # Allocate blocks from the free list
        allocated_block_ids = []
        for _ in range(num_blocks_needed):
            block_id = self.free_block_list.pop(0)
            allocated_block_ids.append(block_id)
            # Actually allocate the memory for the block
            self.blocks[block_id].allocate()

        # Update block_tables to track which blocks belong to which request
        self.block_tables[request_id] = allocated_block_ids
        return allocated_block_ids

    def free_blocks(self, request_id: str):
        """
        Free blocks associated with a request

        Args:
            request_id: ID of the request to free
        """
        # Return blocks to free_blocks list and update block_tables
        if request_id in self.block_tables:
            block_ids = self.block_tables[request_id]
            self.free_block_list.extend(block_ids)
            self.free_block_list.sort()  # Keep the list sorted for efficiency
            del self.block_tables[request_id]

            # Optionally zero out the tensors to free memory
            for block_id in block_ids:
                self.blocks[block_id].keys = None
                self.blocks[block_id].values = None

    def copy_blocks(self, src_block_ids: List[int], dst_block_ids: List[int]):
        """
        Copy key-value pairs from source blocks to destination blocks

        Args:
            src_block_ids: List of source block IDs
            dst_block_ids: List of destination block IDs
        """
        if len(src_block_ids) != len(dst_block_ids):
            raise ValueError("Source and destination block lists must have the same length")

        for src_id, dst_id in zip(src_block_ids, dst_block_ids):
            src_block = self.blocks[src_id]
            dst_block = self.blocks[dst_id]

            # Allocate destination if needed
            if dst_block.keys is None or dst_block.values is None:
                dst_block.allocate()

            # Copy the data
            with torch.no_grad():
                dst_block.keys.copy_(src_block.keys)
                dst_block.values.copy_(src_block.values)

    def get_kv_tensors(
        self, block_ids: List[int], seq_lens: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get key and value tensors for a list of block IDs

        Args:
            block_ids: List of block IDs
            seq_lens: Sequence lengths for each request

        Returns:
            Tuple of key and value tensors
        """
        if not block_ids:
            return torch.empty(0), torch.empty(0)

        # Determine the number of requests
        num_requests = len(seq_lens)

        # Calculate the max sequence length
        max_len = max(seq_lens)

        # Calculate dimensions
        total_tokens = sum(seq_lens)

        # Create output tensors
        key_shape = (1, total_tokens, self.num_heads, self.head_dim)
        value_shape = (1, total_tokens, self.num_heads, self.head_dim)

        # Create paged KV cache tensors
        # Create a contiguous tensor that will be filled from the blocks
        keys = torch.zeros(*key_shape, dtype=self.dtype, device='cuda')
        values = torch.zeros(*value_shape, dtype=self.dtype, device='cuda')

        # Fill the tensors from the blocks
        current_token_idx = 0
        for block_id in block_ids:
            block = self.blocks[block_id]
            if block.keys is not None and block.values is not None:
                # Copy from the block to the output tensor
                tokens_in_block = min(self.block_size, keys.shape[1] - current_token_idx)
                if tokens_in_block > 0:
                    keys[0, current_token_idx:current_token_idx+tokens_in_block] = block.keys[:tokens_in_block]
                    values[0, current_token_idx:current_token_idx+tokens_in_block] = block.values[:tokens_in_block]
                    current_token_idx += tokens_in_block

                    if current_token_idx >= keys.shape[1]:  # We've filled all needed tokens
                        break

        return keys, values
