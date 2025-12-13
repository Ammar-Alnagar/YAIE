"""
SGLang-style scheduler implementation
Implements SGLang's advanced request scheduling with prefix grouping and multi-step processing
"""

import hashlib
import heapq
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

try:
    from kernels.sampling import SamplingKernel
except ImportError:
    from src.kernels.sampling import SamplingKernel



class RequestStatus(Enum):
    PENDING = "pending"
    SCHEDULED_PREFILL = "scheduled_prefill"
    RUNNING_PREFILL = "running_prefill"
    SCHEDULED_DECODE = "scheduled_decode"
    RUNNING_DECODE = "running_decode"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class Request:
    """Represents a single inference request"""

    id: str
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float
    created_at: float
    status: RequestStatus
    input_ids: Optional[List[int]] = None
    output_ids: Optional[List[int]] = None
    current_position: int = 0  # Position in generation
    prefix_hash: Optional[str] = None  # For SGLang-style prefix grouping
    request_group: Optional[str] = None  # Group ID for shared prefixes


class SGLangScheduler:
    """
    SGLang-style scheduler with advanced features:
    - Prefix-based request grouping for computation sharing
    - Separate prefill and decode scheduling
    - Memory-aware batch sizing
    - Continuous batching optimization
    """

    def __init__(
        self,
        max_batch_size: int = 8,
        max_seq_len: int = 2048,
        max_prefill_batch_size: int = 16,
        max_decode_batch_size: int = 256,
    ):
        """
        Initialize the SGLang-style scheduler

        Args:
            max_batch_size: Overall maximum batch size
            max_seq_len: Maximum sequence length
            max_prefill_batch_size: Maximum requests in prefill batch
            max_decode_batch_size: Maximum requests in decode batch
        """
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.max_prefill_batch_size = max_prefill_batch_size
        self.max_decode_batch_size = max_decode_batch_size

        # Request queues by state
        self.pending_requests: List[Request] = []
        self.prefill_requests: List[Request] = []  # Ready for prefill
        self.running_prefill: List[Request] = []  # Currently prefilling
        self.decode_requests: List[Request] = []  # Ready for decode
        self.running_decode: List[Request] = []  # Currently decoding
        self.completed_requests: List[Request] = []

        # SGLang-specific structures
        self.prefix_groups: Dict[str, List[Request]] = defaultdict(
            list
        )  # Group requests by prefix
        self.request_lookup: Dict[str, Request] = {}  # Fast request lookup
        self.memory_manager = None  # Will be connected to KV cache manager
        self.sampling_kernel = SamplingKernel()

    def add_request(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> str:
        """
        Add a new request to the scheduler with SGLang-style prefix analysis

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Request ID
        """
        req_id = str(uuid.uuid4())

        # TODO: Calculate prefix hash based on prompt tokens for grouping
        # This enables SGLang's prefix sharing optimization
        prefix_hash = self._calculate_prefix_hash(prompt)

        request = Request(
            id=req_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            created_at=time.time(),
            status=RequestStatus.PENDING,
            prefix_hash=prefix_hash,
        )

        self.pending_requests.append(request)
        self.request_lookup[req_id] = request

        # Add to appropriate prefix group for SGLang optimization
        if prefix_hash:
            self.prefix_groups[prefix_hash].append(request)
            request.request_group = prefix_hash

        return req_id

    def _calculate_prefix_hash(self, prompt: str) -> Optional[str]:
        """
        Calculate a hash representing the common prefix for SGLang grouping
        For this educational implementation, we use the hash of the prompt string.
        """
        if not prompt:
            return None
        # In a real implementation we would hash the token IDs of the prefix
        # Here we just hash the prompt string for simplicity in the non-kernel code
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def schedule_step(self) -> Tuple[List[Request], List[Request]]:
        """
        Perform one scheduling step - organize requests into prefill and decode batches
        """
        # 1. Prioritize Decode:
        # Existing running decode requests + scheduled decode requests
        # We need to respect max_batch_size and max_decode_batch_size
        
        # First, check our overall capacity
        current_active = len(self.running_prefill) + len(self.running_decode)
        # Note: We are about to schedule new ones, so we might not need to count 'running' 
        # if this step defines what IS running. 
        # Usually, schedule_step determines the NEXT batch to run.
        
        # SGLang strategy:
        # Prio 1: Continue decoding current requests (minimize latency)
        # Prio 2: Start new prefill requests (fill gaps)
        
        decode_batch = []
        prefill_batch = []
        
        # Gather all candidates for decoding
        # In this simplified model, 'running_decode' and 'scheduled_decode' are candidates
        # We move them to a generic 'decode_queue' for selection
        decode_candidates = self.running_decode + self.decode_requests
        
        # Reset running lists as we are rebuilding them for this step
        # (In a real system we might keep them running, but here we select per step)
        self.running_decode = []
        self.running_prefill = []
        
        # Select for Decode
        # Limit by max_decode_batch_size and overall max_batch_size
        num_decodes = min(len(decode_candidates), self.max_decode_batch_size, self.max_batch_size)
        decode_batch = decode_candidates[:num_decodes]
        
        # Remaining decode candidates go back to decode_requests
        remaining_decode = decode_candidates[num_decodes:]
        self.decode_requests = remaining_decode 
        
        # Calculate remaining capacity for Prefill
        remaining_capacity = self.max_batch_size - len(decode_batch)
        
        if remaining_capacity > 0:
            # Select for Prefill
            # We look at pending_requests and prefill_requests
            # (Usually pending -> prefill_requests when tokenized/analyzed, but here we simplify)
            
            # Combine pending and prefill queues for selection
            prefill_candidates = self.running_prefill + self.prefill_requests + self.pending_requests
            
            # Note: pending_requests might need analysis (prefix hash) which is done in add_request
            # So they should be ready.
            
            # Sort/Group by prefix hash could happen here for optimization
            # For now, FIFO
            
            num_prefills = min(len(prefill_candidates), remaining_capacity, self.max_prefill_batch_size)
            prefill_batch = prefill_candidates[:num_prefills]
            
            # Update queues
            # We consumed the first 'num_prefills' from the combined candidates
            # We need to be careful about which list they came from
            # Simplified: Reconstruct lists
            self.running_prefill = []
            self.prefill_requests = prefill_candidates[num_prefills:]
            self.pending_requests = [] # All moved to prefill_requests or scheduled
            
        return prefill_batch, decode_batch

    def process_prefill_batch(self, requests: List[Request]) -> List[Request]:
        """
        Process a prefill batch - handle full prompt processing
        """
        decode_ready = []
        for req in requests:
            req.status = RequestStatus.RUNNING_PREFILL
            self.running_prefill.append(req)
            
            # Simulate Prefill Processing
            # 1. In a real engine, we would run the model's prefill kernel here
            # 2. Allocate KV cache blocks (via memory manager if connected)
            if self.memory_manager:
                # Placeholder for memory allocation
                pass
                
            # After prefill, request is ready for decode
            req.status = RequestStatus.SCHEDULED_DECODE
            
            # Initialize output_ids if not present
            if req.output_ids is None:
                req.output_ids = []
            
            decode_ready.append(req)
            
            # Remove from running_prefill as it's done with prefill phase
            self.running_prefill.remove(req)
            
        # Add to decode queue for next step
        self.decode_requests.extend(decode_ready)
            
        return decode_ready

    def process_decode_batch(self, requests: List[Request]) -> List[Request]:
        """
        Process a decode batch - handle single-token generation
        """
        completed = []
        continue_decode = []

        for req in requests:
            req.status = RequestStatus.RUNNING_DECODE
            self.running_decode.append(req)
            
            # Simulate Decode Processing
            # 1. Run model decode kernel
            # 2. Sample token
            
            # In a real implementation we would get logits from the model
            # Here we just use dummy logits for the placeholder kernel
            import torch
            dummy_logits = torch.randn(1, 1000) # [batch=1, vocab=1000]
            
            try:
                # Use the sampling kernel (which might raise NotImplementedError)
                next_token_tensor = self.sampling_kernel.sample(
                    dummy_logits, 
                    temperature=req.temperature,
                    top_p=req.top_p
                )
                next_token = next_token_tensor.item()
            except NotImplementedError:
                # Fallback for when kernel is not implemented yet (student exercise)
                next_token = 1 # dummy token ID
            
            # Increment position
            req.current_position += 1
            
            # Add token
            if req.output_ids is None:
                req.output_ids = []
            
            req.output_ids.append(next_token) 

            # Check stopping conditions
            if req.current_position >= req.max_tokens:
                req.status = RequestStatus.COMPLETED
                req.created_at = time.time() # Update completion time? Or keep creation time.
                completed.append(req)
                self.completed_requests.append(req)
                # Remove from running
                self.running_decode.remove(req)
            else:
                req.status = RequestStatus.SCHEDULED_DECODE
                continue_decode.append(req)
                # It stays in running_decode until the end of step (where schedule_step re-evaluates)
                # But for our logic, we probably want to move it back to decode_requests
                # managed by schedule_step
                pass

        return completed, continue_decode

    def get_request_result(self, req_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the result for a specific request (non-blocking)

        Args:
            req_id: Request ID

        Returns:
            Result dictionary if available, None otherwise
        """
        # Look in completed requests
        for req in self.completed_requests:
            if req.id == req_id:
                return {
                    "id": req.id,
                    "output": req.output_ids,  # Will need to decode to text
                    "status": req.status.value,
                    "created_at": req.created_at,
                }
        return None

    def get_active_request_count(self) -> int:
        """Get count of active (non-completed) requests"""
        return (
            len(self.pending_requests)
            + len(self.prefill_requests)
            + len(self.running_prefill)
            + len(self.decode_requests)
            + len(self.running_decode)
        )

    def get_queue_status(self) -> Dict[str, int]:
        """Get detailed queue status for monitoring"""
        return {
            "pending": len(self.pending_requests),
            "prefill_queue": len(self.prefill_requests),
            "running_prefill": len(self.running_prefill),
            "decode_queue": len(self.decode_requests),
            "running_decode": len(self.running_decode),
            "completed": len(self.completed_requests),
            "total_active": self.get_active_request_count(),
        }

    def connect_memory_manager(self, memory_manager):
        """Connect the scheduler to the memory manager for KV-cache coordination"""
        self.memory_manager = memory_manager
