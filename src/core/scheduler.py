from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import uuid
import time
from enum import Enum


class RequestStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
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


class Scheduler:
    """
    Handles continuous batching of requests
    Inspired by SGLang's approach to efficient request scheduling
    """

    def __init__(self, max_batch_size: int = 8, max_seq_len: int = 2048):
        """
        Initialize the scheduler

        Args:
            max_batch_size: Maximum number of requests to process together
            max_seq_len: Maximum sequence length
        """
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        # Request queues
        self.pending_requests: List[Request] = []
        self.running_requests: List[Request] = []
        self.completed_requests: List[Request] = []

        # Scheduling components for sophisticated logic
        self.max_prefill_batch_size = max_batch_size * 2  # Allow more prefill requests
        self.max_decode_batch_size = max_batch_size
        self.prefill_requests: List[Request] = []  # Requests ready for prefill
        self.decode_requests: List[Request] = []    # Requests ready for decode
        self.request_lookup: Dict[str, Request] = {}  # Fast lookup

    def add_request(self, prompt: str, max_tokens: int = 128, temperature: float = 1.0, top_p: float = 1.0) -> str:
        """
        Add a new request to the scheduler

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Request ID
        """
        req_id = str(uuid.uuid4())
        request = Request(
            id=req_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            created_at=time.time(),
            status=RequestStatus.PENDING
        )

        self.pending_requests.append(request)
        self.request_lookup[req_id] = request  # Add to lookup for fast access
        return req_id

    def schedule_step(self):
        """
        Perform one scheduling step - batch and dispatch requests for inference
        Implements sophisticated scheduling with separate prefill and decode phases
        """
        # 1. Process any completed requests from the previous step
        # (This is a simplified implementation; in a full system, this would be
        #  handled by the inference backend)
        completed_this_step = []

        # Check running requests for completion
        for req in list(self.running_requests):  # Use list() to avoid modification during iteration
            if req.current_position >= req.max_tokens:
                req.status = RequestStatus.COMPLETED
                self.running_requests.remove(req)
                self.completed_requests.append(req)
                completed_this_step.append(req)

        # 2. Prioritize decode requests (lower latency)
        decode_batch = []
        if self.running_requests:
            # Select decode requests from running ones
            for req in self.running_requests[:self.max_decode_batch_size]:
                if req.status == RequestStatus.RUNNING:
                    decode_batch.append(req)

        # 3. Fill remaining capacity with prefill requests
        remaining_capacity = self.max_batch_size - len(decode_batch)
        prefill_batch = []

        if remaining_capacity > 0 and self.pending_requests:
            # Move some pending requests to prefill
            for req in self.pending_requests[:remaining_capacity]:
                req.status = RequestStatus.RUNNING
                self.pending_requests.remove(req)
                self.running_requests.append(req)
                prefill_batch.append(req)

        # 4. Prepare the full batch (prefill + decode)
        batch = prefill_batch + decode_batch

        # 5. Update request positions for decode steps
        for req in decode_batch:
            req.current_position += 1

        # This is where you'd send the batch to the inference engine
        # For now, we just return the batch information
        return {
            "prefill_batch": prefill_batch,
            "decode_batch": decode_batch,
            "total_batch_size": len(batch),
            "batch": batch
        }

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
                    "created_at": req.created_at
                }

        # Also check if it's still in progress
        for req in self.pending_requests + self.running_requests:
            if req.id == req_id:
                return {
                    "id": req.id,
                    "output": req.output_ids,
                    "status": req.status.value,
                    "current_position": req.current_position,
                    "max_tokens": req.max_tokens,
                    "created_at": req.created_at
                }

        return None

    def get_active_request_count(self) -> int:
        """Get count of active (pending + running) requests"""
        return len(self.pending_requests) + len(self.running_requests)
