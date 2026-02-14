"""
Attention backends for SDXL training.

Supports multiple backends with automatic selection and fallback:
- SageAttention 2.x: Fast for long sequences (1024+ tokens)
- FlashAttention 2/3: Fast for medium sequences
- PyTorch SDPA: Universal fallback
"""

from .processor import (
    AttentionProcessor,
    get_attention_processor,
    AVAILABLE_BACKENDS,
    detect_available_backends,
    print_backend_status,
    SAGE_MIN_SEQ_LENGTH,
)

__all__ = [
    "AttentionProcessor",
    "get_attention_processor",
    "AVAILABLE_BACKENDS",
    "detect_available_backends",
    "print_backend_status",
    "SAGE_MIN_SEQ_LENGTH",
]
