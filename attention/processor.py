"""
Unified attention processor with multiple backend support.

Backends:
- sage: SageAttention 2.x (fastest for long sequences)
- flash: FlashAttention 2/3 (fast for medium sequences)
- sdpa: PyTorch scaled_dot_product_attention (universal fallback)
- auto: Smart selection based on sequence length
"""

import inspect
import warnings
from typing import Optional, Literal, Dict, Any

import torch
import torch.nn.functional as F

# Backend availability tracking
AVAILABLE_BACKENDS: Dict[str, bool] = {
    "sage": False,
    "flash": False,
    "sdpa": True,  # Always available in PyTorch 2.0+
}

# SageAttention
sageattn_func = None
try:
    from sageattention import sageattn as sageattn_func
    AVAILABLE_BACKENDS["sage"] = True
except ImportError:
    pass

# FlashAttention
flash_attn_func = None
flash_attn_backend = None
flash_attn_supports_dropout = False
flash_attn_supports_softmax_scale = False

try:
    from flash_attn_interface import flash_attn_func
    flash_attn_backend = "flash_attn_interface"
    AVAILABLE_BACKENDS["flash"] = True
except ImportError:
    try:
        from flash_attn.flash_attn_interface import flash_attn_func
        flash_attn_backend = "flash_attn.flash_attn_interface"
        AVAILABLE_BACKENDS["flash"] = True
    except ImportError:
        try:
            from flash_attn import flash_attn_func
            flash_attn_backend = "flash_attn"
            AVAILABLE_BACKENDS["flash"] = True
        except ImportError:
            pass

# Detect FlashAttention feature support
if flash_attn_func is not None:
    try:
        signature = inspect.signature(flash_attn_func)
        flash_attn_supports_dropout = "dropout_p" in signature.parameters
        flash_attn_supports_softmax_scale = "softmax_scale" in signature.parameters
    except (ValueError, TypeError):
        pass


def detect_available_backends() -> Dict[str, bool]:
    """Return dict of available attention backends."""
    return AVAILABLE_BACKENDS.copy()


BackendType = Literal["sage", "flash", "sdpa", "auto"]

# Sequence length threshold for auto mode
# SageAttention is faster for long sequences, FlashAttention for shorter ones
SAGE_MIN_SEQ_LENGTH = 1024


class AttentionProcessor:
    """
    Unified attention processor supporting multiple backends.

    Args:
        backend: Which attention backend to use:
            - "sage": SageAttention (fast for long sequences)
            - "flash": FlashAttention 2/3 (fast for medium sequences)
            - "sdpa": PyTorch SDPA (universal fallback)
            - "auto": Smart selection based on sequence length (default)
        sage_min_seq_length: Minimum sequence length to use SageAttention in auto mode.
            Below this threshold, FlashAttention is used if available. Default: 1024
        fallback_on_error: If True, fall back to SDPA on backend errors. Default: True
    """

    SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)

    def __init__(
        self,
        backend: BackendType = "auto",
        sage_min_seq_length: int = SAGE_MIN_SEQ_LENGTH,
        fallback_on_error: bool = True,
    ):
        self.backend = backend
        self.sage_min_seq_length = sage_min_seq_length
        self.fallback_on_error = fallback_on_error

        # Warning flags (only warn once per instance)
        self._warned_sage_unavailable = False
        self._warned_flash_unavailable = False
        self._warned_sage_error = False
        self._warned_flash_error = False
        self._warned_dropout = False

        # Validate backend selection
        if backend == "sage" and not AVAILABLE_BACKENDS["sage"]:
            warnings.warn(
                "SageAttention requested but not available. Install with: "
                "pip install sageattention. Falling back to auto mode.",
                stacklevel=2,
            )
            self.backend = "auto"
        elif backend == "flash" and not AVAILABLE_BACKENDS["flash"]:
            warnings.warn(
                "FlashAttention requested but not available. "
                "Falling back to auto mode.",
                stacklevel=2,
            )
            self.backend = "auto"

    def _select_backend(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        head_dim: int,
        seq_length: int,
    ) -> str:
        """Select the best backend for the given inputs."""

        # Check basic requirements
        is_cuda = hidden_states.device.type == "cuda"
        is_supported_dtype = hidden_states.dtype in self.SUPPORTED_DTYPES
        has_mask = attention_mask is not None
        valid_head_dim = head_dim <= 256 and head_dim % 8 == 0

        # Explicit backend selection (non-auto)
        if self.backend != "auto":
            if self.backend == "sage":
                if not AVAILABLE_BACKENDS["sage"]:
                    return "sdpa"
                if not is_cuda or not is_supported_dtype:
                    return "sdpa"
                if has_mask:
                    # SageAttention doesn't support attention masks
                    return "sdpa"
                return "sage"

            elif self.backend == "flash":
                if not AVAILABLE_BACKENDS["flash"]:
                    return "sdpa"
                if not is_cuda or not is_supported_dtype:
                    return "sdpa"
                if has_mask or not valid_head_dim:
                    return "sdpa"
                return "flash"

            else:  # sdpa
                return "sdpa"

        # Auto mode: smart selection based on sequence length and availability
        if not is_cuda or not is_supported_dtype:
            return "sdpa"

        if has_mask:
            # Only SDPA supports attention masks
            return "sdpa"

        # For long sequences, prefer SageAttention
        if seq_length >= self.sage_min_seq_length and AVAILABLE_BACKENDS["sage"]:
            return "sage"

        # For shorter sequences, prefer FlashAttention
        if AVAILABLE_BACKENDS["flash"] and valid_head_dim:
            return "flash"

        # SageAttention as secondary option for short sequences
        if AVAILABLE_BACKENDS["sage"]:
            return "sage"

        return "sdpa"

    def _run_sage_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: Optional[float],
    ) -> torch.Tensor:
        """
        Run SageAttention.

        Input tensors are in SDPA format: (batch, heads, seq, dim)
        SageAttention with tensor_layout='HND' expects: (batch, heads, seq, dim)
        So no transpose needed!
        """
        # Ensure contiguous tensors
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        # Call SageAttention
        # tensor_layout='HND' means (batch, heads, seq_len, head_dim)
        output = sageattn_func(
            query,
            key,
            value,
            tensor_layout="HND",
            is_causal=False,
            sm_scale=scale,
        )

        return output

    def _run_flash_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: Optional[float],
        dropout_p: float,
    ) -> torch.Tensor:
        """
        Run FlashAttention.

        Input tensors are in SDPA format: (batch, heads, seq, dim)
        FlashAttention expects: (batch, seq, heads, dim)
        """
        # Transpose to FlashAttention format
        flash_query = query.transpose(1, 2).contiguous()
        flash_key = key.transpose(1, 2).contiguous()
        flash_value = value.transpose(1, 2).contiguous()

        # Build kwargs
        flash_kwargs = {"causal": False}

        if scale is not None and flash_attn_supports_softmax_scale:
            flash_kwargs["softmax_scale"] = scale

        if dropout_p > 0.0:
            if flash_attn_supports_dropout:
                flash_kwargs["dropout_p"] = dropout_p
            elif not self._warned_dropout:
                warnings.warn(
                    "FlashAttention backend does not support dropout. "
                    "Dropout will be ignored.",
                    stacklevel=3,
                )
                self._warned_dropout = True

        output = flash_attn_func(flash_query, flash_key, flash_value, **flash_kwargs)

        # Transpose back to SDPA format
        return output.transpose(1, 2)

    def _run_sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        dropout_p: float,
    ) -> torch.Tensor:
        """Run PyTorch scaled_dot_product_attention."""
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Process attention using the selected backend.

        This follows the diffusers AttnProcessor interface.
        """
        # Handle deprecated scale argument
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            warnings.warn(
                "The `scale` argument is deprecated and will be ignored.",
                stacklevel=2,
            )

        residual = hidden_states

        # Spatial norm (for video models)
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        # Handle 4D input (B, C, H, W) -> (B, H*W, C)
        input_ndim = hidden_states.ndim
        height = width = channel = None
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        # Group norm
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)
            ).transpose(1, 2)

        # Compute Q, K, V
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Reshape to multi-head format: (batch, heads, seq, dim)
        head_dim = query.shape[-1] // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Apply Q/K normalization if present
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Get sequence length for backend selection
        seq_length = query.shape[2]

        # Select and run attention backend
        selected_backend = self._select_backend(
            hidden_states, attention_mask, head_dim, seq_length
        )

        dropout_p = attn.dropout if attn.training else 0.0
        scale = attn.scale if getattr(attn, "scale_qk", True) else None

        try:
            if selected_backend == "sage":
                hidden_states = self._run_sage_attention(query, key, value, scale)

            elif selected_backend == "flash":
                hidden_states = self._run_flash_attention(
                    query, key, value, scale, dropout_p
                )

            else:  # sdpa
                hidden_states = self._run_sdpa(
                    query, key, value, attention_mask, dropout_p
                )

        except RuntimeError as err:
            # Fallback to SDPA on error
            if self.fallback_on_error:
                if selected_backend == "sage" and not self._warned_sage_error:
                    warnings.warn(
                        f"SageAttention failed, falling back to SDPA: {err}",
                        stacklevel=2,
                    )
                    self._warned_sage_error = True
                elif selected_backend == "flash" and not self._warned_flash_error:
                    warnings.warn(
                        f"FlashAttention failed, falling back to SDPA: {err}",
                        stacklevel=2,
                    )
                    self._warned_flash_error = True

                hidden_states = self._run_sdpa(
                    query, key, value, attention_mask, 0.0
                )
            else:
                raise

        # Reshape back: (batch, heads, seq, dim) -> (batch, seq, heads*dim)
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        # Handle 4D output
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        # Residual connection
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def get_attention_processor(
    backend: BackendType = "auto",
    sage_min_seq_length: int = SAGE_MIN_SEQ_LENGTH,
    fallback_on_error: bool = True,
) -> AttentionProcessor:
    """
    Factory function to create an attention processor.

    Args:
        backend: Which attention backend to use:
            - "sage": SageAttention (fast for long sequences)
            - "flash": FlashAttention 2/3 (fast for medium sequences)
            - "sdpa": PyTorch SDPA (universal fallback)
            - "auto": Smart selection based on sequence length (default)
        sage_min_seq_length: Minimum sequence length to use SageAttention in auto mode.
        fallback_on_error: If True, fall back to SDPA on backend errors.

    Returns:
        AttentionProcessor instance
    """
    return AttentionProcessor(
        backend=backend,
        sage_min_seq_length=sage_min_seq_length,
        fallback_on_error=fallback_on_error,
    )


def print_backend_status():
    """Print available attention backends and their status."""
    print("Attention Backend Status:")
    print(f"  SageAttention: {'available' if AVAILABLE_BACKENDS['sage'] else 'not installed'}")
    if AVAILABLE_BACKENDS["flash"]:
        print(f"  FlashAttention: available (backend: {flash_attn_backend})")
        print(f"    - dropout support: {flash_attn_supports_dropout}")
        print(f"    - scale kwarg: {flash_attn_supports_softmax_scale}")
    else:
        print("  FlashAttention: not installed")
    print(f"  PyTorch SDPA: available (always)")
