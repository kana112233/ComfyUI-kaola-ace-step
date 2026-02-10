
import torch
import torch.nn as nn
import logging

logger = logging.getLogger("ACE_STEP_SANITIZER")

class TrainableRotaryEmbedding(torch.nn.Module):
    """RotaryEmbedding that explicitly clones cached tensors to enable gradient flow."""
    def __init__(self, original_rotary):
        super().__init__()
        self.dim = original_rotary.dim
        self.base = original_rotary.base
        self.max_position_embeddings = original_rotary.max_position_embeddings
        self.max_seq_len_cached = original_rotary.max_seq_len_cached

        # Clone buffers from original to fresh tensors
        if hasattr(original_rotary, 'inv_freq'):
            inv_freq = original_rotary.inv_freq.data.clone()
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        if hasattr(original_rotary, 'cos_cached'):
            cos_cached = original_rotary.cos_cached.data.clone()
            self.register_buffer("cos_cached", cos_cached, persistent=False)
            
        if hasattr(original_rotary, 'sin_cached'):
            sin_cached = original_rotary.sin_cached.data.clone()
            self.register_buffer("sin_cached", sin_cached, persistent=False)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device, x.dtype)
        # CRITICAL: Clone the tensors to remove inference tensor taint
        # .to() on matching dtype/device returns a view, which preserves inference status
        # .clone() creates a fresh tensor that can participate in autograd
        cos = self.cos_cached[:seq_len].to(dtype=x.dtype, device=x.device).clone()
        sin = self.sin_cached[:seq_len].to(dtype=x.dtype, device=x.device).clone()
        return (cos, sin)

def replace_inference_modules(module, prefix=""):
    """Replace all modules with inference-tainted weights with fresh PyTorch modules."""
    replaced = {"Linear": 0, "RMSNorm": 0, "LayerNorm": 0, "Conv1d": 0, "ConvTranspose1d": 0, "Embedding": 0, "RotaryEmbedding": 0, "Other": 0}
    device = next(module.parameters()).device if list(module.parameters()) else torch.device("cpu")
    dtype = next(module.parameters()).dtype if list(module.parameters()) else torch.float32

    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name

        # Check module type and create appropriate replacement
        replaced_child = None

        if isinstance(child, torch.nn.Linear):
            replaced_child = torch.nn.Linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                device=child.weight.device,
                dtype=child.weight.dtype
            )
            with torch.no_grad():
                replaced_child.weight.copy_(child.weight.data)
                if child.bias is not None:
                    replaced_child.bias.copy_(child.bias.data)
            replaced["Linear"] += 1

        elif isinstance(child, torch.nn.RMSNorm):
            replaced_child = torch.nn.RMSNorm(
                child.normalized_shape,
                eps=child.eps,
                elementwise_affine=child.weight is not None,
                device=child.weight.device if child.weight is not None else device,
                dtype=child.weight.dtype if child.weight is not None else dtype
            )
            if child.weight is not None:
                with torch.no_grad():
                    replaced_child.weight.copy_(child.weight.data)
            replaced["RMSNorm"] += 1

        elif isinstance(child, torch.nn.LayerNorm):
            replaced_child = torch.nn.LayerNorm(
                child.normalized_shape,
                eps=child.eps,
                elementwise_affine=child.weight is not None,
                device=child.weight.device if child.weight is not None else device,
                dtype=child.weight.dtype if child.weight is not None else dtype
            )
            if child.weight is not None:
                with torch.no_grad():
                    replaced_child.weight.copy_(child.weight.data)
            if child.bias is not None:
                with torch.no_grad():
                    replaced_child.bias.copy_(child.bias.data)
            replaced["LayerNorm"] += 1

        elif isinstance(child, torch.nn.Conv1d):
            replaced_child = torch.nn.Conv1d(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None,
                device=child.weight.device,
                dtype=child.weight.dtype
            )
            with torch.no_grad():
                replaced_child.weight.copy_(child.weight.data)
                if child.bias is not None:
                    replaced_child.bias.copy_(child.bias.data)
            replaced["Conv1d"] += 1

        elif isinstance(child, torch.nn.ConvTranspose1d):
            replaced_child = torch.nn.ConvTranspose1d(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                output_padding=child.output_padding,
                groups=child.groups,
                bias=child.bias is not None,
                dilation=child.dilation,
                padding_mode=child.padding_mode,
                device=child.weight.device,
                dtype=child.weight.dtype
            )
            with torch.no_grad():
                replaced_child.weight.copy_(child.weight.data)
                if child.bias is not None:
                    replaced_child.bias.copy_(child.bias.data)
            replaced["ConvTranspose1d"] += 1

        elif isinstance(child, torch.nn.Embedding):
            replaced_child = torch.nn.Embedding(
                child.num_embeddings,
                child.embedding_dim,
                padding_idx=child.padding_idx,
                max_norm=child.max_norm,
                norm_type=child.norm_type,
                scale_grad_by_freq=child.scale_grad_by_freq,
                sparse=child.sparse,
                device=child.weight.device,
                dtype=child.weight.dtype
            )
            with torch.no_grad():
                replaced_child.weight.copy_(child.weight.data)
            replaced["Embedding"] += 1

        # Check for RotaryEmbedding by attribute presence (handles both ComfyUI and HF implementations)
        elif hasattr(child, 'cos_cached') and hasattr(child, 'sin_cached') and hasattr(child, 'inv_freq'):
            # Make sure it's not already replaced
            if not isinstance(child, TrainableRotaryEmbedding):
                replaced_child = TrainableRotaryEmbedding(child)
                replaced["RotaryEmbedding"] += 1
                # logger.info(f"  Replaced RotaryEmbedding at {full_name}")

        if replaced_child is not None:
            setattr(module, name, replaced_child)
        else:
            # Recursively process children
            child_replaced = replace_inference_modules(child, full_name)
            for k, v in child_replaced.items():
                replaced[k] += v

    return replaced

def clone_all_buffers(module, prefix=""):
    """Clone all registered buffers in a module to remove inference taint."""
    cloned_count = 0
    for name, buf in module.named_buffers(recurse=False):
        if buf is not None:
            # Create a fresh clone of the buffer
            new_buf = buf.data.clone()
            # Re-register the buffer with the cloned data
            module.register_buffer(name, new_buf, persistent=False)
            cloned_count += 1
    # Recursively process children
    for child_name, child in module.named_children():
        cloned_count += clone_all_buffers(child, f"{prefix}.{child_name}" if prefix else child_name)
    return cloned_count

def sanitize_model_for_training(model):
    """
    Main function to sanitize a ComfyUI model for training.
    Replaces inference-tainted modules and clones buffers.
    
    Args:
        model: The root model (usually DiT wrapper or similar)
        
    Returns:
        The sanitized model (modified in-place)
    """
    logger.info("Starting model sanitization for training...")
    
    # We mainly target the decoder, as that's what we train
    if hasattr(model, 'decoder'):
        logger.info("Sanitizing decoder...")
        stats = replace_inference_modules(model.decoder, "decoder")
        logger.info(f"Decoder sanitization stats: {stats}")
        
        bufs = clone_all_buffers(model.decoder, "decoder")
        logger.info(f"Cloned {bufs} buffers in decoder")
        
    # Also sanitize other components that might participate in forward pass (e.g. embeddings)
    # This largely follows the reference implementation logic
    
    components = ['time_embed', 'time_embed_r', 'encoder', 'rotary_emb']
    for comp_name in components:
        if hasattr(model, comp_name):
            comp = getattr(model, comp_name)
            if comp is not None:
                stats = replace_inference_modules(comp, comp_name)
                # logger.info(f"{comp_name} sanitization stats: {stats}")
                clone_all_buffers(comp, comp_name)

    # Handle condition_embedder specially if it's just a Linear layer
    if hasattr(model, 'condition_embedder') and model.condition_embedder is not None:
        if isinstance(model.condition_embedder, torch.nn.Linear):
            old = model.condition_embedder
            new_layer = torch.nn.Linear(
                old.in_features, old.out_features,
                bias=old.bias is not None,
                device=old.weight.device, dtype=old.weight.dtype
            )
            with torch.no_grad():
                new_layer.weight.copy_(old.weight.data)
                if old.bias is not None:
                    new_layer.bias.copy_(old.bias.data)
            model.condition_embedder = new_layer
            logger.info("Sanitized condition_embedder (Linear)")

    logger.info("Model sanitization complete.")
    return model
