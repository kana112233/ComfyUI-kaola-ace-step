"""
ACE-Step VRAM Management Nodes

Provides nodes for clearing GPU memory, useful after heavy operations
to prevent OOM errors.
"""

import torch
import gc
import sys


def _get_memory_info():
    """Get GPU memory info if available."""
    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            return {
                'used': allocated,
                'reserved': reserved,
            }
        except Exception:
            pass
    return None


def _delete_model_recursive(obj, visited=None):
    """Recursively delete model and move tensors to CPU then delete."""
    if visited is None:
        visited = set()

    # Avoid infinite recursion
    obj_id = id(obj)
    if obj_id in visited:
        return
    visited.add(obj_id)

    # Handle torch Tensor - move to CPU then delete
    if isinstance(obj, torch.Tensor):
        try:
            obj.cpu()
            del obj
        except Exception:
            pass
        return

    # Handle torch Module - recursively clean up
    if isinstance(obj, torch.nn.Module):
        # Move to CPU first
        try:
            obj.cpu()
        except Exception:
            pass

        # Delete parameters and buffers
        for name, param in list(obj.named_parameters()):
            try:
                param.cpu()
                del param
            except Exception:
                pass

        for name, buf in list(obj.named_buffers()):
            try:
                buf.cpu()
                del buf
            except Exception:
                pass

        # Clear modules dict
        try:
            obj._modules.clear()
        except Exception:
            pass

        # Clear parameters dict
        try:
            obj._parameters.clear()
        except Exception:
            pass

        # Clear buffers dict
        try:
            obj._buffers.clear()
        except Exception:
            pass

    # Handle dict
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            _delete_model_recursive(obj[key], visited)
        obj.clear()

    # Handle list/tuple/set
    elif isinstance(obj, (list, set)):
        for item in list(obj):
            _delete_model_recursive(item, visited)
        obj.clear()


def _clear_handler(handler):
    """Clear a single handler thoroughly."""
    if handler is None:
        return

    # Get all attributes that might contain models/tensors
    model_attrs = ['model', 'vae', 'text_encoder', 'text_tokenizer',
                   'silence_latent', 'lm_model', 'tokenizer']

    for attr in model_attrs:
        if hasattr(handler, attr):
            obj = getattr(handler, attr)
            if obj is not None:
                _delete_model_recursive(obj)
                try:
                    delattr(handler, attr)
                except Exception:
                    pass

    # Also try to move the handler itself if it has a model
    if hasattr(handler, 'model') and handler.model is not None:
        try:
            handler.model.cpu()
        except Exception:
            pass


def _aggressive_memory_cleanup():
    """Aggressive memory cleanup with multiple GC cycles."""
    # Multiple GC cycles
    for _ in range(3):
        gc.collect()

    # Clear CUDA
    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        # Reset peak memory stats
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    # Clear MPS
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()

    # Clear XPU
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        if hasattr(torch.xpu, 'empty_cache'):
            torch.xpu.empty_cache()

    # Final GC
    gc.collect()


class ACE_STEP_CLEAR_VRAM:
    """Clear GPU VRAM to free memory.

    This node clears the GPU memory cache. Use it after heavy operations
    (like model inference) to prevent out-of-memory errors.

    Supports: CUDA, MPS (Apple Silicon), XPU (Intel)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "any_input": ("*", {"tooltip": "Optional input to pass through. Connect from previous node."}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("pass_through",)
    FUNCTION = "clear_vram"
    CATEGORY = "Audio/ACE-Step"
    OUTPUT_NODE = True

    def clear_vram(self, any_input=None):
        """Clear GPU memory cache."""
        mem_before = _get_memory_info()

        _aggressive_memory_cleanup()

        mem_after = _get_memory_info()

        if mem_before and mem_after:
            freed = mem_before['used'] - mem_after['used']
            print(f"[ACE_STEP_ClearVRAM] Memory: {mem_before['used']:.2f}GB -> {mem_after['used']:.2f}GB (freed: {freed:.2f}GB)")
        else:
            print("[ACE_STEP_ClearVRAM] GPU cache cleared")

        return (any_input,)


class ACE_STEP_CLEAR_ACESTEP_CACHE:
    """Clear ACE-Step model handlers cache.

    This node clears the internal ACE-Step handler cache, forcing
    models to be reloaded on next use. Use this when switching
    between different model configurations or when experiencing
    memory issues.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "any_input": ("*", {"tooltip": "Optional input to pass through."}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("pass_through",)
    FUNCTION = "clear_cache"
    CATEGORY = "Audio/ACE-Step"
    OUTPUT_NODE = True

    def clear_cache(self, any_input=None):
        """Clear ACE-Step internal handler cache."""
        mem_before = _get_memory_info()

        # Clear base features handlers
        try:
            import nodes_base_features as nbf

            if hasattr(nbf, '_dit_handler') and nbf._dit_handler is not None:
                _clear_handler(nbf._dit_handler)
                nbf._dit_handler = None

            if hasattr(nbf, '_llm_handler') and nbf._llm_handler is not None:
                _clear_handler(nbf._llm_handler)
                nbf._llm_handler = None

            if hasattr(nbf, '_handlers_initialized'):
                nbf._handlers_initialized = False

            print("[ACE_STEP_ClearCache] Base features handlers cleared")

        except ImportError as e:
            print(f"[ACE_STEP_ClearCache] Could not import base features: {e}")

        # Clear main nodes handlers (ACE_STEP_BASE instances)
        try:
            import nodes as main_nodes

            # Find all ACE_STEP_BASE instances in the module
            for name in dir(main_nodes):
                try:
                    obj = getattr(main_nodes, name)
                    if isinstance(obj, type) and hasattr(obj, '__mro__'):
                        # Check if it's a subclass of ACE_STEP_BASE
                        for base in obj.__mro__:
                            if base.__name__ == 'ACE_STEP_BASE':
                                # Clear class-level handlers if they exist
                                if hasattr(obj, 'dit_handler'):
                                    _clear_handler(obj.dit_handler)
                                    obj.dit_handler = None
                                if hasattr(obj, 'llm_handler'):
                                    _clear_handler(obj.llm_handler)
                                    obj.llm_handler = None
                                if hasattr(obj, 'handlers_initialized'):
                                    obj.handlers_initialized = False
                                break
                except Exception:
                    pass

        except ImportError:
            pass

        # Aggressive cleanup
        _aggressive_memory_cleanup()

        mem_after = _get_memory_info()

        if mem_before and mem_after:
            freed = mem_before['used'] - mem_after['used']
            print(f"[ACE_STEP_ClearCache] Memory: {mem_before['used']:.2f}GB -> {mem_after['used']:.2f}GB (freed: {freed:.2f}GB)")
        else:
            print("[ACE_STEP_ClearCache] ACE-Step cache cleared")

        return (any_input,)


# Node mappings will be added to nodes.py
NODE_CLASS_MAPPINGS = {
    "ACE_STEP_ClearVRAM": ACE_STEP_CLEAR_VRAM,
    "ACE_STEP_ClearCache": ACE_STEP_CLEAR_ACESTEP_CACHE,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ACE_STEP_ClearVRAM": "ACE-Step Clear VRAM",
    "ACE_STEP_ClearCache": "ACE-Step Clear Model Cache",
}
