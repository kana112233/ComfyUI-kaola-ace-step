"""
ACE-Step VRAM Management Nodes

Provides nodes for clearing GPU memory, useful after heavy operations
to prevent OOM errors.
"""

import torch
import gc


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
                "clear_python_gc": ("BOOLEAN", {"default": True, "tooltip": "Also run Python garbage collector."}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("pass_through",)
    FUNCTION = "clear_vram"
    CATEGORY = "Audio/ACE-Step"
    OUTPUT_NODE = True

    def clear_vram(self, any_input=None, clear_python_gc=True):
        """Clear GPU memory cache."""

        # Get memory info before clearing
        mem_before = self._get_memory_info()

        # Clear Python garbage collector
        if clear_python_gc:
            gc.collect()

        # Clear CUDA cache
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("[ACE_STEP_ClearVRAM] CUDA cache cleared")

        # Clear MPS cache (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
                print("[ACE_STEP_ClearVRAM] MPS cache cleared")

        # Clear XPU cache (Intel)
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            if hasattr(torch.xpu, 'empty_cache'):
                torch.xpu.empty_cache()
                print("[ACE_STEP_ClearVRAM] XPU cache cleared")

        # Get memory info after clearing
        mem_after = self._get_memory_info()

        # Print memory report
        if mem_before and mem_after:
            freed = mem_before['used'] - mem_after['used']
            print(f"[ACE_STEP_ClearVRAM] Memory: {mem_before['used']:.2f}GB -> {mem_after['used']:.2f}GB (freed: {freed:.2f}GB)")

        return (any_input,)

    def _get_memory_info(self):
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

        # Import here to avoid circular imports
        try:
            from nodes import ACE_STEP_BASE
            from nodes_base_features import _dit_handler, _llm_handler, _handlers_initialized

            # Clear base features cache
            import nodes_base_features as nbf
            if hasattr(nbf, '_dit_handler') and nbf._dit_handler is not None:
                nbf._dit_handler = None
            if hasattr(nbf, '_llm_handler') and nbf._llm_handler is not None:
                nbf._llm_handler = None
            if hasattr(nbf, '_handlers_initialized'):
                nbf._handlers_initialized = False
            print("[ACE_STEP_ClearCache] Base features cache cleared")

        except ImportError:
            pass

        # Also clear general VRAM
        gc.collect()
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()

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
