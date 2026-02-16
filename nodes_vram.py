"""
ACE-Step VRAM Management Nodes

Provides nodes for clearing GPU memory, useful after heavy operations
to prevent OOM errors.
"""

import torch
import gc


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


def _clear_cuda_memory():
    """Clear CUDA memory aggressively."""
    if not (hasattr(torch, 'cuda') and torch.cuda.is_available()):
        return

    # Synchronize first
    torch.cuda.synchronize()

    # Multiple rounds of cleanup
    for _ in range(5):
        gc.collect()
        torch.cuda.empty_cache()

    # Reset memory stats
    try:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
    except Exception:
        pass

    # Final collection
    gc.collect()
    torch.cuda.empty_cache()


def _delete_model(obj):
    """Delete a model object and free its memory."""
    if obj is None:
        return

    try:
        # Move to CPU first
        if hasattr(obj, 'cpu'):
            obj.cpu()
        elif hasattr(obj, 'to'):
            obj.to('cpu')
    except Exception:
        pass

    # If it's a module, clear its state
    if isinstance(obj, torch.nn.Module):
        # Zero out gradients
        try:
            obj.zero_grad(True)
        except Exception:
            pass

        # Clear hooks
        try:
            for handle in obj._forward_pre_hooks.values():
                handle.remove()
            obj._forward_pre_hooks.clear()
        except Exception:
            pass

        try:
            for handle in obj._forward_hooks.values():
                handle.remove()
            obj._forward_hooks.clear()
        except Exception:
            pass

        try:
            for handle in obj._backward_hooks.values():
                handle.remove()
            obj._backward_hooks.clear()
        except Exception:
            pass

    # Delete the object
    try:
        del obj
    except Exception:
        pass


def _clear_acestep_handlers():
    """Clear all ACE-Step handlers and models."""
    cleared_models = []

    # Clear base features handlers
    try:
        import nodes_base_features as nbf

        # Clear DiT handler
        if hasattr(nbf, '_dit_handler') and nbf._dit_handler is not None:
            handler = nbf._dit_handler
            # Delete all model components
            for attr in ['model', 'vae', 'text_encoder', 'text_tokenizer', 'silence_latent']:
                if hasattr(handler, attr):
                    obj = getattr(handler, attr)
                    if obj is not None:
                        _delete_model(obj)
                        cleared_models.append(f"dit.{attr}")
            nbf._dit_handler = None

        # Clear LLM handler
        if hasattr(nbf, '_llm_handler') and nbf._llm_handler is not None:
            handler = nbf._llm_handler
            # Delete LLM components
            for attr in ['model', 'tokenizer', 'lm_model']:
                if hasattr(handler, attr):
                    obj = getattr(handler, attr)
                    if obj is not None:
                        _delete_model(obj)
                        cleared_models.append(f"llm.{attr}")
            nbf._llm_handler = None

        # Reset initialization flag
        if hasattr(nbf, '_handlers_initialized'):
            nbf._handlers_initialized = False

    except ImportError:
        pass

    # Also try to use ComfyUI's model management
    try:
        import comfy.model_management as mm

        # Free all models from ComfyUI's management
        mm.free_memory(1.0, mm.get_torch_device())
        mm.soft_empty_cache()

        cleared_models.append("comfyui_cache")

    except ImportError:
        pass

    return cleared_models


class ACE_STEP_CLEAR_VRAM:
    """Clear GPU VRAM and ACE-Step models.

    This node clears all GPU memory including ACE-Step models.
    Use it after ACE-Step operations to free VRAM for other tasks.
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
    FUNCTION = "clear_vram"
    CATEGORY = "Audio/ACE-Step"
    OUTPUT_NODE = True

    def clear_vram(self, any_input=None):
        """Clear all GPU memory."""
        mem_before = _get_memory_info()

        # Clear ACE-Step handlers first
        cleared = _clear_acestep_handlers()

        # Clear CUDA memory
        _clear_cuda_memory()

        mem_after = _get_memory_info()

        # Report
        if cleared:
            print(f"[ACE_STEP_ClearVRAM] Cleared: {', '.join(cleared)}")

        if mem_before and mem_after:
            freed = mem_before['used'] - mem_after['used']
            print(f"[ACE_STEP_ClearVRAM] VRAM: {mem_before['used']:.2f}GB -> {mem_after['used']:.2f}GB (freed: {freed:.2f}GB)")
        else:
            print("[ACE_STEP_ClearVRAM] GPU memory cleared")

        return (any_input,)


class ACE_STEP_FREE_MODEL:
    """Free ACE-Step models from memory.

    This node specifically clears the ACE-Step model cache,
    forcing models to be reloaded on next use.
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
    FUNCTION = "free_model"
    CATEGORY = "Audio/ACE-Step"
    OUTPUT_NODE = True

    def free_model(self, any_input=None):
        """Free ACE-Step models only."""
        mem_before = _get_memory_info()

        # Clear handlers
        cleared = _clear_acestep_handlers()

        # Basic cleanup
        gc.collect()
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()

        mem_after = _get_memory_info()

        if cleared:
            print(f"[ACE_STEP_FreeModel] Freed: {', '.join(cleared)}")

        if mem_before and mem_after:
            freed = mem_before['used'] - mem_after['used']
            print(f"[ACE_STEP_FreeModel] VRAM: {mem_before['used']:.2f}GB -> {mem_after['used']:.2f}GB (freed: {freed:.2f}GB)")

        return (any_input,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "ACE_STEP_ClearVRAM": ACE_STEP_CLEAR_VRAM,
    "ACE_STEP_FreeModel": ACE_STEP_FREE_MODEL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ACE_STEP_ClearVRAM": "ACE-Step Clear VRAM",
    "ACE_STEP_FreeModel": "ACE-Step Free Model",
}
