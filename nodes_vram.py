"""
ACE-Step VRAM Management Nodes
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


def _delete_model(obj):
    """Delete a model object and free its memory."""
    if obj is None:
        return

    try:
        if hasattr(obj, 'cpu'):
            obj.cpu()
        elif hasattr(obj, 'to'):
            obj.to('cpu')
    except Exception:
        pass

    if isinstance(obj, torch.nn.Module):
        try:
            obj.zero_grad(True)
        except Exception:
            pass

    try:
        del obj
    except Exception:
        pass


def _clear_acestep_handlers():
    """Clear all ACE-Step handlers and models."""
    cleared_models = []

    try:
        import nodes_base_features as nbf

        # Clear DiT handler
        if hasattr(nbf, '_dit_handler') and nbf._dit_handler is not None:
            handler = nbf._dit_handler
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
            for attr in ['model', 'tokenizer', 'lm_model']:
                if hasattr(handler, attr):
                    obj = getattr(handler, attr)
                    if obj is not None:
                        _delete_model(obj)
                        cleared_models.append(f"llm.{attr}")
            nbf._llm_handler = None

        if hasattr(nbf, '_handlers_initialized'):
            nbf._handlers_initialized = False

    except ImportError:
        pass

    return cleared_models


class ACE_STEP_CLEAR_VRAM:
    """Clear GPU VRAM and ACE-Step models.

    Use this node after ACE-Step operations to free VRAM.
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
        print("[ACE_STEP_ClearVRAM] Starting VRAM cleanup...")

        mem_before = _get_memory_info()
        if mem_before:
            print(f"[ACE_STEP_ClearVRAM] Before: Allocated={mem_before['used']:.2f}GB, Reserved={mem_before['reserved']:.2f}GB")

        # Clear ACE-Step handlers
        print("[ACE_STEP_ClearVRAM] Clearing ACE-Step handlers...")
        cleared = _clear_acestep_handlers()

        # Clear CUDA memory with multiple rounds
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            print("[ACE_STEP_ClearVRAM] Clearing CUDA cache (3 rounds)...")
            torch.cuda.synchronize()
            for i in range(3):
                gc.collect()
                torch.cuda.empty_cache()
                print(f"[ACE_STEP_ClearVRAM]   Round {i+1}/3 done")
            try:
                torch.cuda.reset_peak_memory_stats()
                print("[ACE_STEP_ClearVRAM] Reset peak memory stats")
            except Exception:
                pass

        # Also use ComfyUI's memory management
        try:
            import comfy.model_management as mm
            print("[ACE_STEP_ClearVRAM] Using ComfyUI memory management...")
            mm.free_memory(1.0, mm.get_torch_device())
            mm.soft_empty_cache()
            print("[ACE_STEP_ClearVRAM] ComfyUI cache cleared")
        except ImportError:
            print("[ACE_STEP_ClearVRAM] ComfyUI model_management not available")

        mem_after = _get_memory_info()

        if cleared:
            print(f"[ACE_STEP_ClearVRAM] Models cleared: {', '.join(cleared)}")

        if mem_before and mem_after:
            freed = mem_before['used'] - mem_after['used']
            print(f"[ACE_STEP_ClearVRAM] After: Allocated={mem_after['used']:.2f}GB, Reserved={mem_after['reserved']:.2f}GB")
            print(f"[ACE_STEP_ClearVRAM] Total freed: {freed:.2f}GB")
        else:
            print("[ACE_STEP_ClearVRAM] GPU memory cleared")

        return (any_input,)


NODE_CLASS_MAPPINGS = {
    "ACE_STEP_ClearVRAM": ACE_STEP_CLEAR_VRAM,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ACE_STEP_ClearVRAM": "ACE-Step Clear VRAM",
}
