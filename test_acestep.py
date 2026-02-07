#!/usr/bin/env python3
"""
Standalone ACE-Step test script
Tests initialization and basic functionality without ComfyUI
"""

import os
import sys
import torch

# Add acestep_repo to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "acestep_repo"))

from transformers import AutoModel, AutoTokenizer
from diffusers.models import AutoencoderOobleck


def initialize_acestep_handler(
    checkpoint_dir: str,
    config_path: str,
    device: str = "auto",
    offload_to_cpu: bool = False,
):
    """Initialize ACE-Step handler with all required components"""

    # Auto-detect device
    if device == "auto":
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            device = "xpu"
        elif torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"[ACE_STEP] Using device: {device}")

    # Build paths
    model_path = os.path.join(checkpoint_dir, config_path)

    # Check paths exist
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found: {model_path}")
    print(f"  ✓ model_path: {model_path}")

    silence_latent_path = os.path.join(model_path, "silence_latent.pt")
    if not os.path.exists(silence_latent_path):
        raise RuntimeError(f"Silence latent not found: {silence_latent_path}")
    print(f"  ✓ silence_latent: {silence_latent_path}")

    vae_path = os.path.join(checkpoint_dir, "vae")
    if not os.path.exists(vae_path):
        raise RuntimeError(f"VAE not found: {vae_path}")
    print(f"  ✓ vae: {vae_path}")

    text_encoder_path = os.path.join(checkpoint_dir, "Qwen3-Embedding-0.6B")
    if not os.path.exists(text_encoder_path):
        raise RuntimeError(f"Text encoder not found: {text_encoder_path}")
    print(f"  ✓ text_encoder: {text_encoder_path}")

    # Set dtype
    dtype = torch.bfloat16 if device in ["cuda", "xpu"] else torch.float32

    # Handler object to store components
    class Handler:
        pass
    handler = Handler()
    handler.device = device
    handler.dtype = dtype
    handler.offload_to_cpu = offload_to_cpu

    # Load DiT model
    print("\n[1/4] Loading DiT model...")
    try:
        handler.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            attn_implementation="sdpa",  # Use sdpa as default
            dtype="bfloat16"
        )
    except Exception as e:
        print(f"  Failed with sdpa: {e}")
        print("  Falling back to eager attention...")
        handler.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            attn_implementation="eager"
        )

    handler.model = handler.model.to(device).to(dtype)
    handler.model.eval()
    print(f"  ✓ DiT model loaded")

    # Load silence_latent
    print("\n[2/4] Loading silence_latent...")
    handler.silence_latent = torch.load(silence_latent_path).transpose(1, 2)
    handler.silence_latent = handler.silence_latent.to(device).to(dtype)
    print(f"  ✓ silence_latent shape: {handler.silence_latent.shape}")

    # Load VAE
    print("\n[3/4] Loading VAE...")
    handler.vae = AutoencoderOobleck.from_pretrained(vae_path)
    vae_dtype = torch.bfloat16 if device in ["cuda", "xpu", "mps"] else torch.float32
    handler.vae = handler.vae.to(device).to(vae_dtype)
    handler.vae.eval()
    print(f"  ✓ VAE loaded")

    # Load text encoder
    print("\n[4/4] Loading text encoder...")
    handler.text_tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
    handler.text_encoder = AutoModel.from_pretrained(text_encoder_path)
    handler.text_encoder = handler.text_encoder.to(device).to(dtype)
    handler.text_encoder.eval()
    print(f"  ✓ Text encoder loaded")

    print("\n✅ All components loaded successfully!")
    return handler


def test_handler(handler):
    """Test handler has all required components"""
    print("\n=== Testing handler components ===")

    assert hasattr(handler, 'model'), "Missing model"
    print("  ✓ model exists")

    assert hasattr(handler, 'vae'), "Missing vae"
    print("  ✓ vae exists")

    assert hasattr(handler, 'text_encoder'), "Missing text_encoder"
    print("  ✓ text_encoder exists")

    assert hasattr(handler, 'text_tokenizer'), "Missing text_tokenizer"
    print("  ✓ text_tokenizer exists")

    assert hasattr(handler, 'silence_latent'), "Missing silence_latent"
    print("  ✓ silence_latent exists")

    # Test silence_latent shape (should be [1, T, D])
    assert handler.silence_latent.dim() == 3, f"silence_latent should be 3D, got {handler.silence_latent.dim()}D"
    print(f"  ✓ silence_latent shape correct: {handler.silence_latent.shape}")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    # Configuration
    if len(sys.argv) > 1:
        checkpoint_dir = sys.argv[1]
    else:
        checkpoint_dir = "/home/yons/work/ai/ComfyUI/models/Ace-Step1.5"

    if len(sys.argv) > 2:
        config_path = sys.argv[2]
    else:
        config_path = "acestep-v15-turbo"

    print(f"=== ACE-Step Initialization Test ===")
    print(f"checkpoint_dir: {checkpoint_dir}")
    print(f"config_path: {config_path}")
    print()

    try:
        handler = initialize_acestep_handler(checkpoint_dir, config_path)
        test_handler(handler)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
