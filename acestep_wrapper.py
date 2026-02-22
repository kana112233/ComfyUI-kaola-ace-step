"""
ACE-Step Wrapper Module

Provides a clean interface to initialize and use ACE-Step components
independently of ComfyUI. This makes testing and maintenance easier.
"""

import os
import torch
from typing import Optional


class ACEStepWrapper:
    """Wrapper for ACE-Step initialization and operations"""

    def __init__(self):
        self.device = None
        self.dtype = None
        self.offload_to_cpu = False

        # Components
        self.model = None
        self.vae = None
        self.text_encoder = None
        self.text_tokenizer = None
        self.silence_latent = None

    def initialize(
        self,
        checkpoint_dir: str,
        config_path: str,
        device: str = "auto",
        offload_to_cpu: bool = False,
        quantization: Optional[str] = None,
        compile_model: bool = False,
    ):
        """Initialize all ACE-Step components

        Args:
            checkpoint_dir: Directory containing model checkpoints
            config_path: Model config name (e.g., "acestep-v15-turbo")
            device: Device to use ("auto", "cuda", "cpu", "mps", "xpu")
            offload_to_cpu: Whether to offload models to CPU when not in use
            quantization: Quantization type (None, "int8_weight_only", etc.)
            compile_model: Whether to use torch.compile

        Returns:
            self for chaining
        """
        from transformers import AutoModel, AutoTokenizer
        from diffusers.models import AutoencoderOobleck

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

        # Set attributes
        self.device = device
        self.offload_to_cpu = offload_to_cpu
        self.dtype = torch.bfloat16 if device in ["cuda", "xpu"] else torch.float32

        # Validate quantization requirements
        if quantization is not None:
            assert compile_model, "Quantization requires compile_model to be True"
            try:
                import torchao
            except ImportError:
                raise ImportError("torchao is required for quantization")

        # Build paths
        model_path = os.path.join(checkpoint_dir, config_path)
        silence_latent_path = os.path.join(model_path, "silence_latent.pt")
        vae_path = os.path.join(checkpoint_dir, "vae")
        text_encoder_path = os.path.join(checkpoint_dir, "Qwen3-Embedding-0.6B")

        # Check paths exist
        self._verify_paths(model_path, silence_latent_path, vae_path, text_encoder_path)

        # Load components
        self._load_dit_model(model_path, device, quantization, compile_model)
        self._load_silence_latent(silence_latent_path, device)
        self._load_vae(vae_path, device)
        self._load_text_encoder(text_encoder_path, device)

        print(f"[ACE_STEP] Initialization complete")
        return self

    def _verify_paths(self, model_path: str, silence_latent_path: str,
                      vae_path: str, text_encoder_path: str):
        """Verify all required paths exist"""
        if not os.path.exists(model_path):
            raise RuntimeError(f"Model not found: {model_path}")
        print(f"  ✓ model_path: {model_path}")

        if not os.path.exists(silence_latent_path):
            raise RuntimeError(f"Silence latent not found: {silence_latent_path}")
        print(f"  ✓ silence_latent: {silence_latent_path}")

        if not os.path.exists(vae_path):
            raise RuntimeError(f"VAE not found: {vae_path}")
        print(f"  ✓ vae: {vae_path}")

        if not os.path.exists(text_encoder_path):
            raise RuntimeError(f"Text encoder not found: {text_encoder_path}")
        print(f"  ✓ text_encoder: {text_encoder_path}")

    def _load_dit_model(self, model_path: str, device: str,
                        quantization: Optional[str], compile_model: bool):
        """Load DiT model"""
        from transformers import AutoModel

        print(f"\n[1/4] Loading DiT model from: {model_path}")
        
        # Emergency patch for transformers v5 / compatibility issues
        try:
            import transformers.configuration_utils
            if not hasattr(transformers.configuration_utils, "layer_type_validation"):
                print(f"  [ACE_STEP] Patching layer_type_validation in transformers.configuration_utils")
                def layer_type_validation(*args, **kwargs):
                    return args[0] if args else None
                transformers.configuration_utils.layer_type_validation = layer_type_validation
            else:
                print(f"  [ACE_STEP] layer_type_validation already exists in transformers.configuration_utils")
        except Exception as e:
            print(f"  [ACE_STEP] Warning: Failed to apply transformers patch: {e}")

        # Try to determine best attention implementation
        attn_implementation = self._get_attention_implementation()

        # Load the model
        try:
            print(f"  Using attention: {attn_implementation}")
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                dtype="bfloat16"
            )
        except Exception as e:
            print(f"  Failed with {attn_implementation}: {e}")
            if attn_implementation == "sdpa":
                print("  Falling back to eager attention...")
                attn_implementation = "eager"
                self.model = AutoModel.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    attn_implementation=attn_implementation
                )
            else:
                raise e

        self.model.config._attn_implementation = attn_implementation

        # Move to device
        if not self.offload_to_cpu:
            self.model = self.model.to(device).to(self.dtype)
        else:
            self.model = self.model.to("cpu").to(self.dtype)

        self.model.eval()

        # Compile if requested
        if compile_model:
            self._compile_model(quantization)

        print(f"  ✓ DiT model loaded")

    def _get_attention_implementation(self) -> str:
        """Determine best attention implementation"""
        try:
            import flash_attn
            return "flash_attention_2"
        except ImportError:
            return "sdpa"

    def _compile_model(self, quantization: Optional[str]):
        """Compile model with optional quantization"""
        if not hasattr(self.model.__class__, '__len__'):
            def _model_len(model_self):
                return 0
            self.model.__class__.__len__ = _model_len

        self.model = torch.compile(self.model)

        if quantization:
            from torchao.quantization import quantize_
            if quantization == "int8_weight_only":
                from torchao.quantization import Int8WeightOnlyConfig
                quant_config = Int8WeightOnlyConfig()
            elif quantization == "fp8_weight_only":
                from torchao.quantization import Float8WeightOnlyConfig
                quant_config = Float8WeightOnlyConfig()
            elif quantization == "w8a8_dynamic":
                from torchao.quantization import Int8DynamicActivationInt8WeightConfig, MappingType
                quant_config = Int8DynamicActivationInt8WeightConfig(act_mapping_type=MappingType.ASYMMETRIC)
            else:
                raise ValueError(f"Unsupported quantization: {quantization}")

            quantize_(self.model, quant_config)
            print(f"  ✓ Model quantized with: {quantization}")

    def _load_silence_latent(self, path: str, device: str):
        """Load silence_latent tensor"""
        print(f"\n[2/4] Loading silence_latent...")
        self.silence_latent = torch.load(path).transpose(1, 2)
        self.silence_latent = self.silence_latent.to(device).to(self.dtype)
        print(f"  ✓ silence_latent shape: {self.silence_latent.shape}")

    def _load_vae(self, path: str, device: str):
        """Load VAE model"""
        from diffusers.models import AutoencoderOobleck

        print(f"\n[3/4] Loading VAE...")
        self.vae = AutoencoderOobleck.from_pretrained(path)
        vae_dtype = torch.bfloat16 if device in ["cuda", "xpu", "mps"] else torch.float32

        if not self.offload_to_cpu:
            self.vae = self.vae.to(device).to(vae_dtype)
        else:
            self.vae = self.vae.to("cpu").to(vae_dtype)

        self.vae.eval()
        print(f"  ✓ VAE loaded")

    def _load_text_encoder(self, path: str, device: str):
        """Load text encoder and tokenizer"""
        from transformers import AutoModel, AutoTokenizer

        print(f"\n[4/4] Loading text encoder...")
        self.text_tokenizer = AutoTokenizer.from_pretrained(path)
        self.text_encoder = AutoModel.from_pretrained(path)

        if not self.offload_to_cpu:
            self.text_encoder = self.text_encoder.to(device).to(self.dtype)
        else:
            self.text_encoder = self.text_encoder.to("cpu").to(self.dtype)

        self.text_encoder.eval()
        print(f"  ✓ Text encoder loaded")

    def is_fully_initialized(self) -> bool:
        """Check if all components are loaded"""
        return all([
            self.model is not None,
            self.vae is not None,
            self.text_encoder is not None,
            self.text_tokenizer is not None,
            self.silence_latent is not None,
        ])

    def get_status(self) -> dict:
        """Get current status of components"""
        return {
            "device": self.device,
            "dtype": str(self.dtype),
            "offload_to_cpu": self.offload_to_cpu,
            "model_loaded": self.model is not None,
            "vae_loaded": self.vae is not None,
            "text_encoder_loaded": self.text_encoder is not None,
            "text_tokenizer_loaded": self.text_tokenizer is not None,
            "silence_latent_loaded": self.silence_latent is not None,
            "fully_initialized": self.is_fully_initialized(),
        }


def create_handler_from_wrapper(wrapper: ACEStepWrapper):
    """Create an AceStepHandler-like object from wrapper

    This allows the wrapper to be used with existing acestep code
    that expects an AceStepHandler object.
    """
    from acestep.handler import AceStepHandler

    handler = AceStepHandler()

    # Copy all attributes from wrapper to handler
    for attr in ['device', 'dtype', 'offload_to_cpu', 'model', 'vae',
                 'text_encoder', 'text_tokenizer', 'silence_latent']:
        if hasattr(wrapper, attr):
            setattr(handler, attr, getattr(wrapper, attr))

    # Set additional required attributes
    handler.offload_dit_to_cpu = False
    handler.quantization = None
    handler.compiled = False

    return handler


# Standalone test
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python acestep_wrapper.py <checkpoint_dir> [config_path]")
        print("Example: python acestep_wrapper.py /path/to/models/Ace-Step1.5 acestep-v15-turbo")
        sys.exit(1)

    checkpoint_dir = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else "acestep-v15-turbo"

    print(f"=== ACE-Step Wrapper Test ===")
    print(f"checkpoint_dir: {checkpoint_dir}")
    print(f"config_path: {config_path}")
    print()

    try:
        wrapper = ACEStepWrapper()
        wrapper.initialize(checkpoint_dir, config_path)

        print("\n=== Status ===")
        for key, value in wrapper.get_status().items():
            status = "✓" if value else "✗"
            print(f"  {status} {key}: {value}")

        if wrapper.is_fully_initialized():
            print("\n✅ All components loaded successfully!")
        else:
            print("\n❌ Some components failed to load")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
