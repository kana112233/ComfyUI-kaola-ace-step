"""
ACE-Step Wrapper Module

Provides a clean interface to initialize and use ACE-Step components
independently of ComfyUI. This makes testing and maintenance easier.

Supports two model formats:
1. Standard format (from ACE-Step/ACE-Step-1.5)
2. Split format (from Comfy-Org/ace_step_1.5_ComfyUI_files)
"""

import os
import torch
from typing import Optional, Tuple
from pathlib import Path


def detect_model_format(checkpoint_dir: str) -> Tuple[str, Optional[str]]:
    """Detect whether to use split format or standard format

    Returns:
        ("split", base_dir) for ComfyUI split format at ComfyUI/models/
        ("standard", None) for standard ACE-Step format in ComfyUI/models/Ace-Step1.5/

    Split format checks ComfyUI/models/ for:
    - diffusion_models/acestep_v1.5_turbo.safetensors
    - text_encoders/qwen_*_ace15.safetensors
    - vae/ace_1.5_vae.safetensors
    """
    checkpoint_path = Path(checkpoint_dir)

    # First, check if we should use global ComfyUI/models/ split format
    # This happens when checkpoint_dir is actually the ComfyUI models directory
    # or when the user specifies the models directory directly

    # Check for global split format in ComfyUI/models/
    models_dir = Path(os.environ.get("COMFYUI_MODELS", "models"))
    if models_dir.is_absolute():
        models_dir_abs = models_dir
    else:
        # Relative to current directory
        models_dir_abs = Path.cwd() / models_dir

    # Check if global models has ACE-Step split files
    global_diffusion = models_dir_abs / "diffusion_models" / "acestep_v1.5_turbo.safetensors"
    global_vae = models_dir_abs / "vae" / "ace_1.5_vae.safetensors"
    global_text_enc = models_dir_abs / "text_encoders" / "qwen_1.7b_ace15.safetensors"

    if global_diffusion.exists() or global_vae.exists() or global_text_enc.exists():
        return ("split", str(models_dir_abs))

    # Check for standard format in Ace-Step1.5 subdirectory
    acestep_dir = checkpoint_path / "acestep-v15-turbo"
    standard_vae = checkpoint_path / "vae"
    standard_text_enc = checkpoint_path / "Qwen3-Embedding-0.6B"

    if acestep_dir.exists() or (standard_vae.exists() and standard_text_enc.exists()):
        return ("standard", None)

    # Default to standard
    return ("standard", None)


class ACEStepWrapper:
    """Wrapper for ACE-Step initialization and operations"""

    def __init__(self):
        self.device = None
        self.dtype = None
        self.offload_to_cpu = False
        self.model_format = None
        self.model_base_dir = None  # For split format, this is ComfyUI/models/

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
        prefer_source: Optional[str] = None,
    ):
        """Initialize all ACE-Step components

        Args:
            checkpoint_dir: Directory containing model checkpoints
            config_path: Model config name (e.g., "acestep-v15-turbo")
            device: Device to use ("auto", "cuda", "cpu", "mps", "xpu")
            offload_to_cpu: Whether to offload models to CPU when not in use
            quantization: Quantization type (None, "int8_weight_only", etc.)
            compile_model: Whether to use torch.compile
            prefer_source: Preferred download source ("huggingface", "modelscope", or None for auto)

        Returns:
            self for chaining
        """
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

        # Detect model format
        self.model_format, self.model_base_dir = detect_model_format(checkpoint_dir)
        print(f"[ACE_STEP] Detected model format: {self.model_format}")
        if self.model_base_dir:
            print(f"[ACE_STEP] Split format base directory: {self.model_base_dir}")

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

        # Route to appropriate initialization based on format
        if self.model_format == "split":
            self._initialize_split_format(
                self.model_base_dir, config_path, device, quantization, compile_model, prefer_source
            )
        else:
            self._initialize_standard_format(
                checkpoint_dir, config_path, device, quantization, compile_model, prefer_source
            )

        print(f"[ACE_STEP] Initialization complete")
        return self

    def _initialize_split_format(
        self,
        checkpoint_dir: str,
        config_path: str,
        device: str,
        quantization: Optional[str],
        compile_model: bool,
        prefer_source: Optional[str],
    ):
        """Initialize using ComfyUI split format"""
        from transformers import AutoModel, AutoTokenizer
        from diffusers.models import AutoencoderOobleck

        checkpoint_path = Path(checkpoint_dir)

        # Paths for split format
        dit_path = checkpoint_path / "diffusion_models"
        vae_path = checkpoint_path / "vae"
        text_encoder_path = checkpoint_path / "text_encoders"

        # Ensure models exist or download
        self._ensure_split_models(checkpoint_dir, prefer_source)

        # Load DiT model from split format
        dit_file = dit_path / "acestep_v1.5_turbo.safetensors"
        if not dit_file.exists():
            raise RuntimeError(f"DiT model not found: {dit_file}")

        print(f"\n[1/4] Loading DiT model from: {dit_file}")
        self._load_dit_model_split(dit_file, device, quantization, compile_model)

        # Load silence_latent - may need to be generated or downloaded separately
        # For split format, we might not have this file
        self._ensure_silence_latent(checkpoint_dir, device)

        # Load VAE from split format
        vae_file = vae_path / "ace_1.5_vae.safetensors"
        if not vae_file.exists():
            raise RuntimeError(f"VAE not found: {vae_file}")

        print(f"\n[3/4] Loading VAE from: {vae_file}")
        self._load_vae_split(vae_file, device)

        # Load text encoder from split format
        # Try to find the appropriate qwen model
        # Map common LM sizes to split format files
        text_encoder_files = list(text_encoder_path.glob("qwen_*_ace15.safetensors"))
        if not text_encoder_files:
            raise RuntimeError(f"No text encoder found in: {text_encoder_path}")

        # Use the 1.7B model as default (closest to standard 1.7B)
        # Or select based on user preference
        text_encoder_file = sorted(text_encoder_files, key=lambda x: x.stat().st_size)[1]  # Medium size
        print(f"\n[4/4] Loading text encoder from: {text_encoder_file}")
        self._load_text_encoder_split(text_encoder_file, device)

    def _initialize_standard_format(
        self,
        checkpoint_dir: str,
        config_path: str,
        device: str,
        quantization: Optional[str],
        compile_model: bool,
        prefer_source: Optional[str],
    ):
        """Initialize using standard ACE-Step format"""
        from transformers import AutoModel, AutoTokenizer
        from diffusers.models import AutoencoderOobleck

        checkpoint_path = Path(checkpoint_dir)

        # Build paths for standard format
        model_path = checkpoint_path / config_path
        silence_latent_path = model_path / "silence_latent.pt"
        vae_path = checkpoint_path / "vae"
        text_encoder_path = checkpoint_path / "Qwen3-Embedding-0.6B"

        # Ensure models exist or download
        self._ensure_standard_models(checkpoint_dir, config_path, prefer_source)

        # Verify paths exist
        self._verify_paths(model_path, silence_latent_path, vae_path, text_encoder_path)

        # Load components
        self._load_dit_model(model_path, device, quantization, compile_model)
        self._load_silence_latent(silence_latent_path, device)
        self._load_vae(vae_path, device)
        self._load_text_encoder(text_encoder_path, device)

    def _load_dit_model_split(self, dit_file: Path, device: str,
                               quantization: Optional[str], compile_model: bool):
        """Load DiT model from split format safetensors file"""
        from safetensors.torch import load_file

        # Load from safetensors
        print(f"  Loading from safetensors...")
        try:
            # Try loading as a model directly
            from transformers import AutoModel

            # For split format, the file might be a single tensor or a model checkpoint
            # Try loading with AutoModel first
            try:
                self.model = AutoModel.from_pretrained(
                    str(dit_file.parent),
                    trust_remote_code=True,
                )
            except:
                # If that fails, load the safetensors and create model
                state_dict = load_file(dit_file, device=str(device))
                # This would require knowing the exact model architecture
                # For now, raise error with instructions
                raise RuntimeError(
                    f"Cannot load {dit_file} directly. "
                    f"Please ensure config.json and other model files are present in {dit_file.parent}/"
                )
        except Exception as e:
            raise RuntimeError(f"Failed to load DiT model: {e}")

        self.model = self.model.to(device).to(self.dtype)
        self.model.eval()

        if compile_model:
            self._compile_model(quantization)

        print(f"  ✓ DiT model loaded from split format")

    def _load_vae_split(self, vae_file: Path, device: str):
        """Load VAE from split format safetensors file"""
        from diffusers.models import AutoencoderOobleck

        vae_dtype = torch.bfloat16 if device in ["cuda", "xpu", "mps"] else torch.float32

        print(f"  Loading VAE from safetensors...")
        self.vae = AutoencoderOobleck.from_pretrained(str(vae_file.parent))

        if not self.offload_to_cpu:
            self.vae = self.vae.to(device).to(vae_dtype)
        else:
            self.vae = self.vae.to("cpu").to(vae_dtype)

        self.vae.eval()
        print(f"  ✓ VAE loaded")

    def _load_text_encoder_split(self, text_encoder_file: Path, device: str):
        """Load text encoder from split format safetensors file"""
        from transformers import AutoModel, AutoTokenizer

        print(f"  Loading text encoder from safetensors...")
        # For split format, we need both the model and tokenizer
        # Try loading from the directory
        try:
            self.text_tokenizer = AutoTokenizer.from_pretrained(str(text_encoder_file.parent))
            self.text_encoder = AutoModel.from_pretrained(str(text_encoder_file.parent))
        except Exception as e:
            raise RuntimeError(f"Failed to load text encoder: {e}. You may need to download tokenizer/config files separately.")

        if not self.offload_to_cpu:
            self.text_encoder = self.text_encoder.to(device).to(self.dtype)
        else:
            self.text_encoder = self.text_encoder.to("cpu").to(self.dtype)

        self.text_encoder.eval()
        print(f"  ✓ Text encoder loaded")

    def _ensure_silence_latent(self, checkpoint_dir: str, device: str):
        """Ensure silence_latent exists, generate or download if missing"""
        checkpoint_path = Path(checkpoint_dir)

        # Check for split format location (ComfyUI/models/diffusion_models/)
        # or standard format location (Ace-Step1.5/acestep-v15-turbo/)
        silence_paths = [
            checkpoint_path / "diffusion_models" / "silence_latent.pt",  # split format
            checkpoint_path / "acestep-v15-turbo" / "silence_latent.pt",  # standard format
        ]

        for silence_path in silence_paths:
            if silence_path.exists():
                self.silence_latent = torch.load(silence_path).transpose(1, 2)
                self.silence_latent = self.silence_latent.to(device).to(self.dtype)
                print(f"  ✓ silence_latent loaded from: {silence_path}")
                return

        # If not found, generate a simple silence latent
        print(f"  [WARNING] silence_latent.pt not found, generating a basic one...")
        # Create a basic silence latent (all zeros with correct shape)
        # Typical shape for ACE-Step: [1, 750, 2048] for 5Hz at 30 seconds
        self.silence_latent = torch.zeros(1, 750, 2048, dtype=self.dtype).to(device)
        print(f"  ✓ Generated silence_latent (shape: {self.silence_latent.shape})")

    def _ensure_split_models(self, checkpoint_dir: str, prefer_source: Optional[str] = None):
        """Ensure split format models exist

        Checks for all required files:
        - diffusion_models/acestep_v1.5_turbo.safetensors
        - vae/ace_1.5_vae.safetensors
        - text_encoders/qwen_*_ace15.safetensors (0.6B or 1.7B)

        Format from: https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files
        """
        checkpoint_path = Path(checkpoint_dir)
        missing_files = []

        # Check DiT model
        dit_path = checkpoint_path / "diffusion_models" / "acestep_v1.5_turbo.safetensors"
        if not dit_path.exists():
            missing_files.append(("DiT", dit_path, "split_files/diffusion_models/acestep_v1.5_turbo.safetensors"))

        # Check VAE
        vae_path = checkpoint_path / "vae" / "ace_1.5_vae.safetensors"
        if not vae_path.exists():
            missing_files.append(("VAE", vae_path, "split_files/vae/ace_1.5_vae.safetensors"))

        # Check text encoder (0.6B or 1.7B)
        text_enc_dir = checkpoint_path / "text_encoders"
        text_enc_06b = text_enc_dir / "qwen_0.6b_ace15.safetensors"
        text_enc_17b = text_enc_dir / "qwen_1.7b_ace15.safetensors"

        if not text_enc_06b.exists() and not text_enc_17b.exists():
            missing_files.append(("Text Encoder", text_enc_06b, "split_files/text_encoders/qwen_1.7b_ace15.safetensors"))

        # Report missing files
        if missing_files:
            print(f"\n[ACE_STEP] Missing split format models:")
            for name, path, remote_path in missing_files:
                print(f"  ✗ {name}: {path}")

            print(f"\n[ACE_STEP] To download, use:")
            for name, path, remote_path in missing_files:
                print(f"  huggingface-cli download Comfy-Org/ace_step_1.5_ComfyUI_files \\")
                print(f"    {remote_path} \\")
                print(f"    --local-dir {checkpoint_dir}/{remote_path.split('/')[0]}/")

            raise RuntimeError(f"Missing {len(missing_files)} required model file(s). Please download split format models first.")

        print(f"[ACE_STEP] ✓ All split format models found")

    def _ensure_standard_models(self, checkpoint_dir: str, config_path: str, prefer_source: Optional[str] = None):
        """Ensure standard format models are downloaded"""
        from acestep.model_downloader import ensure_main_model, ensure_dit_model

        checkpoint_dir_path = Path(checkpoint_dir)

        print(f"[ACE_STEP] Checking models in: {checkpoint_dir}")

        # Ensure main model (contains vae, text_encoder, and turbo dit model)
        if not os.path.exists(os.path.join(checkpoint_dir, "vae")):
            print("[ACE_STEP] VAE not found, downloading main model...")
            success, msg = ensure_main_model(checkpoint_dir_path, prefer_source=prefer_source)
            if not success:
                raise RuntimeError(f"Failed to download main model: {msg}")
            print(f"[ACE_STEP] {msg}")

        # Ensure DiT model (if not turbo which is part of main)
        if config_path != "acestep-v15-turbo":
            if not os.path.exists(os.path.join(checkpoint_dir, config_path)):
                print(f"[ACE_STEP] DiT model '{config_path}' not found, downloading...")
                success, msg = ensure_dit_model(config_path, checkpoint_dir_path, prefer_source=prefer_source)
                if not success:
                    raise RuntimeError(f"Failed to download DiT model: {msg}")
                print(f"[ACE_STEP] {msg}")

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
            "model_format": self.model_format,
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

    # Set handler.config from model.config, ensuring is_turbo attribute exists
    if wrapper.model is not None and hasattr(wrapper.model, 'config'):
        handler.config = wrapper.model.config
        # Ensure is_turbo attribute exists (may not be in all model versions)
        if not hasattr(handler.config, 'is_turbo'):
            # Check if model name contains 'turbo' to determine is_turbo
            model_name = getattr(handler.config, 'name_or_path', '') or ''
            handler.config.is_turbo = 'turbo' in model_name.lower()
    else:
        # Fallback: create a simple config object with is_turbo
        class SimpleConfig:
            is_turbo = False  # Default to False for safety
        handler.config = SimpleConfig()

    return handler


# Standalone test
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python acestep_wrapper.py <checkpoint_dir> [config_path]")
        print("Example: python acestep_wrapper.py /path/to/models/Ace-Step1.5 acestep-v15-turbo")
        print("\nSupported formats:")
        print("  - Standard: checkpoints/acestep-v15-turbo/, vae/, Qwen3-Embedding-0.6B/")
        print("  - Split: diffusion_models/, text_encoders/, vae/")
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
