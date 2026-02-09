"""
ACE-Step 1.5 Music Generation Nodes
"""

import os
import ctypes
import sys
import glob

# --------------------------------------------------------------------------------
# Critical TLS Fix for ComfyUI + Conda
# The "Inconsistency detected by ld.so" error is caused by OpenMP library conflicts.
# We attempt to force-load libgomp via ctypes before any other library loads it.
# --------------------------------------------------------------------------------
def _force_load_libgomp():
    try:
        # 1. Try generic path
        ctypes.CDLL("libgomp.so.1", mode=ctypes.RTLD_GLOBAL)
    except OSError:
        # 2. Try to find it in the current conda environment (based on sys.executable)
        try:
            conda_prefix = os.environ.get("CONDA_PREFIX")
            if not conda_prefix and "envs" in sys.executable:
                 # Infer from python path if CONDA_PREFIX is missing
                 conda_prefix = sys.executable.split("/bin/python")[0]
            
            if conda_prefix:
                lib_paths = glob.glob(os.path.join(conda_prefix, "lib", "libgomp.so.1"))
                if lib_paths:
                    ctypes.CDLL(lib_paths[0], mode=ctypes.RTLD_GLOBAL)
        except Exception:
            pass

_force_load_libgomp()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# --------------------------------------------------------------------------------

import sys
import torch
import json
import numpy as np
import tempfile
import soundfile as sf
from typing import Dict, List, Optional, Tuple, Any
import folder_paths

comfy_path = folder_paths.__file__.replace("__init__.py", "")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "acestep_repo"))
sys.path.insert(0, os.path.dirname(__file__))  # For acestep_wrapper

# Import our wrapper module
from acestep_wrapper import ACEStepWrapper, create_handler_from_wrapper

try:
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    import acestep.inference as acestep_inference
    from acestep.inference import (
        GenerationParams,   
        GenerationConfig,
        create_sample,
        format_sample,
        understand_music,
    )
    # Access generate_music through module to ensure monkey patch works
    generate_music = acestep_inference.generate_music
    ACESTEP_AVAILABLE = True
    
    # Helper to instantiate GenerationParams safely if upstream version changes
    def create_generation_params(**kwargs):
        # Filter kwargs to only include those accepted by the upstream GenerationParams dataclass
        import inspect
        sig = inspect.signature(GenerationParams)
        valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return GenerationParams(**valid_kwargs)

except ImportError:
    ACESTEP_AVAILABLE = False
    print("ACE-Step not installed. Please install it first: pip install acestep")

# --------------------------------------------------------------------------------
# MonkeyPatch process_src_audio to use soundfile instead of torchaudio
# This is to avoid format errors or libtorchcodec issues with torchaudio in some envs
# --------------------------------------------------------------------------------
if ACESTEP_AVAILABLE:
    def patched_process_src_audio(self, audio_file) -> Optional[torch.Tensor]:
        if audio_file is None:
            return None
            
        try:
            # Load audio file using soundfile (more robust)
            audio_np, sr = sf.read(audio_file, dtype='float32')
            
            # Convert to torch: [samples, channels] or [samples] -> [channels, samples]
            if audio_np.ndim == 1:
                audio = torch.from_numpy(audio_np).unsqueeze(0)
            else:
                audio = torch.from_numpy(audio_np.T)
            
            # Normalize to stereo 48kHz (using internal helper if available, or manual)
            # The original handler has _normalize_audio_to_stereo_48k
            if hasattr(self, '_normalize_audio_to_stereo_48k'):
                 audio = self._normalize_audio_to_stereo_48k(audio, sr)
            else:
                 # Fallback normalization logic just in case
                 if audio.shape[0] == 1:
                    audio = torch.cat([audio, audio], dim=0)
                 audio = audio[:2]
                 if sr != 48000:
                    import torchaudio.transforms as T
                    # Resample needs channels first
                    resampler = T.Resample(sr, 48000)
                    audio = resampler(audio)
                 audio = torch.clamp(audio, -1.0, 1.0)
            
            return audio
            
        except Exception as e:
            print(f"[patched_process_src_audio] Error processing source audio: {e}")
            import traceback
            traceback.print_exc()
            return None

    # Apply the patch
    AceStepHandler.process_src_audio = patched_process_src_audio

    # --------------------------------------------------------------------------------
    # MonkeyPatch LLM and Inference to fix missing lyrics issue
    # --------------------------------------------------------------------------------
    # MonkeyPatch LLM and Inference to fix missing lyrics and language consistency
    # --------------------------------------------------------------------------------
    
    # 1. Patch generate_music to handle instrumental logic and bridge params
    import acestep.inference
    _original_generate_music = acestep.inference.generate_music
    def patched_generate_music(dit_handler, llm_handler, params, config, **kwargs):
        # Handle instrumental -> lyrics convention
        # If instrumental is True, we force lyrics to be "[Instrumental]"
        # This aligns with upstream logic (often implicit) for instrumental generation
        if params.instrumental:
            print(f"[ACE_STEP] Instrumental mode: Setting lyrics to '[Instrumental]'")
            params.lyrics = "[Instrumental]"

        # Bridge vocal_language to llm_handler for use in generate_with_stop_condition
        if llm_handler is not None:
            # We bridge this so generate_with_stop_condition can use it
            llm_handler._patch_vocal_language = params.vocal_language
            
            # Bridge sampling params
            lm_top_k = getattr(params, 'lm_top_k', None)
            lm_top_p = getattr(params, 'lm_top_p', None)
            llm_handler._patch_top_k = lm_top_k
            llm_handler._patch_top_p = lm_top_p
        
        lm_temp = getattr(params, 'lm_temperature', 0.0)
        
        print(f"[ACE_STEP] Generation Start:")
        print(f"  - Task: {params.task_type}")
        print(f"  - Language: {params.vocal_language}")
        print(f"  - Instrumental: {params.instrumental}")
        print(f"  - LLM Config: temp={lm_temp}, top_k={getattr(params, 'lm_top_k', None)}, top_p={getattr(params, 'lm_top_p', None)}")
        
        if config and hasattr(config, 'seeds'):
             if isinstance(config.seeds, int):
                  config.seeds = [config.seeds]

        print(f"  - DiT Config: steps={params.inference_steps}, seed={config.seeds}")
        return _original_generate_music(dit_handler, llm_handler, params, config, **kwargs)
 
    acestep.inference.generate_music = patched_generate_music
    # CRITICAL: Update module-level reference to use patched version
    generate_music = patched_generate_music

    # 2. Patch generate_with_stop_condition to inject vocal_language into user_metadata
    _original_generate_with_stop_condition = LLMHandler.generate_with_stop_condition
    def patched_generate_with_stop_condition(self, caption: str, lyrics: str, *args, **kwargs):
        # Inject vocal_language into user_metadata
        vocal_language = getattr(self, '_patch_vocal_language', None)
        if vocal_language and vocal_language != 'unknown':
            user_metadata = kwargs.get('user_metadata', {})
            # user_metadata can be None coming from upstream
            if user_metadata is None: user_metadata = {}
            
            if 'language' not in user_metadata:
                print(f"[MONKEY PATCH] Injecting language '{vocal_language}' into user_metadata")
                user_metadata['language'] = vocal_language
                kwargs['user_metadata'] = user_metadata
        
        # Bridge sampling params
        top_k = getattr(self, '_patch_top_k', None)
        top_p = getattr(self, '_patch_top_p', None)
        
        # Override if passed explicitly in kwargs
        if top_k is not None and 'top_k' not in kwargs:
            kwargs['top_k'] = top_k
        if top_p is not None and 'top_p' not in kwargs:
            kwargs['top_p'] = top_p
            
        return _original_generate_with_stop_condition(self, caption, lyrics, *args, **kwargs)

    LLMHandler.generate_with_stop_condition = patched_generate_with_stop_condition



    # --------------------------------------------------------------------------------
    # MonkeyPatch LLMHandler
    # --------------------------------------------------------------------------------
    _original_build_formatted_prompt_for_inspiration = LLMHandler.build_formatted_prompt_for_inspiration
    def patched_build_formatted_prompt_for_inspiration(self, query, **kwargs):
        # Use bridged attributes if they were set by patched_generate_music or create_sample
        vocal_language = getattr(self, "_patch_vocal_language", "unknown")
        instrumental = getattr(self, "_patch_instrumental", False)
        
        # Inject into prompt
        # We manually build the suffix to ensure LLM follows the requested language
        if vocal_language and vocal_language != "unknown":
            query += f"\n\nIMPORTANT: The song must be in {vocal_language} language."
        if instrumental:
             query += "\n\nIMPORTANT: This must be an instrumental track (no vocals). [Instrumental]"
        
        return _original_build_formatted_prompt_for_inspiration(self, query, **kwargs)
    
    LLMHandler.build_formatted_prompt_for_inspiration = patched_build_formatted_prompt_for_inspiration

    _original_create_sample_from_query = LLMHandler.create_sample_from_query
    def patched_create_sample_from_query(self, query, **kwargs):
        # Bridge params to inspiration
        self._patch_vocal_language = kwargs.get("vocal_language", "unknown")
        self._patch_instrumental = kwargs.get("instrumental", False)
        return _original_create_sample_from_query(self, query, **kwargs)
    
    LLMHandler.create_sample_from_query = patched_create_sample_from_query
    print("Monkeypatched AceStepHandler.process_src_audio to use soundfile")

    # --------------------------------------------------------------------------------
    # MonkeyPatch convert_src_audio_to_codes to fix multi-codebook flattening issue
    # The DiT tokenizer might return multiple codebooks (e.g. 8), but 5Hz LM expects
    # only the first semantic codebook. Flattening all of them causes garbage input.
    # --------------------------------------------------------------------------------
    def patched_convert_src_audio_to_codes(self, audio_file) -> str:
        if audio_file is None:
            return "❌ Please upload source audio first"
        
        if self.model is None or self.vae is None:
            return "❌ Model not initialized. Please initialize the service first."
        
        try:
            # Process audio file (uses patched_process_src_audio internally if patched)
            processed_audio = self.process_src_audio(audio_file)
            if processed_audio is None:
                return "❌ Failed to process audio file"
            
            # Encode audio to latents using VAE
            with torch.no_grad():
                with self._load_model_context("vae"):
                    # Check if audio is silence
                    if self.is_silence(processed_audio.unsqueeze(0)):
                        return "❌ Audio file appears to be silent"
                    
                    # Encode to latents using helper method
                    latents = self._encode_audio_to_latents(processed_audio)  # [T, d]
                
                # Create attention mask for latents
                attention_mask = torch.ones(latents.shape[0], dtype=torch.bool, device=self.device)
                
                # Tokenize latents to get code indices
                with self._load_model_context("model"):
                    # Prepare latents for tokenize: [T, d] -> [1, T, d]
                    hidden_states = latents.unsqueeze(0)  # [1, T, d]
                    
                    # Call tokenize method
                    # tokenize returns: (quantized, indices, attention_mask)
                    # Note: indices shape is typically [1, T, num_quantizers]
                    _, indices, _ = self.model.tokenize(hidden_states, self.silence_latent, attention_mask.unsqueeze(0))
                    
                    # FIX: If multiple codebooks, take only the first one (semantic codes)
                    if indices.dim() == 3 and indices.shape[-1] > 1:
                        # print(f"[debug_nodes] Detected multiple codebooks: {indices.shape}. Selecting first one.")
                        indices = indices[..., 0]
                    
                    # Format indices as code string
                    # indices shape now: [1, T_5Hz]
                    indices_flat = indices.flatten().cpu().tolist()
                    codes_string = "".join([f"<|audio_code_{idx}|>" for idx in indices_flat])
                    
                    return codes_string
                    
        except Exception as e:
            error_msg = f"❌ Error converting audio to codes: {str(e)}"
            # import traceback
            # traceback.print_exc()
            return error_msg

    AceStepHandler.convert_src_audio_to_codes = patched_convert_src_audio_to_codes
    print("Monkeypatched AceStepHandler.convert_src_audio_to_codes to select first codebook")


# Register ACE-Step model directory with ComfyUI
# Models should be placed in: ComfyUI/models/Ace-Step1.5/
ACESTEP_MODEL_NAME = "Ace-Step1.5"
if ACESTEP_AVAILABLE:
    folder_paths.add_model_folder_path(ACESTEP_MODEL_NAME, os.path.join(folder_paths.models_dir, ACESTEP_MODEL_NAME))

def get_acestep_models():
    if not ACESTEP_AVAILABLE:
        return []
    model_dir = os.path.join(folder_paths.models_dir, ACESTEP_MODEL_NAME)
    if not os.path.exists(model_dir):
        return []
    # List subdirectories
    models = [name for name in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, name))]
    # Ensure defaults are present if not found (for UI stability)
    defaults = ["acestep-5Hz-lm-1.7B", "acestep-v15-turbo"]
    for d in defaults:
        if d not in models:
            models.append(d)
    return sorted(list(set(models)))

def get_acestep_checkpoints():
    if not ACESTEP_AVAILABLE:
        return [""]
    paths = folder_paths.get_folder_paths(ACESTEP_MODEL_NAME)
    if not paths:
        # Fallback to default path if not found
        return [ACESTEP_MODEL_NAME]
    
    result = []
    for p in paths:
        if os.path.exists(p):
            # Use basename for display
            name = os.path.basename(p.rstrip('/\\'))
            if not name:  # If path ends with separator
                name = os.path.basename(os.path.dirname(p))
            result.append(name if name else ACESTEP_MODEL_NAME)
    
    unique_results = sorted(list(set(result)))
    return unique_results if unique_results else [ACESTEP_MODEL_NAME]

# Cache for checkpoint path resolution
_checkpoint_path_cache = {}

def resolve_checkpoint_path(name: str) -> str:
    """Resolve a checkpoint name to its full path."""
    global _checkpoint_path_cache
    if name in _checkpoint_path_cache:
        return _checkpoint_path_cache[name]
    
    # Try to find the full path
    paths = folder_paths.get_folder_paths(ACESTEP_MODEL_NAME)
    for p in paths:
        if os.path.basename(p.rstrip('/\\')) == name or name in p:
            _checkpoint_path_cache[name] = p
            return p
    
    # Fallback: assume it's relative to models_dir
    full_path = os.path.join(folder_paths.models_dir, name)
    if os.path.exists(full_path):
        _checkpoint_path_cache[name] = full_path
        return full_path
    
    # Last fallback: return the name as-is (might be absolute path already)
    return name

def get_available_peft_loras():
    lora_paths = []
    # Search paths: 
    # 1. ComfyUI/models/loras
    # 2. ComfyUI/models/Ace-Step1.5/loras
    search_paths = [
        os.path.join(folder_paths.models_dir, "loras"),
        os.path.join(folder_paths.models_dir, ACESTEP_MODEL_NAME, "loras")
    ]
    
    # Standardize models_dir for relpath
    base_dir = os.path.abspath(folder_paths.models_dir)
    
    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue
        # PEFT LoRA is a directory containing adapter_config.json
        for root, dirs, files in os.walk(search_path):
            if "adapter_config.json" in files:
                # Found a LoRA directory
                # Use path relative to models_dir for cleaner display
                abs_root = os.path.abspath(root)
                try:
                    rel_path = os.path.relpath(abs_root, base_dir)
                    # Normalize to forward slashes for cross-platform compatibility
                    rel_path = rel_path.replace("\\", "/")
                    lora_paths.append(rel_path)
                except ValueError:
                    # In case they are on different drives
                    lora_paths.append(abs_root)
    
    if not lora_paths:
        return ["None"]
        
    return sorted(list(set(lora_paths)))


class ACE_STEP_BASE:
    """Base class for ACE-Step nodes with handler management"""

    def __init__(self):
        self.dit_handler = None
        self.llm_handler = None
        self.handlers_initialized = False

    @staticmethod
    def auto_detect_device() -> str:
        """
        Auto-detect the best available device.
        Priority: CUDA > MPS > XPU > CPU
        """
        import torch
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            return "xpu"
        else:
            return "cpu"

    def initialize_handlers(
        self,
        checkpoint_dir: str,
        config_path: str,
        lm_model_path: str,
        device: str = "auto",
        offload_to_cpu: bool = False,
        lora_info: Optional[Dict[str, Any]] = None,
        quantization: Optional[str] = None,
        compile_model: bool = False,
    ):
        """Initialize ACE-Step handlers if not already initialized"""
        
        # LoRA - Quantization conflict check
        # PEFT is incompatible with torchao quantization (int8_weight_only etc)
        if lora_info and lora_info.get("path") and quantization and quantization != "None":
            print(f"[initialize_handlers] WARNING: LoRA and Quantization are incompatible. Disabling quantization for LoRA compatibility.")
            quantization = None
            # If quantization is disabled, we can still compile, but upstream initialize_service 
            # might expect compile_model=True for quantization. 
            # If quantization is None, compile_model is optional quality/speed trade-off.

        if self.handlers_initialized:
            # Check if handlers are truly initialized (model loaded)
            if self.dit_handler and getattr(self.dit_handler, "model", None) is not None:
                # Check if quantization status matches
                # If current handler has different quantization, we need to re-initialize
                # because quantization is applied at load time
                if getattr(self.dit_handler, "quantization", None) != (None if quantization == "None" else quantization):
                    print(f"[initialize_handlers] Quantization changed from {self.dit_handler.quantization} to {quantization}. Re-initializing.")
                    self.handlers_initialized = False
                else:
                    # Clear CUDA cache before reusing handlers to prevent OOM
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Update LoRA state even if handlers are reused
                    self._update_lora_state(self.dit_handler, lora_info)
                    return self.dit_handler, self.llm_handler
            
            if self.handlers_initialized: # still?
                print("[initialize_handlers] Handlers marked initialized but model is None. Re-initializing.")
                self.handlers_initialized = False

        if not ACESTEP_AVAILABLE:
            raise RuntimeError("ACE-Step is not installed. Please install it first.")

        # Clear CUDA cache before initialization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Auto-detect device if "auto" is specified
        if device == "auto":
            device = self.auto_detect_device()
            print(f"[ACE_STEP] Auto-detected device: {device}")
        try:
            # Resolve checkpoint path (converts relative name to full path)
            checkpoint_dir = resolve_checkpoint_path(checkpoint_dir)
            
            # Use ComfyUI's model directory if checkpoint_dir is not provided
            if not checkpoint_dir or checkpoint_dir == "./checkpoints":
                checkpoint_dir = folder_paths.get_folder_paths(ACESTEP_MODEL_NAME)[0]

            # Check if checkpoint directory exists
            if not os.path.exists(checkpoint_dir):
                raise RuntimeError(
                    f"Model directory not found: {checkpoint_dir}\n"
                    f"Please download ACE-Step models to this location.\n"
                    f"See https://github.com/ACE-Step/Ace-Step1.5"
                )

            # Initialize DiT handler - bypass upstream's initialize_service to avoid
            # the hardcoded "checkpoints" subdirectory requirement
            self.dit_handler = AceStepHandler()
            self._initialize_dit_service_direct(
                dit_handler=self.dit_handler,
                checkpoint_dir=checkpoint_dir,
                config_path=config_path,
                device=device,
                offload_to_cpu=offload_to_cpu,
                quantization=None if quantization == "None" else quantization,
                compile_model=compile_model,
            )
            print(f"[ACE_STEP] DiT service initialized. Quantization: {self.dit_handler.quantization}")
            # Initialize LLM handler (pass dtype from DiT handler)
            # Use "pt" backend instead of "vllm" to avoid process group conflicts with ComfyUI
            self.llm_handler = LLMHandler()
            llm_status, llm_success = self.llm_handler.initialize(
                checkpoint_dir=checkpoint_dir,
                lm_model_path=lm_model_path,
                backend="pt",  # Use PyTorch backend to avoid vLLM conflicts
                device=device,
                offload_to_cpu=offload_to_cpu,
                dtype=self.dit_handler.dtype,  # Critical: pass dtype from DiT handler
            )
            print(f"[ACE_STEP] LLM handler init: {llm_status}")
            if not llm_success:
                raise RuntimeError(f"LLM initialization failed: {llm_status}")

            # Apply LoRA for fresh handler
            self._update_lora_state(self.dit_handler, lora_info)

            self.handlers_initialized = True
            return self.dit_handler, self.llm_handler

        except Exception as e:
            raise RuntimeError(f"Failed to initialize ACE-Step handlers: {str(e)}")

    def _update_lora_state(self, handler, lora_info):
        """Helper to load/unload LoRA based on lora_info"""
        if not handler:
            return

        if lora_info and lora_info.get("path"):
            path = lora_info["path"]
            scale = lora_info.get("scale", 1.0)
            
            # If a different LoRA is loaded? load_lora handles replacement usually?
            # AceStepHandler implementation of load_lora: "Restore base decoder before loading new LoRA"
            # So just calling load_lora works fine.
            
            # Optimization: check if same LoRA is already loaded? 
            # Handler doesn't track current path.
            # We'll just load it. It handles unloading internally.
            
            print(f"[ACE_STEP] Loading LoRA API: {path} (scale: {scale})")
            handler.load_lora(path)
            handler.set_lora_scale(scale)
            
            # When scale is 0, completely disable LoRA adapter layers
            # to prevent structural interference with the model
            if scale == 0:
                handler.set_use_lora(False)
                print(f"[ACE_STEP] LoRA scale=0, adapter layers disabled")
            else:
                handler.set_use_lora(True)
                print(f"[ACE_STEP] LoRA active with scale {scale}")
        else:
            # Ensure no LoRA is loaded if not requested
            if handler.lora_loaded:
                print(f"[ACE_STEP] Unloading LoRA")
                handler.unload_lora()

    def _initialize_dit_service_direct(
        self,
        dit_handler,
        checkpoint_dir: str,
        config_path: str,
        device: str,
        offload_to_cpu: bool = False,
        quantization: Optional[str] = None,
        compile_model: bool = False,
    ):
        """Initialize DiT handler using ACEStepWrapper

        This avoids the hardcoded "checkpoints" subdirectory requirement and
        works with ComfyUI's model directory structure.
        """
        # Use wrapper to initialize all components
        wrapper = ACEStepWrapper()
        wrapper.initialize(
            checkpoint_dir=checkpoint_dir,
            config_path=config_path,
            device=device,
            offload_to_cpu=offload_to_cpu,
            quantization=quantization,
            compile_model=compile_model,
        )

        # Copy wrapper attributes to dit_handler
        for attr in ['device', 'dtype', 'offload_to_cpu', 'model', 'vae',
                     'text_encoder', 'text_tokenizer', 'silence_latent']:
            if hasattr(wrapper, attr):
                setattr(dit_handler, attr, getattr(wrapper, attr))

        # Set additional required attributes
        dit_handler.offload_dit_to_cpu = False
        dit_handler.config = dit_handler.model.config
        dit_handler.quantization = quantization

        print(f"[ACE_STEP] DiT handler initialized successfully")


class ACE_STEP_TEXT_TO_MUSIC(ACE_STEP_BASE):
    """Generate music from text description"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "caption": ("STRING", {"default": "", "multiline": True, "tooltip": "Text prompt or natural language description for music generation."}),
                "checkpoint_dir": (get_acestep_checkpoints(), {"default": get_acestep_checkpoints()[0], "tooltip": "Directory containing ACE-Step model weights (DiT model)."}),
                "config_path": (get_acestep_models(), {"default": "acestep-v15-turbo", "tooltip": "Specific model configuration to use (e.g., v1.5 turbo)."}),
                "lm_model_path": (get_acestep_models(), {"default": "acestep-5Hz-lm-1.7B", "tooltip": "Path to the language model used for generating lyrics and metadata."}),
                "duration": ("FLOAT", {"default": 30.0, "min": 10.0, "max": 600.0, "tooltip": "Target duration of the generated music in seconds."}),
                "batch_size": ("INT", {"default": 2, "min": 1, "max": 8, "tooltip": "Number of audio samples to generate in a single batch."}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFffffffff, "control_after_generate": True, "tooltip": "Random seed for reproducibility. Set to -1 for random generation."}),
                "inference_steps": ("INT", {"default": 8, "min": 1, "max": 64, "tooltip": "Number of diffusion steps. Higher values (e.g., 25-50) improve quality but are slower."}),
                "device": (["auto", "cuda", "cpu", "mps", "xpu"], {"default": "auto", "tooltip": "Computing platform to run the model on."}),
            },
            "optional": {
                "lora_info": ("ACE_STEP_LORA_INFO", {"tooltip": "Optional LoRA model information for style fine-tuning."}),
                "lyrics": ("STRING", {"default": "", "multiline": True, "tooltip": "Song lyrics. Leave empty for automatic generation by the language model."}),
                "bpm": ("INT", {"default": 0, "min": 0, "max": 300, "tooltip": "Beats per minute. 0 for automatic detection."}),
                "keyscale": ("STRING", {"default": "", "tooltip": "Musical key and scale (e.g., C Major)."}),
                "timesignature": ("STRING", {"default": "", "tooltip": "Musical time signature (e.g., 4/4)."}),
                "vocal_language": ("STRING", {"default": "unknown", "tooltip": "Vocal language (e.g., zh, en, ja, auto, unknown). Accepts string input from CreateSample node."}),
                "instrumental": ("BOOLEAN", {"default": False, "tooltip": "Whether to generate instrumental music only (no vocals)."}),
                "guidance_scale": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 15.0, "tooltip": "Strength of prompt following."}),
                "shift": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 5.0, "tooltip": "Sequence length scaling factor, default is 1.0."}),
                "thinking": ("BOOLEAN", {"default": True, "tooltip": "Whether to show the language model's Chain-of-Thought reasoning."}),
                "lm_temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "tooltip": "Sampling temperature for the language model. 0.0 is most stable (recommended)."}),
                "quantization": (["None", "int8_weight_only"], {"default": "None", "tooltip": "Model quantization (e.g., int8). Reduces VRAM usage but requires torchao and compile_model=True. Incompatible with LoRA."}),
                "compile_model": ("BOOLEAN", {"default": False, "tooltip": "Whether to use torch.compile to optimize the model. Required for quantization. Slow on first run but faster afterwards."}),
                "audio_format": (["flac", "mp3", "wav"], {"default": "flac", "tooltip": "Output audio file format."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio", "audio_path", "metadata")
    FUNCTION = "generate"
    CATEGORY = "Audio/ACE-Step"
    OUTPUT_NODE = True

    def generate(
        self,
        caption: str,
        checkpoint_dir: str,
        config_path: str,
        lm_model_path: str,
        duration: float,
        batch_size: int,
        seed: int,
        inference_steps: int,
        device: str,
        lyrics: str = "",
        bpm: int = 0,
        keyscale: str = "",
        timesignature: str = "",
        vocal_language: str = "unknown",
        instrumental: bool = False,
        guidance_scale: float = 7.0,
        shift: float = 1.0,
        thinking: bool = True,
        lm_temperature: float = 0.0,
        quantization: str = "None",
        compile_model: bool = False,
        audio_format: str = "flac",
        lora_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], str, str]:
        # Clear CUDA cache before generation to prevent OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Initialize handlers
        dit_handler, llm_handler = self.initialize_handlers(
            checkpoint_dir=checkpoint_dir,
            config_path=config_path,
            lm_model_path=lm_model_path,
            device=device,
            lora_info=lora_info,
            quantization=quantization,
            compile_model=compile_model,
        )

        # Prepare generation parameters
        params = create_generation_params(
            task_type="text2music",
            caption=caption,
            lyrics=lyrics if lyrics else "",
            instrumental=instrumental,
            duration=duration if duration > 0 else -1.0,
            bpm=bpm if bpm > 0 else None,
            keyscale=keyscale,
            timesignature=timesignature,
            vocal_language=vocal_language,
            inference_steps=inference_steps,
            seed=seed,
            guidance_scale=guidance_scale,
            shift=shift,
            thinking=thinking,
            # Disable all CoT features when thinking=False (required for LoRA compatibility)
            use_cot_caption=thinking,
            use_cot_language=thinking,
            use_cot_metas=thinking,
            lm_temperature=lm_temperature,
            lm_top_k=0,  # These might be filtered out if not in upstream
            lm_top_p=1.0,
        )

        # Prepare generation config
        config = GenerationConfig(
            batch_size=batch_size,
            use_random_seed=(seed == -1),
            audio_format=audio_format,
        )

        # Generate music
        output_dir = folder_paths.get_output_directory()
        result = generate_music(dit_handler, llm_handler, params, config, save_dir=output_dir)

        if not result.success:
            raise RuntimeError(f"Generation failed: {result.error}")

        # Get the first audio from batch
        audio_data = result.audios[0]
        audio_path = audio_data["path"]
        audio_tensor = audio_data["tensor"]
        sample_rate = audio_data["sample_rate"]

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Prepare ComfyUI audio format
        # audio_tensor shape: [channels, samples] -> [1, channels, samples]
        audio_output = {
            "waveform": audio_tensor.cpu().unsqueeze(0),
            "sample_rate": sample_rate,
        }

        # Prepare metadata
        metadata = json.dumps(
            {
                "caption": audio_data["params"].get("caption", caption),
                "bpm": audio_data["params"].get("bpm", bpm),
                "duration": duration,
                "seed": audio_data["params"].get("seed", seed),
                "keyscale": keyscale,
                "sample_rate": sample_rate,
            },
            indent=2,
        )

        return audio_output, audio_path, metadata


class ACE_STEP_COVER(ACE_STEP_BASE):
    """Generate cover version of an audio file"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "src_audio": ("AUDIO", {"tooltip": "Source audio to be covered (remade in new style)."}),
                "caption": ("STRING", {"default": "", "multiline": True, "tooltip": "Description of the target musical style (e.g., 'A jazz version of this song')."}),
                "checkpoint_dir": (get_acestep_checkpoints(), {"default": get_acestep_checkpoints()[0], "tooltip": "Directory containing ACE-Step model weights."}),
                "config_path": (get_acestep_models(), {"default": "acestep-v15-turbo", "tooltip": "Model configuration (turbo is faster)."}),
                "lm_model_path": (get_acestep_models(), {"default": "acestep-5Hz-lm-1.7B", "tooltip": "Language model for metadata/lyrics."}),
                "audio_cover_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Strength of preserving original audio structure. Use LOWER (0.1-0.3) for more style change."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4, "tooltip": "Number of variations."}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "Random seed."}),
                "inference_steps": ("INT", {"default": 8, "min": 1, "max": 100, "tooltip": "Steps. Turbo: 8, Base: 50+."}),
                "guidance_scale": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 15.0, "tooltip": "CFG strength."}),
                "device": (["auto", "cuda", "cpu", "mps", "xpu"], {"default": "auto", "tooltip": "Processing platform."}),
            },
            "optional": {
                "lyrics": ("STRING", {"default": "", "multiline": True, "tooltip": "Optional lyrics."}),
                "vocal_language": (["unknown", "auto", "en", "zh", "ja", "ko", "es", "fr", "de", "ru", "pt", "it", "bn"], {"default": "unknown", "tooltip": "Target language."}),
                "instrumental": ("BOOLEAN", {"default": False, "tooltip": "Instrumental mode."}),
                "bpm": ("INT", {"default": 0, "min": 0, "max": 300, "tooltip": "Target BPM (0 for keep original)."}),
                "keyscale": ("STRING", {"default": "", "tooltip": "Musical key."}),
                "timesignature": ("STRING", {"default": "", "tooltip": "Time signature."}),
                "use_adg": ("BOOLEAN", {"default": False, "tooltip": "Adaptive Dual Guidance."}),
                "thinking": ("BOOLEAN", {"default": True, "tooltip": "Show LLM reasoning."}),
                "audio_format": (["flac", "mp3", "wav"], {"default": "flac", "tooltip": "Output format."}),
                "lora_info": ("ACE_STEP_LORA_INFO", {"tooltip": "Optional LoRA style model."}),
                "instruction": ("STRING", {"default": "", "multiline": True, "tooltip": "Custom instruction (overrides default cover instruction)."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio", "audio_path", "metadata")
    FUNCTION = "generate_cover"
    CATEGORY = "Audio/ACE-Step"

    def generate_cover(
        self,
        src_audio: Dict[str, Any],
        caption: str,
        checkpoint_dir: str,
        config_path: str,
        lm_model_path: str,
        audio_cover_strength: float,
        batch_size: int,
        seed: int,
        inference_steps: int,
        device: str,
        quantization: str = "None",
        compile_model: bool = False,
        lyrics: str = "",
        vocal_language: str = "unknown",
        instrumental: bool = False,
        bpm: int = 0,
        keyscale: str = "",
        timesignature: str = "",
        use_adg: bool = False,
        guidance_scale: float = 7.0,
        thinking: bool = True,
        audio_format: str = "flac",
        lora_info: Optional[Dict[str, Any]] = None,
        instruction: str = "",
    ) -> Tuple[Dict[str, Any], str, str]:
        import tempfile

        # Clear CUDA cache before generation to prevent OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save input audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = tmp.name
            # ComfyUI waveform format: (batch, channels, samples) -> (samples, channels) for soundfile
            waveform = src_audio["waveform"].squeeze(0).numpy().T
            import soundfile as sf
            sf.write(temp_path, waveform, src_audio["sample_rate"])

        try:
            # Initialize handlers
            dit_handler, llm_handler = self.initialize_handlers(
                checkpoint_dir=checkpoint_dir,
                config_path=config_path,
                lm_model_path=lm_model_path,
                device=device,
                lora_info=lora_info,
                quantization=quantization,
                compile_model=compile_model,
            )

            # Auto-set instruction for cover task
            # Cover task requires specific instruction for model to recognize the task type
            cover_instruction = "Generate audio semantic tokens based on the given conditions:"

            # Prepare generation parameters
            params = GenerationParams(
                task_type="cover",
                src_audio=temp_path,
                caption=caption,
                lyrics=lyrics if lyrics else "",
                instrumental=instrumental,
                vocal_language=vocal_language,
                bpm=bpm if bpm > 0 else None,
                keyscale=keyscale,
                timesignature=timesignature,
                audio_cover_strength=audio_cover_strength,
                inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                use_adg=use_adg,
                instruction=cover_instruction,  # Auto-set for cover task
                seed=seed,
                thinking=thinking,
                # Disable all CoT features when thinking=False (required for LoRA compatibility)
                use_cot_caption=thinking,
                use_cot_language=thinking,
                use_cot_metas=thinking,
            )


            # Prepare generation config
            config = GenerationConfig(
                batch_size=batch_size,
                use_random_seed=(seed == -1),
                audio_format=audio_format,
            )

            # Generate music
            output_dir = folder_paths.get_output_directory()
            result = generate_music(dit_handler, llm_handler, params, config, save_dir=output_dir)

            if not result.success:
                raise RuntimeError(f"Generation failed: {result.error}")

            # Get the first audio from batch
            audio_data = result.audios[0]
            audio_path = audio_data["path"]
            audio_tensor = audio_data["tensor"]
            sample_rate = audio_data["sample_rate"]

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Prepare ComfyUI audio format
            # audio_tensor shape: [channels, samples] -> [1, channels, samples]
            audio_output = {
                "waveform": audio_tensor.cpu().unsqueeze(0),
                "sample_rate": sample_rate,
            }

            # Prepare metadata
            metadata = json.dumps(
                {
                    "task_type": "cover",
                    "caption": caption,
                    "audio_cover_strength": audio_cover_strength,
                    "seed": audio_data["params"].get("seed", seed),
                    "sample_rate": sample_rate,
                },
                indent=2,
            )

            return audio_output, audio_path, metadata

        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)


class ACE_STEP_REPAINT(ACE_STEP_BASE):
    """Repaint a specific segment of audio"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "src_audio": ("AUDIO", {"tooltip": "The original audio signal to be repainted."}),
                "caption": ("STRING", {"default": "", "multiline": True, "tooltip": "Style description prompt for the repainted region."}),
                "repainting_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 600.0, "tooltip": "Start time for the repainting region in seconds."}),
                "repainting_end": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 600.0, "tooltip": "End time for the repainting region in seconds. -1 means until the end of the audio."}),
                "checkpoint_dir": (get_acestep_checkpoints(), {"default": get_acestep_checkpoints()[0], "tooltip": "Directory containing ACE-Step model weights (DiT model)."}),
                "config_path": (get_acestep_models(), {"default": "acestep-v15-turbo", "tooltip": "Specific model configuration to use (e.g., v1.5 turbo)."}),
                "lm_model_path": (get_acestep_models(), {"default": "acestep-5Hz-lm-1.7B", "tooltip": "Path to the language model used for processing metadata."}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFffffffff, "control_after_generate": True, "tooltip": "Random seed for reproducibility. Set to -1 for random generation."}),
                "inference_steps": ("INT", {"default": 8, "min": 1, "max": 64, "tooltip": "Number of diffusion steps. Higher values (e.g., 25-50) improve quality but are slower."}),
                "device": (["auto", "cuda", "cpu", "mps", "xpu"], {"default": "auto", "tooltip": "Computing platform to run the model on."}),
            },
            "optional": {
                "thinking": ("BOOLEAN", {"default": True, "tooltip": "Whether to show the language model's Chain-of-Thought reasoning."}),
                "quantization": (["None", "int8_weight_only"], {"default": "None", "tooltip": "Model quantization (e.g., int8). Reduces VRAM usage but requires torchao and compile_model=True. Incompatible with LoRA."}),
                "compile_model": ("BOOLEAN", {"default": False, "tooltip": "Whether to use torch.compile to optimize the model. Required for quantization. Slow on first run but faster afterwards."}),
                "audio_format": (["flac", "mp3", "wav"], {"default": "flac", "tooltip": "Output audio file format."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio", "audio_path", "metadata")
    FUNCTION = "repaint_audio"
    CATEGORY = "Audio/ACE-Step"

    def repaint_audio(
        self,
        src_audio: Dict[str, Any],
        caption: str,
        repainting_start: float,
        repainting_end: float,
        checkpoint_dir: str,
        config_path: str,
        lm_model_path: str,
        seed: int,
        inference_steps: int,
        device: str,
        quantization: str = "None",
        compile_model: bool = False,
        thinking: bool = True,
        audio_format: str = "flac",
        lora_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], str, str]:
        import tempfile

        # Clear CUDA cache before generation to prevent OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save input audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = tmp.name
            # ComfyUI waveform format: (batch, channels, samples) -> (samples, channels) for soundfile
            waveform = src_audio["waveform"].squeeze(0).numpy().T
            import soundfile as sf
            sf.write(temp_path, waveform, src_audio["sample_rate"])

        try:
            # Initialize handlers
            dit_handler, llm_handler = self.initialize_handlers(
                checkpoint_dir=checkpoint_dir,
                config_path=config_path,
                lm_model_path=lm_model_path,
                device=device,
                lora_info=lora_info,
                quantization=quantization,
                compile_model=compile_model,
            )

            # Auto-set instruction for repaint task
            # Repaint task requires specific instruction for model to recognize the task type
            repaint_instruction = "Repaint the mask area based on the given conditions:"

            # Prepare generation parameters
            params = GenerationParams(
                task_type="repaint",
                src_audio=temp_path,
                caption=caption,
                repainting_start=repainting_start,
                repainting_end=repainting_end if repainting_end > 0 else -1.0,
                instruction=repaint_instruction,  # Auto-set for repaint task
                inference_steps=inference_steps,
                seed=seed,
                thinking=thinking,
                # Disable all CoT features when thinking=False (required for LoRA compatibility)
                use_cot_caption=thinking,
                use_cot_language=thinking,
                use_cot_metas=thinking,
            )


            # Prepare generation config
            config = GenerationConfig(
                batch_size=1,
                use_random_seed=(seed == -1),
                audio_format=audio_format,
            )

            # Generate music
            output_dir = folder_paths.get_output_directory()
            result = generate_music(dit_handler, llm_handler, params, config, save_dir=output_dir)

            if not result.success:
                raise RuntimeError(f"Generation failed: {result.error}")

            # Get the audio
            audio_data = result.audios[0]
            audio_path = audio_data["path"]
            audio_tensor = audio_data["tensor"]
            sample_rate = audio_data["sample_rate"]

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Prepare ComfyUI audio format
            # audio_tensor shape: [channels, samples] -> [1, channels, samples]
            audio_output = {
                "waveform": audio_tensor.cpu().unsqueeze(0),
                "sample_rate": sample_rate,
            }

            # Prepare metadata
            metadata = json.dumps(
                {
                    "task_type": "repaint",
                    "caption": caption,
                    "repainting_start": repainting_start,
                    "repainting_end": repainting_end,
                    "seed": audio_data["params"].get("seed", seed),
                    "sample_rate": sample_rate,
                },
                indent=2,
            )

            return audio_output, audio_path, metadata

        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)


class ACE_STEP_SIMPLE_MODE(ACE_STEP_BASE):
    """Simple mode: Generate music from natural language description"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "query": ("STRING", {"default": "", "multiline": True, "tooltip": "Natural language description or prompt for music generation."}),
                "checkpoint_dir": (get_acestep_checkpoints(), {"default": get_acestep_checkpoints()[0], "tooltip": "Directory containing ACE-Step model weights (DiT model)."}),
                "config_path": (get_acestep_models(), {"default": "acestep-v15-turbo", "tooltip": "Specific model configuration to use (e.g., v1.5 turbo)."}),
                "lm_model_path": (get_acestep_models(), {"default": "acestep-5Hz-lm-1.7B", "tooltip": "Path to the language model used for generating lyrics and metadata."}),
                "batch_size": ("INT", {"default": 2, "min": 1, "max": 8, "tooltip": "Number of audio samples to generate in a single batch."}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFffffffff, "control_after_generate": True, "tooltip": "Random seed for reproducibility. Set to -1 for random generation."}),
                "inference_steps": ("INT", {"default": 8, "min": 1, "max": 64, "tooltip": "Number of diffusion steps. Higher values (e.g., 25-50) improve quality but are slower."}),
                "device": (["auto", "cuda", "cpu", "mps", "xpu"], {"default": "auto", "tooltip": "Computing platform to run the model on."}),
            },
            "optional": {
                "instrumental": ("BOOLEAN", {"default": False, "tooltip": "Whether to generate instrumental music only (no vocals)."}),
                "vocal_language": (["auto", "en", "zh", "ja", "ko", "es", "fr", "de", "ru", "pt", "it", "bn"], {"default": "auto", "tooltip": "Vocal language (e.g., zh, en, ja)."}),
                "quantization": (["None", "int8_weight_only"], {"default": "None", "tooltip": "Model quantization (e.g., int8). Reduces VRAM usage but requires torchao and compile_model=True. Incompatible with LoRA."}),
                "compile_model": ("BOOLEAN", {"default": False, "tooltip": "Whether to use torch.compile to optimize the model. Required for quantization. Slow on first run but faster afterwards."}),
                "thinking": ("BOOLEAN", {"default": True, "tooltip": "Whether to show the language model's Chain-of-Thought reasoning."}),
                "audio_format": (["flac", "mp3", "wav"], {"default": "flac", "tooltip": "Output audio file format."}),
                "lora_info": ("ACE_STEP_LORA_INFO", {"tooltip": "Optional LoRA model information for style fine-tuning."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("audio", "audio_path", "metadata", "sample_info")
    FUNCTION = "simple_generate"
    CATEGORY = "Audio/ACE-Step"

    def simple_generate(
        self,
        query: str,
        checkpoint_dir: str,
        lm_model_path: str,
        config_path: str,
        batch_size: int,
        seed: int,
        inference_steps: int,
        device: str,
        quantization: str = "None",
        compile_model: bool = False,
        instrumental: bool = False,
        vocal_language: str = "unknown",
        bpm: int = 120,
        thinking: bool = True,
        audio_format: str = "flac",
        lora_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], str, str, str]:
        # Clear CUDA cache before generation to prevent OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Initialize handlers
        dit_handler, llm_handler = self.initialize_handlers(
            checkpoint_dir=checkpoint_dir,
            config_path=config_path,
            lm_model_path=lm_model_path,
            device=device,
            lora_info=lora_info,
            quantization=quantization,
            compile_model=compile_model,
        )

        # Step 1: Create sample from description
        sample_result = create_sample(
            llm_handler=llm_handler,
            query=query,
            instrumental=instrumental,
            vocal_language=None if vocal_language == "auto" else vocal_language,
            temperature=0.0,
        )

        if not sample_result.success:
            raise RuntimeError(f"Sample creation failed: {sample_result.error}")

        # Step 2: Generate music using the sample
        params = create_generation_params(
            task_type="text2music",
            caption=sample_result.caption,
            lyrics=sample_result.lyrics,
            bpm=sample_result.bpm,
            duration=sample_result.duration,
            keyscale=sample_result.keyscale,
            vocal_language=sample_result.language,
            instrumental=instrumental,
            inference_steps=inference_steps,
            seed=seed,
            thinking=thinking,
            # Disable all CoT features when thinking=False (required for LoRA compatibility)
            use_cot_caption=thinking,
            use_cot_language=thinking,
            use_cot_metas=thinking,
        )

        # Prepare generation config
        config = GenerationConfig(
            batch_size=batch_size,
            use_random_seed=(seed == -1),
            audio_format=audio_format,
        )

        # Generate music
        output_dir = folder_paths.get_output_directory()
        result = generate_music(dit_handler, llm_handler, params, config, save_dir=output_dir)

        if not result.success:
            raise RuntimeError(f"Generation failed: {result.error}")

        # Get the first audio from batch
        audio_data = result.audios[0]
        audio_path = audio_data["path"]
        audio_tensor = audio_data["tensor"]
        sample_rate = audio_data["sample_rate"]

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Prepare ComfyUI audio format
        # audio_tensor shape: [channels, samples] -> [1, channels, samples]
        audio_output = {
            "waveform": audio_tensor.cpu().unsqueeze(0),
            "sample_rate": sample_rate,
        }

        # Prepare metadata
        metadata = json.dumps(
            {
                "task_type": "simple_mode",
                "query": query,
                "caption": sample_result.caption,
                "bpm": sample_result.bpm,
                "duration": sample_result.duration,
                "seed": audio_data["params"].get("seed", seed),
                "sample_rate": sample_rate,
            },
            indent=2,
        )

        # Prepare sample info
        sample_info = json.dumps(
            {
                "caption": sample_result.caption,
                "lyrics": sample_result.lyrics,
                "bpm": sample_result.bpm,
                "duration": sample_result.duration,
                "keyscale": sample_result.keyscale,
                "language": sample_result.language,
                "timesignature": sample_result.timesignature,
                "instrumental": sample_result.instrumental,
            },
            indent=2,
        )

        return audio_output, audio_path, metadata, sample_info


class ACE_STEP_FORMAT_SAMPLE(ACE_STEP_BASE):
    """Format and enhance user-provided caption and lyrics"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "caption": ("STRING", {"default": "", "multiline": True, "tooltip": "Natural language description or prompt for music generation."}),
                "lyrics": ("STRING", {"default": "", "multiline": True, "tooltip": "Song lyrics to be formatted."}),
                "checkpoint_dir": (get_acestep_checkpoints(), {"default": get_acestep_checkpoints()[0], "tooltip": "Directory containing ACE-Step model weights (DiT model)."}),
                "lm_model_path": (get_acestep_models(), {"default": "acestep-5Hz-lm-1.7B", "tooltip": "Path to the language model used for formatting and enhancement."}),
                "device": (["auto", "cuda", "cpu", "mps", "xpu"], {"default": "auto", "tooltip": "Computing platform to run the model on."}),
            },
            "optional": {
                "user_metadata": ("STRING", {"default": "{}", "tooltip": "Custom metadata in JSON format."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("formatted_caption", "formatted_lyrics", "formatted_metadata")
    FUNCTION = "format_input"
    CATEGORY = "Audio/ACE-Step"

    def format_input(
        self,
        caption: str,
        lyrics: str,
        checkpoint_dir: str,
        lm_model_path: str,
        device: str,
        user_metadata: str = "{}",
    ) -> Tuple[str, str, str]:
        # Initialize handlers
        dit_handler, llm_handler = self.initialize_handlers(
            checkpoint_dir=checkpoint_dir,
            config_path="acestep-v15-turbo",  # Dummy, will be overwritten
            lm_model_path=lm_model_path,
            device=device,
        )

        # Parse user metadata
        try:
            metadata_dict = json.loads(user_metadata) if user_metadata else {}
        except json.JSONDecodeError:
            metadata_dict = {}

        # Format sample
        result = format_sample(
            llm_handler=llm_handler,
            caption=caption,
            lyrics=lyrics,
            user_metadata=metadata_dict if metadata_dict else None,
            temperature=0.85,
        )

        if not result.success:
            raise RuntimeError(f"Format failed: {result.error}")

        # Prepare formatted metadata
        formatted_metadata = json.dumps(
            {
                "caption": result.caption,
                "lyrics": result.lyrics,
                "bpm": result.bpm,
                "duration": result.duration,
                "keyscale": result.keyscale,
                "language": result.language,
                "timesignature": result.timesignature,
            },
            indent=2,
        )

        return result.caption, result.lyrics, formatted_metadata


class ACE_STEP_CREATE_SAMPLE(ACE_STEP_BASE):
    """Generate music description and lyrics from natural language query"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "query": ("STRING", {"default": "", "multiline": True, "tooltip": "Natural language query describing the music you want to generate."}),
                "checkpoint_dir": (get_acestep_checkpoints(), {"default": get_acestep_checkpoints()[0], "tooltip": "Directory containing ACE-Step model weights (DiT model)."}),
                "lm_model_path": (get_acestep_models(), {"default": "acestep-5Hz-lm-1.7B", "tooltip": "Path to the language model used for generating lyrics and metadata."}),
                "device": (["auto", "cuda", "cpu", "mps", "xpu"], {"default": "auto", "tooltip": "Computing platform to run the model on."}),
            },
            "optional": {
                "instrumental": ("BOOLEAN", {"default": False, "tooltip": "Whether to generate instrumental music only (no vocals)."}),
                "vocal_language": (["auto", "en", "zh", "ja", "ko", "es", "fr", "de", "ru", "pt", "it", "bn"], {"default": "auto", "tooltip": "Vocal language (e.g., zh, en, ja)."}),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 1000, "tooltip": "Top-K filtering parameter for sampling."}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "tooltip": "Top-P (nucleus sampling) filtering parameter."}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "tooltip": "Sampling temperature for the language model. 0.0 is most stable (recommended)."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("caption", "lyrics", "duration", "bpm", "keyscale", "vocal_language")
    FUNCTION = "generate_sample"
    CATEGORY = "Audio/ACE-Step"

    def generate_sample(
        self,
        query: str,
        checkpoint_dir: str,
        lm_model_path: str,
        device: str,
        instrumental: bool = False,
        vocal_language: str = "auto",
        top_k: int = 50,
        top_p: float = 0.95,
        temperature: float = 0.0,
    ) -> Tuple[str, str, float, int, str, str]:
        # Initialize handlers
        dit_handler, llm_handler = self.initialize_handlers(
            checkpoint_dir=checkpoint_dir,
            config_path="acestep-v15-turbo",  # Dummy
            lm_model_path=lm_model_path,
            device=device,
        )

        # Create sample (note: upstream create_sample doesn't support seed)
        result = create_sample(
            llm_handler=llm_handler,
            query=query,
            instrumental=instrumental,
            vocal_language=None if vocal_language == "auto" else vocal_language,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        if not result.success:
            raise RuntimeError(f"Sample creation failed: {result.error}")

        return (
            result.caption,
            result.lyrics,
            float(result.duration or 30.0),
            int(result.bpm or 120),
            result.keyscale or "",
            result.language or "unknown",
        )


class ACE_STEP_UNDERSTAND(ACE_STEP_BASE):
    """Understand and analyze audio"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "The audio signal to be analyzed."}),
                "checkpoint_dir": (get_acestep_checkpoints(), {"default": get_acestep_checkpoints()[0], "tooltip": "Directory containing ACE-Step model weights (DiT model)."}),
                "config_path": (get_acestep_models(), {"default": "acestep-v15-turbo", "tooltip": "Specific model configuration to use (e.g., v1.5 turbo)."}),
                "lm_model_path": (get_acestep_models(), {"default": "acestep-5Hz-lm-1.7B", "tooltip": "Path to the language model used for audio analysis and understanding."}),
                "target_duration": ("FLOAT", {"default": 30.0, "min": 10.0, "max": 600.0, "tooltip": "Target duration to reference during analysis."}),
                "device": (["auto", "cuda", "cpu", "mps", "xpu"], {"default": "auto", "tooltip": "Computing platform to run the model on."}),
            },
            "optional": {
                "language": (["auto", "en", "zh", "ja", "ko", "fr", "de", "es", "it", "ru", "pt", "nl", "tr", "pl", "ar", "vi", "th"], {"default": "auto", "tooltip": "Hint the model about the vocal language in the audio."}),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.1, "tooltip": "Sampling temperature. Lower (0.0-0.3) = more precise/faithful, Higher (0.5+) = more creative. Try 0.1 for better accuracy."}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 100, "tooltip": "Top-K sampling. 0 = disabled. Lower values (e.g., 20-50) can improve accuracy by limiting token choices."}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Top-P (nucleus) sampling. 1.0 = disabled. Lower values (e.g., 0.8-0.9) can improve accuracy."}),
                "repetition_penalty": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1, "tooltip": "Repetition penalty. 1.0 = no penalty. Higher values (1.1-1.3) reduce repetitive lyrics."}),
                "thinking": ("BOOLEAN", {"default": True, "tooltip": "Whether to show the language model's Chain-of-Thought reasoning."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("analysis_text", "caption", "duration", "bpm", "keyscale", "lyrics")
    FUNCTION = "understand"
    CATEGORY = "Audio/ACE-Step"
    OUTPUT_NODE = True

    def understand(
        self,
        audio: Dict[str, Any],
        checkpoint_dir: str,
        lm_model_path: str,
        config_path: str,
        target_duration: float,
        device: str,
        language: str = "auto",
        temperature: float = 0.3,
        top_k: int = 0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        thinking: bool = True,
    ) -> Tuple[str, str, float, str, str, str]:
        import tempfile
        import acestep.llm_inference

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save input audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = tmp.name
            waveform = audio["waveform"].squeeze(0).numpy().T
            import soundfile as sf
            sf.write(temp_path, waveform, audio["sample_rate"])

        try:
            # Initialize handlers
            dit_handler, llm_handler = self.initialize_handlers(
                checkpoint_dir=checkpoint_dir,
                config_path=config_path,
                lm_model_path=lm_model_path,
                device=device,
            )

            # Convert audio to codes
            print(f"[understand] Converting audio to codes for {target_duration}s...")
            input_codes = dit_handler.convert_src_audio_to_codes(temp_path)
            
            # Handle potential error message from conversion
            if isinstance(input_codes, str) and input_codes.startswith("❌"):
                raise RuntimeError(input_codes)
            
            # Dynamic Monkeypatch: Inject language hint if specified
            original_instruction = acestep.llm_inference.DEFAULT_LM_UNDERSTAND_INSTRUCTION
            if language != "auto":
                lang_instruction = f" The vocal language is {language}."
                # Append if not already present (to avoid double appending on re-runs if global state persists differently)
                if lang_instruction not in acestep.llm_inference.DEFAULT_LM_UNDERSTAND_INSTRUCTION:
                     acestep.llm_inference.DEFAULT_LM_UNDERSTAND_INSTRUCTION += lang_instruction
                     print(f"[understand] Injected language hint: {lang_instruction}")

            try:
                # Call understand_music with additional parameters for better accuracy
                result = understand_music(
                    llm_handler=llm_handler,
                    audio_codes=input_codes,
                    temperature=temperature,
                    top_k=top_k if top_k > 0 else None,
                    top_p=top_p if top_p < 1.0 else None,
                    repetition_penalty=repetition_penalty,
                )
            finally:
                # Restore original instruction strictly
                if language != "auto":
                     acestep.llm_inference.DEFAULT_LM_UNDERSTAND_INSTRUCTION = original_instruction

            if not result.success:
                raise RuntimeError(f"Understanding failed: {result.error}")
            
            # Format output text analysis
            analysis = (
                f"🎵 Analysis Result:\n"
                f"----------------\n"
                f"📝 Caption: {result.caption}\n"
                f"⏱️ BPM: {result.bpm or 'N/A'}\n"
                f"⏳ Duration: {result.duration or 'N/A'}s\n"
                f"🎼 Key: {result.keyscale or 'N/A'}\n"
                f"🗣️ Language: {result.language or 'N/A'}\n"
                f"🎼 Time Signature: {result.timesignature or 'N/A'}\n\n"
                f"📜 Lyrics:\n{result.lyrics}"
            )

            return (
                analysis,
                result.caption,
                float(result.duration or 0.0),
                str(result.bpm or ""),
                result.keyscale,
                result.lyrics,
            )

        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)



class ACE_STEP_LORA_LOADER:
    """Load PEFT LoRA for ACE-Step"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_path": (get_available_peft_loras(), {"default": "None", "tooltip": "Select the LoRA model to load."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Strength of the LoRA effect."}),
            }
        }

    RETURN_TYPES = ("ACE_STEP_LORA_INFO",)
    RETURN_NAMES = ("lora_info",)
    FUNCTION = "load_lora_config"
    CATEGORY = "Audio/ACE-Step"

    def load_lora_config(self, lora_path, strength):
        if lora_path == "None" or not lora_path:
            return (None,)
            
        # Resolve relative path back to absolute
        abs_path = os.path.join(folder_paths.models_dir, lora_path)
        if not os.path.exists(abs_path):
             # Fallback: maybe it's already absolute (e.g. from old workflow)
             if os.path.exists(lora_path):
                 abs_path = lora_path
             else:
                 print(f"[ACE_STEP_LoRALoader] LoRA path not found: {lora_path}")
                 return (None,)

        return ({
            "path": abs_path,
            "scale": strength
        },)


class ACE_STEP_LORA_PREPARE_TRAINING_DATA(ACE_STEP_BASE):
    """Prepare training data from audio files and metadata

    This node preprocesses audio files with their metadata (caption, lyrics, etc.)
    into tensor files that can be used for efficient LoRA training.

    Input format:
    - audio_files: Comma-separated list of audio file paths, OR load audio through ComfyUI
    - captions: Comma-separated list of captions
    - lyrics_list: Comma-separated list of lyrics (use [Instrumental] for no vocals)
    - bpm_list: Comma-separated list of BPM values (use 0 for auto-detect)
    - keyscale_list: Comma-separated list of keys (e.g., C Major, Am)
    - language_list: Comma-separated list of languages (en, zh, instrumental)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_files": ("STRING", {"default": "", "multiline": True, "tooltip": "Comma-separated list of audio file paths OR use audio input connections"}),
                "captions": ("STRING", {"default": "", "multiline": True, "tooltip": "Comma-separated list of captions/descriptions for each audio file. Use one caption per audio file."}),
                "lyrics_list": ("STRING", {"default": "[Instrumental]", "multiline": True, "tooltip": "Comma-separated list of lyrics for each audio file. Use '[Instrumental]' for instrumental tracks."}),
                "bpm_list": ("STRING", {"default": "0", "tooltip": "Comma-separated list of BPM values. Use '0' or leave empty for auto-detection."}),
                "keyscale_list": ("STRING", {"default": "", "tooltip": "Comma-separated list of musical keys (e.g., 'C Major', 'Am')."}),
                "language_list": ("STRING", {"default": "instrumental", "tooltip": "Comma-separated list of languages (e.g., 'en', 'zh', 'instrumental')."}),
                "output_dir": ("STRING", {"default": "./datasets/preprocessed_tensors", "tooltip": "Directory to save preprocessed tensor files."}),
                "checkpoint_dir": (get_acestep_checkpoints(), {"default": get_acestep_checkpoints()[0], "tooltip": "Directory containing ACE-Step model weights."}),
                "config_path": (get_acestep_models(), {"default": "acestep-v15-turbo", "tooltip": "Model configuration to use."}),
                "device": (["auto", "cuda", "cpu", "mps", "xpu"], {"default": "auto", "tooltip": "Computing platform."}),
            },
            "optional": {
                "audio_input_1": ("AUDIO", {"tooltip": "Optional: Connect audio directly from ComfyUI"}),
                "audio_input_2": ("AUDIO", {}),
                "audio_input_3": ("AUDIO", {}),
                "audio_input_4": ("AUDIO", {}),
                "custom_tag": ("STRING", {"default": "", "tooltip": "Custom tag to prepend/append to all captions (e.g., '8bit_retro')."}),
                "tag_position": (["append", "prepend", "replace"], {"default": "append", "tooltip": "Where to place the custom tag in the caption."}),
                "max_duration": ("FLOAT", {"default": 240.0, "min": 10.0, "max": 600.0, "tooltip": "Maximum audio duration in seconds. Longer audio will be truncated."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("tensor_dir", "status")
    FUNCTION = "prepare_training_data"
    CATEGORY = "Audio/ACE-Step/Training"
    OUTPUT_NODE = False

    def prepare_training_data(
        self,
        audio_files: str,
        captions: str,
        lyrics_list: str,
        bpm_list: str,
        keyscale_list: str,
        language_list: str,
        output_dir: str,
        checkpoint_dir: str,
        config_path: str,
        device: str,
        audio_input_1: Optional[Dict] = None,
        audio_input_2: Optional[Dict] = None,
        audio_input_3: Optional[Dict] = None,
        audio_input_4: Optional[Dict] = None,
        custom_tag: str = "",
        tag_position: str = "append",
        max_duration: float = 240.0,
    ) -> Tuple[str, str]:
        """Prepare training data by preprocessing audio files to tensor files."""

        import os

        # Collect audio files from both string paths and audio inputs
        audio_file_paths = []
        audio_data_list = []

        # From string input (file paths)
        if audio_files:
            for path in audio_files.split(","):
                path = path.strip()
                if path and os.path.exists(path):
                    audio_file_paths.append(path)

        # From audio input connections
        for audio_input in [audio_input_1, audio_input_2, audio_input_3, audio_input_4]:
            if audio_input is not None:
                audio_data_list.append(audio_input)

        # Save connected audio to temp files
        for i, audio_data in enumerate(audio_data_list):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=output_dir) as tmp:
                temp_path = tmp.name
                waveform = audio_data["waveform"].squeeze(0).numpy().T
                import soundfile as sf
                sf.write(temp_path, waveform, audio_data["sample_rate"])
                audio_file_paths.append(temp_path)

        if not audio_file_paths:
            return "", "❌ No audio files provided. Please provide audio file paths OR connect audio inputs."

        # Initialize handlers
        dit_handler, llm_handler = self.initialize_handlers(
            checkpoint_dir=checkpoint_dir,
            config_path=config_path,
            lm_model_path="acestep-5Hz-lm-1.7B",  # Required for text encoding
            device=device,
        )

        # Parse input lists
        caption_list = [c.strip() for c in captions.split(",") if c.strip()] if captions else []
        lyrics_items = [l.strip() for l in lyrics_list.split(",") if l.strip()] if lyrics_list else []
        bpm_items = [b.strip() for b in bpm_list.split(",") if b.strip()] if bpm_list else []
        keyscale_items = [k.strip() for k in keyscale_list.split(",") if k.strip()] if keyscale_list else []
        language_items = [lang.strip() for lang in language_list.split(",") if lang.strip()] if language_list else []

        # Ensure we have captions for all audio files
        num_audios = len(audio_file_paths)
        if not caption_list:
            caption_list = [""] * num_audios
        if len(caption_list) < num_audios:
            caption_list.extend([""] * (num_audios - len(caption_list)))

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Import training modules
        try:
            from acestep.training.dataset_builder import AudioSample, DatasetBuilder, DatasetMetadata
        except ImportError:
            return "", "❌ ACE-Step training modules not available. Please ensure ACE-Step is properly installed."

        # Create dataset builder
        builder = DatasetBuilder()
        builder.metadata.name = "comfyui_training_dataset"
        builder.metadata.custom_tag = custom_tag
        builder.metadata.tag_position = tag_position

        # Convert to AudioSample objects
        samples = []
        for i, audio_path in enumerate(audio_file_paths):
            caption = caption_list[i] if i < len(caption_list) else ""
            lyrics = lyrics_items[i] if i < len(lyrics_items) else "[Instrumental]"
            bpm = int(bpm_items[i]) if i < len(bpm_items) and bpm_items[i].isdigit() else None
            keyscale = keyscale_items[i] if i < len(keyscale_items) else ""
            language = language_items[i] if i < len(language_items) else "instrumental"

            sample = AudioSample(
                audio_path=audio_path,
                filename=os.path.basename(audio_path),
                caption=caption,
                lyrics=lyrics,
                bpm=bpm,
                keyscale=keyscale,
                language=language,
                is_instrumental=(lyrics == "[Instrumental]"),
                labeled=True,
            )
            samples.append(sample)

        builder.samples = samples

        # Preprocess to tensors
        output_paths, status = builder.preprocess_to_tensors(
            dit_handler=dit_handler,
            output_dir=output_dir,
            max_duration=max_duration,
        )

        # Clean up temp files (only those we created)
        for i, audio_data in enumerate(audio_data_list):
            temp_path = os.path.join(output_dir, f"tmp{i}.wav")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass

        return output_dir, status


class ACE_STEP_LORA_TRAIN(ACE_STEP_BASE):
    """Train LoRA adapter from preprocessed tensor files

    This node performs LoRA fine-tuning on the DiT decoder using preprocessed
    tensor files. Training uses Flow Matching with discrete timesteps.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor_dir": ("STRING", {"default": "./datasets/preprocessed_tensors", "tooltip": "Directory containing preprocessed .pt tensor files."}),
                "checkpoint_dir": (get_acestep_checkpoints(), {"default": get_acestep_checkpoints()[0], "tooltip": "Directory containing ACE-Step model weights."}),
                "config_path": (get_acestep_models(), {"default": "acestep-v15-turbo", "tooltip": "Model configuration (turbo recommended for training)."}),
                "output_dir": ("STRING", {"default": "./lora_output", "tooltip": "Directory to save trained LoRA adapter."}),
                "lora_rank": ("INT", {"default": 64, "min": 4, "max": 256, "tooltip": "LoRA rank (dimension of low-rank matrices). Higher = more capacity but slower."}),
                "lora_alpha": ("INT", {"default": 128, "min": 4, "max": 512, "tooltip": "LoRA scaling factor. Typically 2x the rank."}),
                "lora_dropout": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.05, "tooltip": "Dropout probability for LoRA layers."}),
                "learning_rate": ("FLOAT", {"default": 3e-4, "min": 1e-6, "max": 1e-2, "tooltip": "Learning rate for training."}),
                "train_epochs": ("INT", {"default": 1000, "min": 100, "max": 4000, "tooltip": "Number of training epochs."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8, "tooltip": "Training batch size."}),
                "gradient_accumulation": ("INT", {"default": 1, "min": 1, "max": 16, "tooltip": "Gradient accumulation steps."}),
                "save_every_n_epochs": ("INT", {"default": 200, "min": 50, "max": 1000, "tooltip": "Save checkpoint every N epochs."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFffffffff, "tooltip": "Random seed for training."}),
                "device": (["auto", "cuda", "cpu", "mps", "xpu"], {"default": "auto", "tooltip": "Computing platform."}),
            },
            "optional": {
                "resume_from": ("STRING", {"default": "", "tooltip": "Path to checkpoint directory to resume training from."}),
                "target_modules": ("STRING", {"default": "q_proj,k_proj,v_proj,o_proj", "tooltip": "Comma-separated list of target modules for LoRA."}),
            },
        }

    RETURN_TYPES = ("ACE_STEP_LORA_INFO", "STRING")
    RETURN_NAMES = ("lora_info", "training_log")
    FUNCTION = "train_lora"
    CATEGORY = "Audio/ACE-Step/Training"
    OUTPUT_NODE = True

    def train_lora(
        self,
        tensor_dir: str,
        checkpoint_dir: str,
        config_path: str,
        output_dir: str,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
        learning_rate: float,
        train_epochs: int,
        batch_size: int,
        gradient_accumulation: int,
        save_every_n_epochs: int,
        seed: int,
        device: str,
        resume_from: str = "",
        target_modules: str = "q_proj,k_proj,v_proj,o_proj",
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Train LoRA adapter from preprocessed tensors."""

        import os

        # Validate tensor directory
        if not os.path.exists(tensor_dir):
            return None, f"❌ Tensor directory not found: {tensor_dir}"

        # Check for required dependencies
        try:
            from lightning.fabric import Fabric
            from peft import get_peft_model, LoraConfig
        except ImportError as e:
            return None, f"❌ Missing required packages: {e}\nPlease install: pip install peft lightning"

        # Initialize DiT handler (without LLM to save memory)
        # Training only needs DiT handler, not LLM
        dit_handler, _ = self.initialize_handlers(
            checkpoint_dir=checkpoint_dir,
            config_path=config_path,
            lm_model_path="acestep-5Hz-lm-1.7B",  # Still needed for init but not for training
            device=device,
        )

        # Check for incompatible settings
        if getattr(dit_handler, 'offload_to_cpu', False):
            return None, "❌ Offload to CPU is enabled - this will slow down training. Please disable it."
        if getattr(dit_handler, 'compiled', False):
            return None, "❌ Compile Model is enabled - this is incompatible with LoRA training."
        if getattr(dit_handler, 'quantization', None) is not None:
            return None, "❌ INT8 Quantization is enabled - this is incompatible with LoRA training."

        # Unload LLM to free up memory for training
        if self.llm_handler is not None:
            self.llm_handler = None

        # Create training configs
        from acestep.training.configs import LoRAConfig, TrainingConfig

        # Parse target modules
        target_module_list = [m.strip() for m in target_modules.split(",") if m.strip()]

        lora_config = LoRAConfig(
            r=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=target_module_list,
        )

        training_config = TrainingConfig(
            shift=3.0,  # Fixed for turbo model
            num_inference_steps=8,  # Fixed for turbo model
            learning_rate=learning_rate,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            max_epochs=train_epochs,
            save_every_n_epochs=save_every_n_epochs,
            seed=seed,
            output_dir=output_dir,
        )

        # Create trainer
        from acestep.training.trainer import LoRATrainer

        trainer = LoRATrainer(
            dit_handler=dit_handler,
            lora_config=lora_config,
            training_config=training_config,
        )

        # Run training
        log_messages = []
        training_state = {}

        try:
            for step, loss, status in trainer.train_from_preprocessed(
                tensor_dir=tensor_dir,
                training_state=training_state,
                resume_from=resume_from if resume_from else None,
            ):
                log_messages.append(f"Step {step}: {status}")
                print(f"[ACE_STEP_LORA_TRAIN] {status}")

            final_status = log_messages[-1] if log_messages else "Training completed"

            # Get the final LoRA path
            final_lora_path = os.path.join(output_dir, "final")

            if os.path.exists(final_lora_path):
                lora_info = {
                    "path": final_lora_path,
                    "scale": 1.0,
                }
                return lora_info, "\n".join(log_messages)
            else:
                return None, "\n".join(log_messages) + "\n❌ LoRA output not found"

        except Exception as e:
            import traceback
            error_msg = f"❌ Training failed: {str(e)}\n{traceback.format_exc()}"
            print(f"[ACE_STEP_LORA_TRAIN] {error_msg}")
            return None, error_msg


class ACE_STEP_CREATE_TRAINING_SAMPLE:
    """Save a training sample to file for later use

    Helper node to prepare individual training data items by saving
    audio to a file with associated metadata. This creates a single
    training sample that can be referenced by file path.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Audio file for training."}),
                "caption": ("STRING", {"default": "", "multiline": True, "tooltip": "Description of the music style."}),
                "output_dir": ("STRING", {"default": "./training_samples", "tooltip": "Directory to save training samples."}),
            },
            "optional": {
                "lyrics": ("STRING", {"default": "[Instrumental]", "multiline": True, "tooltip": "Song lyrics. Use '[Instrumental]' for instrumental tracks."}),
                "bpm": ("INT", {"default": 0, "min": 0, "max": 300, "tooltip": "Beats per minute. 0 for auto-detect."}),
                "keyscale": ("STRING", {"default": "", "tooltip": "Musical key (e.g., 'C Major', 'Am')."}),
                "language": (["instrumental", "en", "zh", "ja", "ko", "es", "fr", "de"], {"default": "instrumental", "tooltip": "Language of the vocals."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("sample_info",)
    FUNCTION = "create_sample"
    CATEGORY = "Audio/ACE-Step/Training"

    def create_sample(
        self,
        audio: Dict[str, Any],
        caption: str,
        output_dir: str,
        lyrics: str = "[Instrumental]",
        bpm: int = 0,
        keyscale: str = "",
        language: str = "instrumental",
    ) -> Tuple[str]:
        """Create a single training sample by saving audio to file."""

        import os
        import json
        import tempfile

        os.makedirs(output_dir, exist_ok=True)

        # Generate unique filename
        import time
        timestamp = int(time.time() * 1000)
        audio_filename = f"sample_{timestamp}.wav"
        audio_path = os.path.join(output_dir, audio_filename)

        # Save audio to file
        waveform = audio["waveform"].squeeze(0).numpy().T
        import soundfile as sf
        sf.write(audio_path, waveform, audio["sample_rate"])

        # Save metadata to JSON
        metadata = {
            "audio_path": audio_path,
            "caption": caption,
            "lyrics": lyrics,
            "bpm": bpm,
            "keyscale": keyscale,
            "language": language,
            "is_instrumental": (lyrics == "[Instrumental]"),
        }

        metadata_filename = f"sample_{timestamp}.json"
        metadata_path = os.path.join(output_dir, metadata_filename)

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return f"Saved: {audio_filename}"


class ACE_STEP_COLLECT_TRAINING_SAMPLES:
    """Collect training samples from a directory

    This helper node reads all training samples (audio + metadata JSON files)
    from a directory and formats them for the Prepare Training Data node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples_dir": ("STRING", {"default": "./training_samples", "tooltip": "Directory containing training samples (.wav and .json files)."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("audio_files", "captions", "lyrics_list", "bpm_list", "keyscale_list", "language_list")
    FUNCTION = "collect_samples"
    CATEGORY = "Audio/ACE-Step/Training"

    def collect_samples(
        self,
        samples_dir: str,
    ) -> Tuple[str, str, str, str, str, str]:
        """Collect training samples from directory."""

        import os
        import json

        if not os.path.exists(samples_dir):
            return "", "", "", "", "", f"❌ Directory not found: {samples_dir}"

        audio_files = []
        captions = []
        lyrics_list = []
        bpm_list = []
        keyscale_list = []
        language_list = []

        # Find all JSON metadata files
        json_files = [f for f in os.listdir(samples_dir) if f.endswith('.json')]

        for json_file in json_files:
            json_path = os.path.join(samples_dir, json_file)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                audio_path = metadata.get("audio_path", "")
                if os.path.exists(audio_path):
                    audio_files.append(audio_path)
                    captions.append(metadata.get("caption", ""))
                    lyrics_list.append(metadata.get("lyrics", "[Instrumental]"))
                    bpm_list.append(str(metadata.get("bpm", 0)))
                    keyscale_list.append(metadata.get("keyscale", ""))
                    language_list.append(metadata.get("language", "instrumental"))

            except Exception as e:
                print(f"[ACE_STEP_COLLECT_TRAINING_SAMPLES] Error reading {json_file}: {e}")

        if not audio_files:
            return "", "", "", "", "", "❌ No valid training samples found in directory"

        # Join with commas
        audio_files_str = ",".join(audio_files)
        captions_str = ",".join(captions)
        lyrics_str = ",".join(lyrics_list)
        bpm_str = ",".join(bpm_list)
        keyscale_str = ",".join(keyscale_list)
        language_str = ",".join(language_list)

        return audio_files_str, captions_str, lyrics_str, bpm_str, keyscale_str, language_str


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ACE_STEP_TextToMusic": ACE_STEP_TEXT_TO_MUSIC,
    "ACE_STEP_Cover": ACE_STEP_COVER,
    "ACE_STEP_Repaint": ACE_STEP_REPAINT,
    "ACE_STEP_SimpleMode": ACE_STEP_SIMPLE_MODE,
    "ACE_STEP_FormatSample": ACE_STEP_FORMAT_SAMPLE,
    "ACE_STEP_CreateSample": ACE_STEP_CREATE_SAMPLE,
    "ACE_STEP_Understand": ACE_STEP_UNDERSTAND,
    "ACE_STEP_LoRALoader": ACE_STEP_LORA_LOADER,
    # Training nodes
    "ACE_STEP_LoRAPrepareTrainingData": ACE_STEP_LORA_PREPARE_TRAINING_DATA,
    "ACE_STEP_LoRATrain": ACE_STEP_LORA_TRAIN,
    "ACE_STEP_CreateTrainingSample": ACE_STEP_CREATE_TRAINING_SAMPLE,
    "ACE_STEP_CollectTrainingSamples": ACE_STEP_COLLECT_TRAINING_SAMPLES,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ACE_STEP_TextToMusic": "ACE-Step Text to Music",
    "ACE_STEP_Cover": "ACE-Step Cover",
    "ACE_STEP_Repaint": "ACE-Step Repaint",
    "ACE_STEP_SimpleMode": "ACE-Step Simple Mode",
    "ACE_STEP_FormatSample": "ACE-Step Format Sample",
    "ACE_STEP_CreateSample": "ACE-Step Create Sample",
    "ACE_STEP_Understand": "ACE-Step Understand",
    "ACE_STEP_LoRALoader": "ACE-Step LoRA Loader",
    # Training nodes
    "ACE_STEP_LoRAPrepareTrainingData": "ACE-Step LoRA Prepare Training Data",
    "ACE_STEP_LoRATrain": "ACE-Step LoRA Train",
    "ACE_STEP_CreateTrainingSample": "ACE-Step Create Training Sample",
    "ACE_STEP_CollectTrainingSamples": "ACE-Step Collect Training Samples",
}
