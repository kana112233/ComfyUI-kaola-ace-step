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

try:
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import (
        GenerationParams,
        GenerationConfig,
        generate_music,
        create_sample,
        format_sample,
        understand_music,
    )
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
    _original_build_formatted_prompt = LLMHandler.build_formatted_prompt
    def patched_build_formatted_prompt(self, caption: str, lyrics: str = "", *args, **kwargs):
        vocal_language = getattr(self, '_patch_vocal_language', None)
        instrumental = getattr(self, '_patch_instrumental', False)
        
        # Check if is_negative_prompt is passed as pos arg or kwarg
        is_negative_prompt = kwargs.get('is_negative_prompt', False)
        if len(args) > 0: is_negative_prompt = args[0]
        
        if not is_negative_prompt:
            from acestep.constants import DEFAULT_LM_INSTRUCTION
            instrumental_str = "true" if instrumental else "false"
            prompt = f"# Caption\n{caption}\n\ninstrumental: {instrumental_str}"
            if vocal_language and vocal_language.strip() and vocal_language.strip().lower() != "unknown":
                prompt += f"\nlanguage: {vocal_language.strip()}"
            prompt += f"\n\n# Lyric\n{lyrics}\n"
            
            return self.llm_tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": f"# Instruction\n{DEFAULT_LM_INSTRUCTION}\n\n"},
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        return _original_build_formatted_prompt(self, caption, lyrics, *args, **kwargs)
    
    LLMHandler.build_formatted_prompt = patched_build_formatted_prompt

    _original_build_formatted_prompt_with_cot = LLMHandler.build_formatted_prompt_with_cot
    def patched_build_formatted_prompt_with_cot(self, caption: str, lyrics: str, cot_text: str, *args, **kwargs):
        vocal_language = getattr(self, '_patch_vocal_language', None)
        instrumental = getattr(self, '_patch_instrumental', False)
        
        is_negative_prompt = kwargs.get('is_negative_prompt', False)
        if len(args) > 0: is_negative_prompt = args[0]
        
        if not is_negative_prompt:
            from acestep.constants import DEFAULT_LM_INSTRUCTION
            instrumental_str = "true" if instrumental else "false"
            user_prompt = f"# Caption\n{caption}\n\ninstrumental: {instrumental_str}"
            if vocal_language and vocal_language.strip() and vocal_language.strip().lower() != "unknown":
                user_prompt += f"\nlanguage: {vocal_language.strip()}"
            user_prompt += f"\n\n# Lyric\n{lyrics}\n"
            
            return self.llm_tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": f"# Instruction\n{DEFAULT_LM_INSTRUCTION}\n\n"},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": cot_text},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
        return _original_build_formatted_prompt_with_cot(self, caption, lyrics, cot_text, *args, **kwargs)
    
    LLMHandler.build_formatted_prompt_with_cot = patched_build_formatted_prompt_with_cot

    import acestep.inference
    _original_generate_music = acestep.inference.generate_music
    def patched_generate_music(params, dit_handler, llm_handler=None, **kwargs):
        if llm_handler is not None:
            llm_handler._patch_vocal_language = params.vocal_language
            llm_handler._patch_instrumental = params.instrumental
            
        # Fix: Seed handling in upstream is sensitive to int vs list
        # We ensure it's a string if it's an int, to avoid len(config.seeds) crash in upstream
        # Actually, let's fix the call params if they exist
        config = kwargs.get('config')
        if config and hasattr(config, 'seeds'):
             if isinstance(config.seeds, int):
                  # We can't easily change the class definition but we can modify the instance 
                  # before it's used in the upstream function
                  # However, upstream does `if isinstance(config.seeds, list)`.
                  # If we change it to a list, it should work fine.
                  config.seeds = [config.seeds]
        
        # Sampling parameters extraction
        # Since we removed them from upstream GenerationParams, we might need to handle them here
        # IF they are being passed through params.
        # Check if params has these attributes (to avoid AttributeError)
        lm_top_k = getattr(params, 'lm_top_k', None)
        lm_top_p = getattr(params, 'lm_top_p', None)
        
        # If upstream llm_handler.generate_with_stop_condition doesn't support them anymore,
        # we might need to bridge them via attributes on llm_handler too.
        if llm_handler is not None:
            llm_handler._patch_top_k = lm_top_k
            llm_handler._patch_top_p = lm_top_p

        return _original_generate_music(params, dit_handler, llm_handler, **kwargs)
    
    acestep.inference.generate_music = patched_generate_music

    # --------------------------------------------------------------------------------
    # MonkeyPatch LLMHandler.generate_with_stop_condition
    # --------------------------------------------------------------------------------
    _original_generate_with_stop_condition = LLMHandler.generate_with_stop_condition
    def patched_generate_with_stop_condition(self, caption: str, lyrics: str, *args, **kwargs):
        # Use bridged attributes if they were set by patched_generate_music or create_sample
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
    # MonkeyPatch LLMHandler.build_formatted_prompt_for_inspiration
    # --------------------------------------------------------------------------------
    _original_build_formatted_prompt_for_inspiration = LLMHandler.build_formatted_prompt_for_inspiration
    def patched_build_formatted_prompt_for_inspiration(self, query: str, instrumental: bool = False, *args, **kwargs):
        vocal_language = getattr(self, '_patch_vocal_language', None)
        is_negative_prompt = kwargs.get('is_negative_prompt', False)
        if len(args) > 0: is_negative_prompt = args[0]
        
        if not is_negative_prompt:
            from acestep.constants import DEFAULT_LM_INSPIRED_INSTRUCTION
            instrumental_str = "true" if instrumental else "false"
            user_content = f"{query}\n\ninstrumental: {instrumental_str}"
            if vocal_language and vocal_language.strip() and vocal_language.strip().lower() != "unknown":
                user_content += f"\nlanguage: {vocal_language.strip()}"
            
            return self.llm_tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": f"# Instruction\n{DEFAULT_LM_INSPIRED_INSTRUCTION}\n\n"},
                    {"role": "user", "content": user_content},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        return _original_build_formatted_prompt_for_inspiration(self, query, instrumental, *args, **kwargs)
    
    LLMHandler.build_formatted_prompt_for_inspiration = patched_build_formatted_prompt_for_inspiration

    # --------------------------------------------------------------------------------
    # MonkeyPatch LLMHandler.create_sample_from_query
    # --------------------------------------------------------------------------------
    _original_create_sample_from_query = LLMHandler.create_sample_from_query
    def patched_create_sample_from_query(self, query: str, instrumental: bool = False, vocal_language: Optional[str] = None, *args, **kwargs):
        # Extract custom parameters from kwargs if provided (passed from Create Sample node)
        temperature = kwargs.pop('temperature', 0.0)
        top_k = kwargs.pop('top_k', None)
        top_p = kwargs.pop('top_p', None)
        repetition_penalty = kwargs.pop('repetition_penalty', 1.0)
        seed = kwargs.pop('seed', -1)
        
        # Set bridge attributes for other patches to use
        self._patch_vocal_language = vocal_language
        self._patch_instrumental = instrumental
        
        # Original create_sample_from_query logic uses generate_from_formatted_prompt with a dict cfg
        # We need to ensure regenerate_from_formatted_prompt also respects our parameters
        # However, a simpler way is to re-implement the core logic here with our parameters
        if not getattr(self, "llm_initialized", False):
            return {}, "❌ 5Hz LM not initialized. Please initialize it first."
            
        if not query or not query.strip():
            query = "NO USER INPUT"
            
        # Build prompt
        formatted_prompt = self.build_formatted_prompt_for_inspiration(query=query, instrumental=instrumental)
        
        # Build metadata injection
        user_metadata = None
        if vocal_language and vocal_language.strip() and vocal_language.strip().lower() != "unknown":
            user_metadata = {"language": vocal_language.strip()}
            
        # Use generate_from_formatted_prompt with OUR parameters
        output_text, status = self.generate_from_formatted_prompt(
            formatted_prompt=formatted_prompt,
            cfg={
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "target_duration": None,
                "user_metadata": user_metadata,
                "skip_caption": False,
                "skip_language": False,
                "skip_genres": False,
                "generation_phase": "understand",
                "caption": "",
                "lyrics": "",
                "seed": seed,
            },
            use_constrained_decoding=kwargs.get('use_constrained_decoding', True),
            constrained_decoding_debug=kwargs.get('constrained_decoding_debug', False),
            stop_at_reasoning=False,
        )
        
        if not output_text:
            return {}, status
            
        # Parse and extract
        metadata, _ = self.parse_lm_output(output_text)
        lyrics = self._extract_lyrics_from_output(output_text)
        if lyrics:
            metadata['lyrics'] = lyrics
        elif instrumental:
            metadata['lyrics'] = "[Instrumental]"
        metadata['instrumental'] = instrumental
        
        return metadata, f"✅ Sample created successfully\nGenerated fields: {metadata}"

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
    # Return relative paths (just the model folder name)
    # Store the mapping for later use when resolving the actual path
    result = []
    for p in paths:
        # Get the relative name (last component of the path)
        name = os.path.basename(p.rstrip('/\\'))
        if not name:  # If path ends with separator
            name = os.path.basename(os.path.dirname(p))
        result.append(name if name else ACESTEP_MODEL_NAME)
    return list(set(result)) if result else [ACESTEP_MODEL_NAME]

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
    
    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue
        # PEFT LoRA is a directory containing adapter_config.json
        for root, dirs, files in os.walk(search_path):
            if "adapter_config.json" in files:
                # Found a LoRA directory
                # Use path relative to models_dir for cleaner display
                rel_path = os.path.relpath(root, folder_paths.models_dir)
                lora_paths.append(rel_path)
    
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
    ):
        """Initialize ACE-Step handlers if not already initialized"""
        if self.handlers_initialized:
            # Check if handlers are truly initialized (model loaded)
            if self.dit_handler and getattr(self.dit_handler, "model", None) is not None:
                # Clear CUDA cache before reusing handlers to prevent OOM
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Update LoRA state even if handlers are reused
                self._update_lora_state(self.dit_handler, lora_info)
                return self.dit_handler, self.llm_handler
            
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

            # Create checkpoints symlink if it doesn't exist
            # ACE-Step expects: {project_root}/checkpoints/{model_name}
            # ComfyUI uses: {models_dir}/acestep/{model_name}
            checkpoints_dir = os.path.join(checkpoint_dir, "checkpoints")
            if not os.path.exists(checkpoints_dir):
                # Create a symlink: checkpoints -> checkpoint_dir (current directory)
                # This makes both paths work:
                # - {models_dir}/acestep/{model_name}
                # - {models_dir}/acestep/checkpoints/{model_name}
                os.symlink(checkpoint_dir, checkpoints_dir, target_is_directory=True)

            # Initialize DiT handler
            self.dit_handler = AceStepHandler()

            # Monkey patch _get_project_root to use ComfyUI's model directory
            def _patched_get_project_root():
                return checkpoint_dir

            self.dit_handler._get_project_root = _patched_get_project_root

            self.dit_handler.initialize_service(
                project_root=checkpoint_dir,
                config_path=config_path,
                device=device,
                offload_to_cpu=offload_to_cpu,
            )

            # Initialize LLM handler (pass dtype from DiT handler)
            # Use "pt" backend instead of "vllm" to avoid process group conflicts with ComfyUI
            self.llm_handler = LLMHandler()
            self.llm_handler.initialize(
                checkpoint_dir=checkpoint_dir,
                lm_model_path=lm_model_path,
                backend="pt",  # Use PyTorch backend to avoid vLLM conflicts
                device=device,
                offload_to_cpu=offload_to_cpu,
                dtype=self.dit_handler.dtype,  # Critical: pass dtype from DiT handler
            )

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
            
            print(f"[initialize_handlers] Loading LoRA: {path} (scale: {scale})")
            handler.load_lora(path)
            handler.set_lora_scale(scale)
        else:
            # Ensure no LoRA is loaded if not requested
            if handler.lora_loaded:
                print(f"[initialize_handlers] Unloading LoRA")
                handler.unload_lora()


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
                "lyrics": ("STRING", {"default": "", "multiline": True, "tooltip": "Song lyrics. Leave empty for automatic generation by the language model."}),
                "bpm": ("INT", {"default": 0, "min": 0, "max": 300, "tooltip": "Beats per minute. 0 for automatic detection."}),
                "keyscale": ("STRING", {"default": "", "tooltip": "Musical key and scale (e.g., C Major)."}),
                "timesignature": ("STRING", {"default": "", "tooltip": "Musical time signature (e.g., 4/4)."}),
                "vocal_language": (["unknown", "auto", "en", "zh", "ja", "ko", "es", "fr", "de", "ru", "pt", "it", "bn"], {"default": "unknown", "tooltip": "Vocal language (e.g., zh, en, ja)."}),
                "instrumental": ("BOOLEAN", {"default": False, "tooltip": "Whether to generate instrumental music only (no vocals)."}),
                "guidance_scale": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 15.0, "tooltip": "Strength of prompt following."}),
                "shift": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 5.0, "tooltip": "Sequence length scaling factor, default is 1.0."}),
                "thinking": ("BOOLEAN", {"default": True, "tooltip": "Whether to show the language model's Chain-of-Thought reasoning."}),
                "lm_temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "tooltip": "Sampling temperature for the language model. 0.0 is most stable (recommended)."}),
                "audio_format": (["flac", "mp3", "wav"], {"default": "flac", "tooltip": "Output audio file format."}),
                "lora_info": ("ACE_STEP_LORA_INFO", {"tooltip": "Optional LoRA model information for style fine-tuning."}),
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
                "src_audio": ("AUDIO", {"tooltip": "The original audio signal to be covered."}),
                "caption": ("STRING", {"default": "", "multiline": True, "tooltip": "Description of the style for the cover version."}),
                "checkpoint_dir": (get_acestep_checkpoints(), {"default": get_acestep_checkpoints()[0], "tooltip": "Directory containing ACE-Step model weights (DiT model)."}),
                "config_path": (get_acestep_models(), {"default": "acestep-v15-turbo", "tooltip": "Specific model configuration to use (e.g., v1.5 turbo)."}),
                "lm_model_path": (get_acestep_models(), {"default": "acestep-5Hz-lm-1.7B", "tooltip": "Path to the language model used for processing metadata."}),
                "audio_cover_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "tooltip": "Strength of preserving the original audio structure. 1.0 means full preservation, 0.0 means complete reconstruction."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8, "tooltip": "Number of audio samples to generate in a single batch."}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFffffffff, "control_after_generate": True, "tooltip": "Random seed for reproducibility. Set to -1 for random generation."}),
                "inference_steps": ("INT", {"default": 8, "min": 1, "max": 64, "tooltip": "Number of diffusion steps. Higher values (e.g., 25-50) improve quality but are slower."}),
                "device": (["auto", "cuda", "cpu", "mps", "xpu"], {"default": "auto", "tooltip": "Computing platform to run the model on."}),
            },
            "optional": {
                "lyrics": ("STRING", {"default": "", "multiline": True, "tooltip": "Song lyrics. Leave empty to attempt extraction from original or keep consistency."}),
                "thinking": ("BOOLEAN", {"default": True, "tooltip": "Whether to show the language model's Chain-of-Thought reasoning."}),
                "audio_format": (["flac", "mp3", "wav"], {"default": "flac", "tooltip": "Output audio file format."}),
                "lora_info": ("ACE_STEP_LORA_INFO", {"tooltip": "Optional LoRA model information for style fine-tuning."}),
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
        lyrics: str = "",
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
            )

            # Prepare generation parameters
            params = GenerationParams(
                task_type="cover",
                src_audio=temp_path,
                caption=caption,
                lyrics=lyrics if lyrics else "",
                audio_cover_strength=audio_cover_strength,
                inference_steps=inference_steps,
                seed=seed,
                thinking=thinking,
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
                "audio_format": (["flac", "mp3", "wav"], {"default": "flac", "tooltip": "Output audio file format."}),
                "lora_info": ("ACE_STEP_LORA_INFO", {"tooltip": "Optional LoRA model information for style fine-tuning."}),
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
            )

            # Prepare generation parameters
            params = GenerationParams(
                task_type="repaint",
                src_audio=temp_path,
                caption=caption,
                repainting_start=repainting_start,
                repainting_end=repainting_end if repainting_end > 0 else -1.0,
                inference_steps=inference_steps,
                seed=seed,
                thinking=thinking,
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
                "lm_model_path": (get_acestep_models(), {"default": "acestep-5Hz-lm-1.7B", "tooltip": "Path to the language model used for generating lyrics and metadata."}),
                "config_path": (get_acestep_models(), {"default": "acestep-v15-turbo", "tooltip": "Specific model configuration to use (e.g., v1.5 turbo)."}),
                "batch_size": ("INT", {"default": 2, "min": 1, "max": 8, "tooltip": "Number of audio samples to generate in a single batch."}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFffffffff, "control_after_generate": True, "tooltip": "Random seed for reproducibility. Set to -1 for random generation."}),
                "inference_steps": ("INT", {"default": 8, "min": 1, "max": 64, "tooltip": "Number of diffusion steps. Higher values (e.g., 25-50) improve quality but are slower."}),
                "device": (["auto", "cuda", "cpu", "mps", "xpu"], {"default": "auto", "tooltip": "Computing platform to run the model on."}),
            },
            "optional": {
                "instrumental": ("BOOLEAN", {"default": False, "tooltip": "Whether to generate instrumental music only (no vocals)."}),
                "vocal_language": (["auto", "en", "zh", "ja", "ko", "es", "fr", "de", "ru", "pt", "it", "bn"], {"default": "auto", "tooltip": "Vocal language (e.g., zh, en, ja)."}),
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
        instrumental: bool = False,
        vocal_language: str = "auto",
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
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFffffffff, "control_after_generate": True, "tooltip": "Random seed for reproducibility. Set to -1 for random generation."}),
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
        seed: int = -1,
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
                "lm_model_path": (get_acestep_models(), {"default": "acestep-5Hz-lm-1.7B", "tooltip": "Path to the language model used for audio analysis and understanding."}),
                "config_path": (get_acestep_models(), {"default": "acestep-v15-turbo", "tooltip": "Specific model configuration to use (e.g., v1.5 turbo)."}),
                "target_duration": ("FLOAT", {"default": 30.0, "min": 10.0, "max": 600.0, "tooltip": "Target duration to reference during analysis."}),
                "device": (["auto", "cuda", "cpu", "mps", "xpu"], {"default": "auto", "tooltip": "Computing platform to run the model on."}),
            },
            "optional": {
                "language": (["auto", "en", "zh", "ja", "ko", "fr", "de", "es", "it", "ru", "pt", "nl", "tr", "pl", "ar", "vi", "th"], {"default": "auto", "tooltip": "Hint the model about the vocal language in the audio."}),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "tooltip": "Sampling temperature for the understanding task. Lower values are more precise."}),
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
                # Call understand_music
                result = understand_music(
                    llm_handler=llm_handler,
                    audio_codes=input_codes,
                    temperature=temperature,
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
}
