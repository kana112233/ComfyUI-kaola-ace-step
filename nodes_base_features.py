"""
ACE-Step Base Model Features

These features are ONLY available with acestep-v15-base model (NOT turbo):
- Extract: Extract a specific track/instrument from audio
- Lego: Generate a specific track based on audio context
- Complete: Complete input track with other instruments

Track names: woodwinds, brass, fx, synth, strings, percussion, keyboard, guitar, bass, drums, backing_vocals, vocals
"""

import os
import sys
import torch
import json
import tempfile
import soundfile as sf
from typing import Dict, Any, Tuple, Optional, List

import folder_paths
import comfy.utils
import comfy.model_management

# Add repo path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "acestep_repo"))
sys.path.insert(0, os.path.dirname(__file__))  # For acestep_wrapper

# Check if acestep is available
try:
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import generate_music, GenerationParams, GenerationConfig
    from acestep_wrapper import ACEStepWrapper
    ACESTEP_BASE_AVAILABLE = True
except ImportError:
    ACESTEP_BASE_AVAILABLE = False
    print("[ACE-Step Base Features] acestep module not found. Please ensure acestep_repo is downloaded.")


class ComfyProgressCallback:
    """Adapter to connect acestep progress to ComfyUI progress bar."""

    def __init__(self, total_steps: int = 100):
        self.pbar = comfy.utils.ProgressBar(total_steps)
        self.last_progress = 0

    def __call__(self, progress: float, desc: str = ""):
        """Called by acestep with progress 0.0-1.0."""
        current_step = int(progress * 100)
        steps_to_add = current_step - self.last_progress
        if steps_to_add > 0:
            for _ in range(steps_to_add):
                self.pbar.update(1)
                comfy.model_management.throw_exception_if_processing_interrupted()
            self.last_progress = current_step
            if desc:
                print(f"[ACE-Step] {desc} ({int(progress * 100)}%)")


# Constants
ACESTEP_MODEL_NAME = "Ace-Step1.5"

# Available track names for Extract/Lego/Complete
TRACK_NAMES = [
    "vocals", "backing_vocals", "drums", "bass", "guitar",
    "keyboard", "percussion", "strings", "synth", "fx", "brass", "woodwinds"
]

# Common languages for vocal generation
VOCAL_LANGUAGES = [
    "unknown", "zh", "en", "ja", "yue", "ko", "fr", "de", "es", "it", "ru", "pt"
]


def get_acestep_base_checkpoints():
    """Get available base model checkpoints."""
    model_dir = os.path.join(folder_paths.models_dir, ACESTEP_MODEL_NAME)
    checkpoints = []

    if os.path.exists(model_dir):
        for name in os.listdir(model_dir):
            full_path = os.path.join(model_dir, name)
            if os.path.isdir(full_path):
                checkpoints.append(name)

    # Filter to only include base model
    base_checkpoints = [c for c in checkpoints if "base" in c.lower()]
    if not base_checkpoints:
        base_checkpoints = ["acestep-v15-base"]

    return base_checkpoints


def get_acestep_lm_models():
    """Get available LM models."""
    model_dir = os.path.join(folder_paths.models_dir, ACESTEP_MODEL_NAME)
    models = []

    if os.path.exists(model_dir):
        for name in os.listdir(model_dir):
            full_path = os.path.join(model_dir, name)
            if os.path.isdir(full_path) and "lm" in name.lower():
                models.append(name)

    if not models:
        models = ["acestep-5Hz-lm-1.7B", "acestep-5Hz-lm-0.6B", "acestep-5Hz-lm-4B"]

    return sorted(models)


# Handler cache
_dit_handler = None
_llm_handler = None
_handlers_initialized = False


def resolve_checkpoint_path(name: str) -> str:
    """Resolve a checkpoint name to its full path."""
    model_dir = os.path.join(folder_paths.models_dir, ACESTEP_MODEL_NAME)
    full_path = os.path.join(model_dir, name)
    if os.path.exists(full_path):
        return full_path
    return name


def auto_detect_device() -> str:
    """Auto-detect the best available device."""
    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def initialize_handlers(checkpoint_dir: str, config_path: str, lm_model_path: str, device: str = "auto"):
    """Initialize ACE-Step handlers using ACEStepWrapper."""
    global _dit_handler, _llm_handler, _handlers_initialized

    if _handlers_initialized:
        if _dit_handler and getattr(_dit_handler, "model", None) is not None:
            return _dit_handler, _llm_handler

    if device == "auto":
        device = auto_detect_device()

    # Get the root model directory (Ace-Step1.5)
    # checkpoint_dir from UI is like "acestep-v15-base", but we need the parent directory
    model_root = os.path.join(folder_paths.models_dir, ACESTEP_MODEL_NAME)

    # config_path is the actual model subdirectory (e.g., "acestep-v15-base")
    # If checkpoint_dir and config_path are the same, use model_root as checkpoint_dir
    if checkpoint_dir == config_path or not config_path:
        actual_checkpoint_dir = model_root
        actual_config_path = checkpoint_dir
    else:
        actual_checkpoint_dir = os.path.join(model_root, checkpoint_dir) if not os.path.isabs(checkpoint_dir) else checkpoint_dir
        actual_config_path = config_path

    # Check if model directory exists
    if not os.path.exists(actual_checkpoint_dir):
        raise RuntimeError(
            f"Model directory not found: {actual_checkpoint_dir}\n"
            f"Please download ACE-Step models to: {model_root}\n"
            f"See https://github.com/ACE-Step/Ace-Step1.5"
        )

    # Use ACEStepWrapper to initialize DiT components (avoids hardcoded checkpoints subdirectory)
    _dit_handler = AceStepHandler()
    wrapper = ACEStepWrapper()
    wrapper.initialize(
        checkpoint_dir=actual_checkpoint_dir,
        config_path=actual_config_path,
        device=device,
    )

    # Copy wrapper attributes to dit_handler
    for attr in ['device', 'dtype', 'offload_to_cpu', 'model', 'vae',
                 'text_encoder', 'text_tokenizer', 'silence_latent']:
        if hasattr(wrapper, attr):
            setattr(_dit_handler, attr, getattr(wrapper, attr))

    _dit_handler.offload_dit_to_cpu = False
    _dit_handler.config = _dit_handler.model.config
    _dit_handler.quantization = None

    # Initialize LLM handler
    lm_model_path_resolved = os.path.join(actual_checkpoint_dir, lm_model_path)
    if not os.path.exists(lm_model_path_resolved):
        lm_model_path_resolved = lm_model_path

    _llm_handler = LLMHandler()
    _llm_handler.initialize(
        checkpoint_dir=actual_checkpoint_dir,
        lm_model_path=lm_model_path_resolved,
        backend="pt",
        device=device,
        dtype=_dit_handler.dtype,
    )

    _handlers_initialized = True
    print(f"[ACE-Step Base Features] Handlers initialized successfully")
    return _dit_handler, _llm_handler


class ACE_STEP_EXTRACT:
    """
    Extract a specific track/instrument from audio.
    IMPORTANT: Only works with acestep-v15-base model, NOT turbo!
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "src_audio": ("AUDIO", {"tooltip": "Source audio to extract track from."}),
                "track_name": (TRACK_NAMES, {"default": "vocals", "tooltip": "Track to extract."}),
                "checkpoint_dir": (get_acestep_base_checkpoints(), {"default": "acestep-v15-base", "tooltip": "Must use base model, not turbo."}),
                "config_path": (["acestep-v15-base"], {"default": "acestep-v15-base", "tooltip": "Model config name."}),
                "lm_model_path": (get_acestep_lm_models(), {"default": "acestep-5Hz-lm-1.7B", "tooltip": "Language model for reasoning."}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFffffffff, "tooltip": "Random seed. -1 for random."}),
                "inference_steps": ("INT", {"default": 50, "min": 20, "max": 100, "tooltip": "Diffusion steps. Higher = better quality."}),
                "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto", "tooltip": "Compute device."}),
            },
            "optional": {
                "guidance_scale": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 15.0, "tooltip": "CFG scale. Higher = more prompt adherence."}),
                "audio_format": (["flac", "mp3", "wav"], {"default": "flac", "tooltip": "Output audio format."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio", "audio_path", "metadata")
    FUNCTION = "extract"
    CATEGORY = "Audio/ACE-Step"

    def extract(self, src_audio, track_name, checkpoint_dir, config_path, lm_model_path, seed, inference_steps, device, guidance_scale=7.0, audio_format="flac"):
        from acestep.inference import generate_music, GenerationParams, GenerationConfig

        print(f"[ACE_STEP_EXTRACT] Starting extraction...")
        print(f"  - Track: {track_name}")
        print(f"  - Inference steps: {inference_steps}")
        print(f"  - Device: {device}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        dit_handler, llm_handler = initialize_handlers(checkpoint_dir, config_path, lm_model_path, device)

        # Save input audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = tmp.name
            waveform = src_audio["waveform"].squeeze(0).numpy().T
            sf.write(temp_path, waveform, src_audio["sample_rate"])

        try:
            instruction = f"Extract the {track_name.upper()} track from the audio:"
            print(f"[ACE_STEP_EXTRACT] Running diffusion...")

            # Create progress callback for ComfyUI
            progress_callback = ComfyProgressCallback(total_steps=100)

            params = GenerationParams(
                task_type="extract",
                src_audio=temp_path,
                caption=f"Extract {track_name} track",
                instruction=instruction,
                inference_steps=inference_steps,
                seed=seed,
                guidance_scale=guidance_scale,
            )

            config = GenerationConfig(batch_size=1, use_random_seed=(seed == -1), audio_format=audio_format)

            output_dir = folder_paths.get_output_directory()
            result = generate_music(dit_handler, llm_handler, params, config, save_dir=output_dir, progress=progress_callback)

            if not result.success:
                raise RuntimeError(f"Extract failed: {result.error}")

            audio_data = result.audios[0]
            audio_output = {
                "waveform": audio_data["tensor"].cpu().unsqueeze(0),
                "sample_rate": audio_data["sample_rate"],
            }

            metadata = json.dumps({
                "task_type": "extract",
                "track_name": track_name,
                "seed": audio_data["params"].get("seed", seed),
            }, indent=2)

            return audio_output, audio_data["path"], metadata

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class ACE_STEP_LEGO:
    """
    Generate a specific track based on audio context.
    IMPORTANT: Only works with acestep-v15-base model, NOT turbo!
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "src_audio": ("AUDIO", {"tooltip": "Source audio as context."}),
                "track_name": (TRACK_NAMES, {"default": "drums", "tooltip": "Track to generate."}),
                "caption": ("STRING", {"default": "", "multiline": True, "tooltip": "Style description (optional)."}),
                "checkpoint_dir": (get_acestep_base_checkpoints(), {"default": "acestep-v15-base", "tooltip": "Must use base model, not turbo."}),
                "config_path": (["acestep-v15-base"], {"default": "acestep-v15-base", "tooltip": "Model config name."}),
                "lm_model_path": (get_acestep_lm_models(), {"default": "acestep-5Hz-lm-1.7B", "tooltip": "Language model for reasoning."}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFffffffff, "tooltip": "Random seed. -1 for random."}),
                "inference_steps": ("INT", {"default": 50, "min": 20, "max": 100, "tooltip": "Diffusion steps. Higher = better quality."}),
                "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto", "tooltip": "Compute device."}),
            },
            "optional": {
                "repainting_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 600.0, "tooltip": "Start time for repainting region (seconds)."}),
                "repainting_end": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 600.0, "tooltip": "End time for repainting region. -1 for until end."}),
                "guidance_scale": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 15.0, "tooltip": "CFG scale. Higher = more prompt adherence."}),
                "audio_format": (["flac", "mp3", "wav"], {"default": "flac", "tooltip": "Output audio format."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio", "audio_path", "metadata")
    FUNCTION = "lego"
    CATEGORY = "Audio/ACE-Step"

    def lego(self, src_audio, track_name, caption, checkpoint_dir, config_path, lm_model_path, seed, inference_steps, device, repainting_start=0.0, repainting_end=-1.0, guidance_scale=7.0, audio_format="flac"):
        from acestep.inference import generate_music, GenerationParams, GenerationConfig

        print(f"[ACE_STEP_LEGO] Starting generation...")
        print(f"  - Track: {track_name}")
        print(f"  - Caption: {caption[:50] if caption else '(auto)'}")
        print(f"  - Inference steps: {inference_steps}")
        print(f"  - Device: {device}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        dit_handler, llm_handler = initialize_handlers(checkpoint_dir, config_path, lm_model_path, device)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = tmp.name
            waveform = src_audio["waveform"].squeeze(0).numpy().T
            sf.write(temp_path, waveform, src_audio["sample_rate"])

        try:
            print(f"[ACE_STEP_LEGO] Running diffusion...")
            instruction = f"Generate the {track_name.upper()} track based on the audio context:"

            # Create progress callback for ComfyUI
            progress_callback = ComfyProgressCallback(total_steps=100)

            params = GenerationParams(
                task_type="lego",
                src_audio=temp_path,
                caption=caption if caption else f"Generate {track_name} track",
                instruction=instruction,
                repainting_start=repainting_start,
                repainting_end=repainting_end if repainting_end > 0 else -1.0,
                inference_steps=inference_steps,
                seed=seed,
                guidance_scale=guidance_scale,
            )

            config = GenerationConfig(batch_size=1, use_random_seed=(seed == -1), audio_format=audio_format)

            output_dir = folder_paths.get_output_directory()
            result = generate_music(dit_handler, llm_handler, params, config, save_dir=output_dir, progress=progress_callback)

            if not result.success:
                raise RuntimeError(f"Lego failed: {result.error}")

            audio_data = result.audios[0]
            audio_output = {
                "waveform": audio_data["tensor"].cpu().unsqueeze(0),
                "sample_rate": audio_data["sample_rate"],
            }

            metadata = json.dumps({
                "task_type": "lego",
                "track_name": track_name,
                "repainting_start": repainting_start,
                "repainting_end": repainting_end,
                "seed": audio_data["params"].get("seed", seed),
            }, indent=2)

            return audio_output, audio_data["path"], metadata

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class ACE_STEP_COMPLETE:
    """
    Complete input track with other instruments.
    IMPORTANT: Only works with acestep-v15-base model, NOT turbo!
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "src_audio": ("AUDIO", {"tooltip": "Source audio to complete."}),
                "checkpoint_dir": (get_acestep_base_checkpoints(), {"default": "acestep-v15-base", "tooltip": "Model checkpoint directory."}),
                "config_path": (["acestep-v15-base"], {"default": "acestep-v15-base", "tooltip": "Model config name."}),
                "lm_model_path": (get_acestep_lm_models(), {"default": "acestep-5Hz-lm-1.7B", "tooltip": "Language model for reasoning."}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFffffffff, "tooltip": "Random seed. -1 for random."}),
                "inference_steps": ("INT", {"default": 50, "min": 20, "max": 100, "tooltip": "Diffusion steps. Higher = better quality."}),
                "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto", "tooltip": "Compute device."}),
            },
            "optional": {
                # Track selection as boolean switches
                "add_drums": ("BOOLEAN", {"default": True, "tooltip": "Add drums track"}),
                "add_bass": ("BOOLEAN", {"default": True, "tooltip": "Add bass track"}),
                "add_guitar": ("BOOLEAN", {"default": False, "tooltip": "Add guitar track"}),
                "add_keyboard": ("BOOLEAN", {"default": False, "tooltip": "Add keyboard/piano track"}),
                "add_strings": ("BOOLEAN", {"default": True, "tooltip": "Add strings track"}),
                "add_synths": ("BOOLEAN", {"default": False, "tooltip": "Add synthesizer track"}),
                "add_percussion": ("BOOLEAN", {"default": False, "tooltip": "Add percussion track"}),
                "add_brass": ("BOOLEAN", {"default": False, "tooltip": "Add brass track"}),
                "add_woodwinds": ("BOOLEAN", {"default": False, "tooltip": "Add woodwinds track"}),
                "add_backing_vocals": ("BOOLEAN", {"default": False, "tooltip": "Add backing vocals track"}),
                "add_fx": ("BOOLEAN", {"default": False, "tooltip": "Add FX/sound effects track"}),
                "add_vocals": ("BOOLEAN", {"default": False, "tooltip": "Add vocals track"}),
                # Vocal settings (important when add_vocals is True)
                "vocal_language": (VOCAL_LANGUAGES, {"default": "unknown", "tooltip": "Language for vocals. Set when add_vocals=True."}),
                "lyrics": ("STRING", {"default": "", "multiline": True, "tooltip": "Lyrics text. Set when add_vocals=True."}),
                # Other optional parameters
                "caption": ("STRING", {"default": "", "multiline": True, "tooltip": "Style description (optional)."}),
                "guidance_scale": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 15.0, "tooltip": "CFG scale. Higher = more prompt adherence."}),
                "audio_format": (["flac", "mp3", "wav"], {"default": "flac", "tooltip": "Output audio format."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio", "audio_path", "metadata")
    FUNCTION = "complete"
    CATEGORY = "Audio/ACE-Step"

    def complete(
        self,
        src_audio,
        checkpoint_dir,
        config_path,
        lm_model_path,
        seed,
        inference_steps,
        device,
        add_drums=True,
        add_bass=True,
        add_guitar=False,
        add_keyboard=False,
        add_strings=True,
        add_synths=False,
        add_percussion=False,
        add_brass=False,
        add_woodwinds=False,
        add_backing_vocals=False,
        add_fx=False,
        add_vocals=False,
        vocal_language="unknown",
        lyrics="",
        caption="",
        guidance_scale=7.0,
        audio_format="flac",
    ):
        from acestep.inference import generate_music, GenerationParams, GenerationConfig

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Build track list from boolean parameters
        # IMPORTANT: Use lowercase track names to match official TRACK_NAMES
        track_list = []
        if add_drums:
            track_list.append("drums")
        if add_bass:
            track_list.append("bass")
        if add_guitar:
            track_list.append("guitar")
        if add_keyboard:
            track_list.append("keyboard")
        if add_strings:
            track_list.append("strings")
        if add_synths:
            track_list.append("synth")
        if add_percussion:
            track_list.append("percussion")
        if add_brass:
            track_list.append("brass")
        if add_woodwinds:
            track_list.append("woodwinds")
        if add_backing_vocals:
            track_list.append("backing_vocals")
        if add_fx:
            track_list.append("fx")
        if add_vocals:
            track_list.append("vocals")

        if not track_list:
            raise ValueError("Please select at least one track to add.")

        track_classes_str = ", ".join(track_list)

        print(f"[ACE_STEP_COMPLETE] Starting generation...")
        print(f"  - Tracks to add: {track_classes_str}")
        print(f"  - Vocal language: {vocal_language}")
        print(f"  - Inference steps: {inference_steps}")
        print(f"  - Device: {device}")

        dit_handler, llm_handler = initialize_handlers(checkpoint_dir, config_path, lm_model_path, device)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = tmp.name
            waveform = src_audio["waveform"].squeeze(0).numpy().T
            sf.write(temp_path, waveform, src_audio["sample_rate"])

        try:
            instruction = f"Complete the input track with {track_classes_str}:"
            print(f"[ACE_STEP_COMPLETE] Running diffusion...")

            # Create progress callback for ComfyUI
            progress_callback = ComfyProgressCallback(total_steps=100)

            params = GenerationParams(
                task_type="complete",
                src_audio=temp_path,
                caption=caption if caption else f"Complete with {track_classes_str}",
                instruction=instruction,
                inference_steps=inference_steps,
                seed=seed,
                guidance_scale=guidance_scale,
                vocal_language=vocal_language,
                lyrics=lyrics if lyrics else "",
            )

            config = GenerationConfig(batch_size=1, use_random_seed=(seed == -1), audio_format=audio_format)

            output_dir = folder_paths.get_output_directory()
            result = generate_music(dit_handler, llm_handler, params, config, save_dir=output_dir, progress=progress_callback)

            if not result.success:
                raise RuntimeError(f"Complete failed: {result.error}")

            audio_data = result.audios[0]
            audio_output = {
                "waveform": audio_data["tensor"].cpu().unsqueeze(0),
                "sample_rate": audio_data["sample_rate"],
            }

            metadata = json.dumps({
                "task_type": "complete",
                "track_classes": track_classes_str,
                "seed": audio_data["params"].get("seed", seed),
            }, indent=2)

            return audio_output, audio_data["path"], metadata

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
