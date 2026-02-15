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

# Add repo path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "acestep_repo"))

# Check if acestep is available
try:
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import generate_music, GenerationParams, GenerationConfig
    ACESTEP_BASE_AVAILABLE = True
except ImportError:
    ACESTEP_BASE_AVAILABLE = False
    print("[ACE-Step Base Features] acestep module not found. Please ensure acestep_repo is downloaded.")

# Constants
ACESTEP_MODEL_NAME = "Ace-Step1.5"

# Available track names for Extract/Lego/Complete
TRACK_NAMES = [
    "vocals", "backing_vocals", "drums", "bass", "guitar",
    "keyboard", "percussion", "strings", "synth", "fx", "brass", "woodwinds"
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
    """Initialize ACE-Step handlers."""
    global _dit_handler, _llm_handler, _handlers_initialized

    if device == "auto":
        device = auto_detect_device()

    # Resolve paths
    checkpoint_dir = resolve_checkpoint_path(checkpoint_dir)
    lm_model_path_resolved = os.path.join(checkpoint_dir, lm_model_path)
    if not os.path.exists(lm_model_path_resolved):
        lm_model_path_resolved = lm_model_path

    # Import acestep modules
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler

    # Initialize DiT handler
    _dit_handler = AceStepHandler()
    _dit_handler.initialize_service(
        project_root=checkpoint_dir,
        config_path=config_path,
        device=device,
    )

    # Initialize LLM handler
    _llm_handler = LLMHandler()
    _llm_handler.initialize(
        checkpoint_dir=checkpoint_dir,
        lm_model_path=lm_model_path_resolved,
        backend="pt",
        device=device,
        dtype=_dit_handler.dtype,
    )

    _handlers_initialized = True
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
                "checkpoint_dir": (get_acestep_base_checkpoints(), {"default": "acestep-v15-base", "tooltip": "Must use base model."}),
                "config_path": (["acestep-v15-base"], {"default": "acestep-v15-base"}),
                "lm_model_path": (get_acestep_lm_models(), {"default": "acestep-5Hz-lm-1.7B"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFffffffff}),
                "inference_steps": ("INT", {"default": 50, "min": 20, "max": 100}),
                "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto"}),
            },
            "optional": {
                "guidance_scale": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 15.0}),
                "audio_format": (["flac", "mp3", "wav"], {"default": "flac"}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio", "audio_path", "metadata")
    FUNCTION = "extract"
    CATEGORY = "Audio/ACE-Step"

    def extract(self, src_audio, track_name, checkpoint_dir, config_path, lm_model_path, seed, inference_steps, device, guidance_scale=7.0, audio_format="flac"):
        from acestep.inference import generate_music, GenerationParams, GenerationConfig

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
            result = generate_music(dit_handler, llm_handler, params, config, save_dir=output_dir)

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
                "caption": ("STRING", {"default": "", "multiline": True, "tooltip": "Style description."}),
                "checkpoint_dir": (get_acestep_base_checkpoints(), {"default": "acestep-v15-base"}),
                "config_path": (["acestep-v15-base"], {"default": "acestep-v15-base"}),
                "lm_model_path": (get_acestep_lm_models(), {"default": "acestep-5Hz-lm-1.7B"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFffffffff}),
                "inference_steps": ("INT", {"default": 50, "min": 20, "max": 100}),
                "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto"}),
            },
            "optional": {
                "repainting_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 600.0}),
                "repainting_end": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 600.0}),
                "guidance_scale": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 15.0}),
                "audio_format": (["flac", "mp3", "wav"], {"default": "flac"}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio", "audio_path", "metadata")
    FUNCTION = "lego"
    CATEGORY = "Audio/ACE-Step"

    def lego(self, src_audio, track_name, caption, checkpoint_dir, config_path, lm_model_path, seed, inference_steps, device, repainting_start=0.0, repainting_end=-1.0, guidance_scale=7.0, audio_format="flac"):
        from acestep.inference import generate_music, GenerationParams, GenerationConfig

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        dit_handler, llm_handler = initialize_handlers(checkpoint_dir, config_path, lm_model_path, device)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = tmp.name
            waveform = src_audio["waveform"].squeeze(0).numpy().T
            sf.write(temp_path, waveform, src_audio["sample_rate"])

        try:
            instruction = f"Generate the {track_name.upper()} track based on the audio context:"

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
            result = generate_music(dit_handler, llm_handler, params, config, save_dir=output_dir)

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
                "track_classes": ("STRING", {"default": "drums, bass, strings", "tooltip": "Tracks to add."}),
                "caption": ("STRING", {"default": "", "multiline": True, "tooltip": "Style description."}),
                "checkpoint_dir": (get_acestep_base_checkpoints(), {"default": "acestep-v15-base"}),
                "config_path": (["acestep-v15-base"], {"default": "acestep-v15-base"}),
                "lm_model_path": (get_acestep_lm_models(), {"default": "acestep-5Hz-lm-1.7B"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFffffffff}),
                "inference_steps": ("INT", {"default": 50, "min": 20, "max": 100}),
                "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto"}),
            },
            "optional": {
                "guidance_scale": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 15.0}),
                "audio_format": (["flac", "mp3", "wav"], {"default": "flac"}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio", "audio_path", "metadata")
    FUNCTION = "complete"
    CATEGORY = "Audio/ACE-Step"

    def complete(self, src_audio, track_classes, caption, checkpoint_dir, config_path, lm_model_path, seed, inference_steps, device, guidance_scale=7.0, audio_format="flac"):
        from acestep.inference import generate_music, GenerationParams, GenerationConfig

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        dit_handler, llm_handler = initialize_handlers(checkpoint_dir, config_path, lm_model_path, device)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = tmp.name
            waveform = src_audio["waveform"].squeeze(0).numpy().T
            sf.write(temp_path, waveform, src_audio["sample_rate"])

        try:
            track_list = [t.strip().upper() for t in track_classes.split(",") if t.strip()]
            track_classes_str = ", ".join(track_list)
            instruction = f"Complete the input track with {track_classes_str}:"

            params = GenerationParams(
                task_type="complete",
                src_audio=temp_path,
                caption=caption if caption else f"Complete with {track_classes_str}",
                instruction=instruction,
                inference_steps=inference_steps,
                seed=seed,
                guidance_scale=guidance_scale,
            )

            config = GenerationConfig(batch_size=1, use_random_seed=(seed == -1), audio_format=audio_format)

            output_dir = folder_paths.get_output_directory()
            result = generate_music(dit_handler, llm_handler, params, config, save_dir=output_dir)

            if not result.success:
                raise RuntimeError(f"Complete failed: {result.error}")

            audio_data = result.audios[0]
            audio_output = {
                "waveform": audio_data["tensor"].cpu().unsqueeze(0),
                "sample_rate": audio_data["sample_rate"],
            }

            metadata = json.dumps({
                "task_type": "complete",
                "track_classes": track_classes,
                "seed": audio_data["params"].get("seed", seed),
            }, indent=2)

            return audio_output, audio_data["path"], metadata

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
