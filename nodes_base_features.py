"""
ACE-Step Base Model Features

These features are ONLY available with acestep-v15-base model (NOT turbo):
- Extract: Extract a specific track/instrument from audio
- Lego: Generate a specific track based on audio context
- Complete: Complete input track with other instruments

Track names: woodwinds, brass, fx, synth, strings, percussion, keyboard, guitar, bass, drums, backing_vocals, vocals
"""

import os
import torch
import tempfile
import soundfile as sf
from typing import Dict, Any, Tuple, Optional

import folder_paths

# Constants
ACESTEP_MODEL_NAME = "Ace-Step1.5"

# Available track names for Extract/Lego/Complete
TRACK_NAMES = [
    "vocals", "backing_vocals", "drums", "bass", "guitar",
    "keyboard", "percussion", "strings", "synth", "fx", "brass", "woodwinds"
]


def get_acestep_base_checkpoints():
    """Get available base model checkpoints."""
    from nodes import get_acestep_checkpoints
    checkpoints = get_acestep_checkpoints()
    # Filter to only include base model
    base_checkpoints = [c for c in checkpoints if "base" in c.lower()]
    if not base_checkpoints:
        base_checkpoints = ["acestep-v15-base"]
    return base_checkpoints


class ACE_STEP_EXTRACT:
    """
    Extract a specific track/instrument from audio.

    This node extracts a specific instrument or vocal track from a mixed audio file.
    Example: Extract vocals, drums, bass, etc. from a song.

    IMPORTANT: Only works with acestep-v15-base model, NOT turbo!
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "src_audio": ("AUDIO", {"tooltip": "Source audio to extract track from."}),
                "track_name": (TRACK_NAMES, {
                    "default": "vocals",
                    "tooltip": "Track/instrument to extract: vocals, drums, bass, guitar, etc."
                }),
                "checkpoint_dir": (get_acestep_base_checkpoints(), {
                    "default": "acestep-v15-base",
                    "tooltip": "Must use acestep-v15-base model (NOT turbo)."
                }),
                "config_path": (["acestep-v15-base"], {
                    "default": "acestep-v15-base",
                    "tooltip": "Must be acestep-v15-base for this feature."
                }),
                "lm_model_path": (["acestep-5Hz-lm-1.7B", "acestep-5Hz-lm-0.6B", "acestep-5Hz-lm-4B"], {
                    "default": "acestep-5Hz-lm-1.7B",
                    "tooltip": "Language model for metadata generation."
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0xFFFFFFFFffffffff,
                    "control_after_generate": True,
                    "tooltip": "Random seed for reproducibility."
                }),
                "inference_steps": ("INT", {
                    "default": 50,
                    "min": 20,
                    "max": 100,
                    "tooltip": "Diffusion steps. Base model needs 50+ steps."
                }),
                "device": (["auto", "cuda", "cpu", "mps", "xpu"], {
                    "default": "auto",
                    "tooltip": "Computing platform."
                }),
            },
            "optional": {
                "guidance_scale": ("FLOAT", {
                    "default": 7.0,
                    "min": 1.0,
                    "max": 15.0,
                    "tooltip": "CFG strength."
                }),
                "audio_format": (["flac", "mp3", "wav"], {
                    "default": "flac",
                    "tooltip": "Output audio format."
                }),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio", "audio_path", "metadata")
    FUNCTION = "extract"
    CATEGORY = "Audio/ACE-Step"

    def extract(
        self,
        src_audio: Dict[str, Any],
        track_name: str,
        checkpoint_dir: str,
        config_path: str,
        lm_model_path: str,
        seed: int,
        inference_steps: int,
        device: str,
        guidance_scale: float = 7.0,
        audio_format: str = "flac",
    ) -> Tuple[Dict[str, Any], str, str]:
        from nodes import ACE_STEP_BASE, generate_music, GenerationConfig, create_generation_params
        import json

        # Create a temporary instance to use the base class methods
        base_node = ACE_STEP_BASE()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save input audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = tmp.name
            waveform = src_audio["waveform"].squeeze(0).numpy().T
            sf.write(temp_path, waveform, src_audio["sample_rate"])

        try:
            # Initialize handlers
            dit_handler, llm_handler = base_node.initialize_handlers(
                checkpoint_dir=checkpoint_dir,
                config_path=config_path,
                lm_model_path=lm_model_path,
                device=device,
            )

            # Build extract instruction
            instruction = f"Extract the {track_name.upper()} track from the audio:"

            # Prepare generation parameters
            params = create_generation_params(
                task_type="extract",
                src_audio=temp_path,
                caption=f"Extract {track_name} track",
                instruction=instruction,
                inference_steps=inference_steps,
                seed=seed,
                guidance_scale=guidance_scale,
                thinking=False,  # Extract doesn't need LM reasoning
            )

            # Prepare generation config
            config = GenerationConfig(
                batch_size=1,
                use_random_seed=(seed == -1),
                audio_format=audio_format,
            )

            # Generate
            output_dir = folder_paths.get_output_directory()
            result = generate_music(dit_handler, llm_handler, params, config, save_dir=output_dir)

            if not result.success:
                raise RuntimeError(f"Extract failed: {result.error}")

            # Get the audio
            audio_data = result.audios[0]
            audio_path = audio_data["path"]
            audio_tensor = audio_data["tensor"]
            sample_rate = audio_data["sample_rate"]

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Prepare ComfyUI audio format
            audio_output = {
                "waveform": audio_tensor.cpu().unsqueeze(0),
                "sample_rate": sample_rate,
            }

            # Prepare metadata
            metadata = json.dumps({
                "task_type": "extract",
                "track_name": track_name,
                "seed": audio_data["params"].get("seed", seed),
                "sample_rate": sample_rate,
            }, indent=2)

            return audio_output, audio_path, metadata

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class ACE_STEP_LEGO:
    """
    Generate a specific track based on audio context.

    This node generates a new instrument/vocal track that fits with the existing audio.
    You can also specify a time region to regenerate (repainting).

    IMPORTANT: Only works with acestep-v15-base model, NOT turbo!
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "src_audio": ("AUDIO", {"tooltip": "Source audio as context for generation."}),
                "track_name": (TRACK_NAMES, {
                    "default": "drums",
                    "tooltip": "Track/instrument to generate: drums, bass, guitar, etc."
                }),
                "caption": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Description of the track style to generate."
                }),
                "checkpoint_dir": (get_acestep_base_checkpoints(), {
                    "default": "acestep-v15-base",
                    "tooltip": "Must use acestep-v15-base model (NOT turbo)."
                }),
                "config_path": (["acestep-v15-base"], {
                    "default": "acestep-v15-base",
                    "tooltip": "Must be acestep-v15-base for this feature."
                }),
                "lm_model_path": (["acestep-5Hz-lm-1.7B", "acestep-5Hz-lm-0.6B", "acestep-5Hz-lm-4B"], {
                    "default": "acestep-5Hz-lm-1.7B",
                    "tooltip": "Language model for metadata generation."
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0xFFFFFFFFffffffff,
                    "control_after_generate": True,
                    "tooltip": "Random seed for reproducibility."
                }),
                "inference_steps": ("INT", {
                    "default": 50,
                    "min": 20,
                    "max": 100,
                    "tooltip": "Diffusion steps. Base model needs 50+ steps."
                }),
                "device": (["auto", "cuda", "cpu", "mps", "xpu"], {
                    "default": "auto",
                    "tooltip": "Computing platform."
                }),
            },
            "optional": {
                "repainting_start": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 600.0,
                    "tooltip": "Start time for the region to regenerate (seconds)."
                }),
                "repainting_end": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 600.0,
                    "tooltip": "End time for the region to regenerate (-1 for until end)."
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 7.0,
                    "min": 1.0,
                    "max": 15.0,
                    "tooltip": "CFG strength."
                }),
                "audio_format": (["flac", "mp3", "wav"], {
                    "default": "flac",
                    "tooltip": "Output audio format."
                }),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio", "audio_path", "metadata")
    FUNCTION = "lego"
    CATEGORY = "Audio/ACE-Step"

    def lego(
        self,
        src_audio: Dict[str, Any],
        track_name: str,
        caption: str,
        checkpoint_dir: str,
        config_path: str,
        lm_model_path: str,
        seed: int,
        inference_steps: int,
        device: str,
        repainting_start: float = 0.0,
        repainting_end: float = -1.0,
        guidance_scale: float = 7.0,
        audio_format: str = "flac",
    ) -> Tuple[Dict[str, Any], str, str]:
        from nodes import ACE_STEP_BASE, generate_music, GenerationConfig, create_generation_params
        import json

        base_node = ACE_STEP_BASE()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = tmp.name
            waveform = src_audio["waveform"].squeeze(0).numpy().T
            sf.write(temp_path, waveform, src_audio["sample_rate"])

        try:
            dit_handler, llm_handler = base_node.initialize_handlers(
                checkpoint_dir=checkpoint_dir,
                config_path=config_path,
                lm_model_path=lm_model_path,
                device=device,
            )

            # Build lego instruction
            instruction = f"Generate the {track_name.upper()} track based on the audio context:"

            params = create_generation_params(
                task_type="lego",
                src_audio=temp_path,
                caption=caption if caption else f"Generate {track_name} track",
                instruction=instruction,
                repainting_start=repainting_start,
                repainting_end=repainting_end if repainting_end > 0 else -1.0,
                inference_steps=inference_steps,
                seed=seed,
                guidance_scale=guidance_scale,
                thinking=False,
            )

            config = GenerationConfig(
                batch_size=1,
                use_random_seed=(seed == -1),
                audio_format=audio_format,
            )

            output_dir = folder_paths.get_output_directory()
            result = generate_music(dit_handler, llm_handler, params, config, save_dir=output_dir)

            if not result.success:
                raise RuntimeError(f"Lego failed: {result.error}")

            audio_data = result.audios[0]
            audio_path = audio_data["path"]
            audio_tensor = audio_data["tensor"]
            sample_rate = audio_data["sample_rate"]

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            audio_output = {
                "waveform": audio_tensor.cpu().unsqueeze(0),
                "sample_rate": sample_rate,
            }

            metadata = json.dumps({
                "task_type": "lego",
                "track_name": track_name,
                "repainting_start": repainting_start,
                "repainting_end": repainting_end,
                "seed": audio_data["params"].get("seed", seed),
                "sample_rate": sample_rate,
            }, indent=2)

            return audio_output, audio_path, metadata

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class ACE_STEP_COMPLETE:
    """
    Complete input track with other instruments.

    This node takes a partial audio track and completes it by adding
    other instruments/tracks. Great for turning a melody into a full arrangement.

    IMPORTANT: Only works with acestep-v15-base model, NOT turbo!
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "src_audio": ("AUDIO", {"tooltip": "Source audio to complete."}),
                "track_classes": ("STRING", {
                    "default": "drums, bass, strings",
                    "tooltip": "Comma-separated list of tracks to add: drums, bass, strings, synth, etc."
                }),
                "caption": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Description of the music style."
                }),
                "checkpoint_dir": (get_acestep_base_checkpoints(), {
                    "default": "acestep-v15-base",
                    "tooltip": "Must use acestep-v15-base model (NOT turbo)."
                }),
                "config_path": (["acestep-v15-base"], {
                    "default": "acestep-v15-base",
                    "tooltip": "Must be acestep-v15-base for this feature."
                }),
                "lm_model_path": (["acestep-5Hz-lm-1.7B", "acestep-5Hz-lm-0.6B", "acestep-5Hz-lm-4B"], {
                    "default": "acestep-5Hz-lm-1.7B",
                    "tooltip": "Language model for metadata generation."
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0xFFFFFFFFffffffff,
                    "control_after_generate": True,
                    "tooltip": "Random seed for reproducibility."
                }),
                "inference_steps": ("INT", {
                    "default": 50,
                    "min": 20,
                    "max": 100,
                    "tooltip": "Diffusion steps. Base model needs 50+ steps."
                }),
                "device": (["auto", "cuda", "cpu", "mps", "xpu"], {
                    "default": "auto",
                    "tooltip": "Computing platform."
                }),
            },
            "optional": {
                "guidance_scale": ("FLOAT", {
                    "default": 7.0,
                    "min": 1.0,
                    "max": 15.0,
                    "tooltip": "CFG strength."
                }),
                "audio_format": (["flac", "mp3", "wav"], {
                    "default": "flac",
                    "tooltip": "Output audio format."
                }),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio", "audio_path", "metadata")
    FUNCTION = "complete"
    CATEGORY = "Audio/ACE-Step"

    def complete(
        self,
        src_audio: Dict[str, Any],
        track_classes: str,
        caption: str,
        checkpoint_dir: str,
        config_path: str,
        lm_model_path: str,
        seed: int,
        inference_steps: int,
        device: str,
        guidance_scale: float = 7.0,
        audio_format: str = "flac",
    ) -> Tuple[Dict[str, Any], str, str]:
        from nodes import ACE_STEP_BASE, generate_music, GenerationConfig, create_generation_params
        import json

        base_node = ACE_STEP_BASE()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = tmp.name
            waveform = src_audio["waveform"].squeeze(0).numpy().T
            sf.write(temp_path, waveform, src_audio["sample_rate"])

        try:
            dit_handler, llm_handler = base_node.initialize_handlers(
                checkpoint_dir=checkpoint_dir,
                config_path=config_path,
                lm_model_path=lm_model_path,
                device=device,
            )

            # Parse track classes and format instruction
            track_list = [t.strip().upper() for t in track_classes.split(",") if t.strip()]
            track_classes_str = ", ".join(track_list)
            instruction = f"Complete the input track with {track_classes_str}:"

            params = create_generation_params(
                task_type="complete",
                src_audio=temp_path,
                caption=caption if caption else f"Complete with {track_classes_str}",
                instruction=instruction,
                inference_steps=inference_steps,
                seed=seed,
                guidance_scale=guidance_scale,
                thinking=False,
            )

            config = GenerationConfig(
                batch_size=1,
                use_random_seed=(seed == -1),
                audio_format=audio_format,
            )

            output_dir = folder_paths.get_output_directory()
            result = generate_music(dit_handler, llm_handler, params, config, save_dir=output_dir)

            if not result.success:
                raise RuntimeError(f"Complete failed: {result.error}")

            audio_data = result.audios[0]
            audio_path = audio_data["path"]
            audio_tensor = audio_data["tensor"]
            sample_rate = audio_data["sample_rate"]

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            audio_output = {
                "waveform": audio_tensor.cpu().unsqueeze(0),
                "sample_rate": sample_rate,
            }

            metadata = json.dumps({
                "task_type": "complete",
                "track_classes": track_classes,
                "seed": audio_data["params"].get("seed", seed),
                "sample_rate": sample_rate,
            }, indent=2)

            return audio_output, audio_path, metadata

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
