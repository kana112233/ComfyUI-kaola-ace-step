"""
ACE-Step 1.5 Music Generation Nodes
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import folder_paths

comfy_path = folder_paths.__file__.replace("__init__.py", "")
import sys
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
except ImportError:
    ACESTEP_AVAILABLE = False
    print("ACE-Step not installed. Please install it first: pip install acestep")

# Register ACE-Step model directory with ComfyUI
# Models should be placed in: ComfyUI/models/Ace-Step1.5/
ACESTEP_MODEL_NAME = "Ace-Step1.5"
if ACESTEP_AVAILABLE:
    folder_paths.add_model_folder_path(ACESTEP_MODEL_NAME, os.path.join(folder_paths.models_dir, ACESTEP_MODEL_NAME))


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
    ):
        """Initialize ACE-Step handlers if not already initialized"""
        if self.handlers_initialized:
            return self.dit_handler, self.llm_handler

        if not ACESTEP_AVAILABLE:
            raise RuntimeError("ACE-Step is not installed. Please install it first.")

        # Auto-detect device if "auto" is specified
        if device == "auto":
            device = self.auto_detect_device()

        try:
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

            # Initialize LLM handler
            self.llm_handler = LLMHandler()
            self.llm_handler.initialize(
                checkpoint_dir=checkpoint_dir,
                lm_model_path=lm_model_path,
                backend="vllm",
                device=device,
            )

            self.handlers_initialized = True
            return self.dit_handler, self.llm_handler

        except Exception as e:
            raise RuntimeError(f"Failed to initialize ACE-Step handlers: {str(e)}")


class ACE_STEP_TEXT_TO_MUSIC(ACE_STEP_BASE):
    """Generate music from text description"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "caption": ("STRING", {"default": "", "multiline": True}),
                "checkpoint_dir": ("STRING", {"default": ""}),
                "config_path": ("STRING", {"default": "acestep-v15-turbo"}),
                "lm_model_path": ("STRING", {"default": "acestep-5Hz-lm-1.7B"}),
                "duration": ("FLOAT", {"default": 30.0, "min": 10.0, "max": 600.0}),
                "batch_size": ("INT", {"default": 2, "min": 1, "max": 8}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFffffffff, "control_after_generate": True}),
                "inference_steps": ("INT", {"default": 8, "min": 1, "max": 64}),
                "device": (["auto", "cuda", "cpu", "mps", "xpu"], {"default": "auto"}),
            },
            "optional": {
                "lyrics": ("STRING", {"default": "", "multiline": True}),
                "bpm": ("INT", {"default": 0, "min": 0, "max": 300}),
                "keyscale": ("STRING", {"default": ""}),
                "timesignature": ("STRING", {"default": ""}),
                "vocal_language": (["unknown", "auto", "en", "zh", "ja", "ko", "es", "fr", "de", "ru", "pt", "it", "bn"], {"default": "unknown"}),
                "instrumental": ("BOOLEAN", {"default": False}),
                "guidance_scale": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 15.0}),
                "shift": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 5.0}),
                "thinking": ("BOOLEAN", {"default": True}),
                "lm_temperature": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 2.0}),
                "audio_format": (["flac", "mp3", "wav"], {"default": "flac"}),
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
        lm_temperature: float = 0.85,
        audio_format: str = "flac",
    ) -> Tuple[Dict[str, Any], str, str]:
        # Initialize handlers
        dit_handler, llm_handler = self.initialize_handlers(
            checkpoint_dir=checkpoint_dir,
            config_path=config_path,
            lm_model_path=lm_model_path,
            device=device,
        )

        # Prepare generation parameters
        params = GenerationParams(
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

        # Convert tensor to numpy (channels, samples) -> (samples, channels)
        audio_np = audio_tensor.cpu().numpy().T

        # Prepare ComfyUI audio format
        audio_output = {
            "waveform": torch.from_numpy(audio_np).unsqueeze(0),  # Add batch dimension
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
                "src_audio": ("AUDIO",),
                "caption": ("STRING", {"default": "", "multiline": True}),
                "checkpoint_dir": ("STRING", {"default": ""}),
                "config_path": ("STRING", {"default": "acestep-v15-turbo"}),
                "lm_model_path": ("STRING", {"default": "acestep-5Hz-lm-1.7B"}),
                "audio_cover_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFffffffff, "control_after_generate": True}),
                "inference_steps": ("INT", {"default": 8, "min": 1, "max": 64}),
                "device": (["auto", "cuda", "cpu", "mps", "xpu"], {"default": "auto"}),
            },
            "optional": {
                "lyrics": ("STRING", {"default": "", "multiline": True}),
                "thinking": ("BOOLEAN", {"default": True}),
                "audio_format": (["flac", "mp3", "wav"], {"default": "flac"}),
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
    ) -> Tuple[Dict[str, Any], str, str]:
        import tempfile

        # Save input audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = tmp.name
            waveform = src_audio["waveform"].squeeze(0).numpy()
            import soundfile as sf
            sf.write(temp_path, waveform, src_audio["sample_rate"])

        try:
            # Initialize handlers
            dit_handler, llm_handler = self.initialize_handlers(
                checkpoint_dir=checkpoint_dir,
                config_path=config_path,
                lm_model_path=lm_model_path,
                device=device,
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

            # Convert tensor to numpy
            audio_np = audio_tensor.cpu().numpy().T

            # Prepare ComfyUI audio format
            audio_output = {
                "waveform": torch.from_numpy(audio_np).unsqueeze(0),
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
                "src_audio": ("AUDIO",),
                "caption": ("STRING", {"default": "", "multiline": True}),
                "repainting_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 600.0}),
                "repainting_end": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 600.0}),
                "checkpoint_dir": ("STRING", {"default": ""}),
                "config_path": ("STRING", {"default": "acestep-v15-turbo"}),
                "lm_model_path": ("STRING", {"default": "acestep-5Hz-lm-1.7B"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFffffffff, "control_after_generate": True}),
                "inference_steps": ("INT", {"default": 8, "min": 1, "max": 64}),
                "device": (["auto", "cuda", "cpu", "mps", "xpu"], {"default": "auto"}),
            },
            "optional": {
                "thinking": ("BOOLEAN", {"default": True}),
                "audio_format": (["flac", "mp3", "wav"], {"default": "flac"}),
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
    ) -> Tuple[Dict[str, Any], str, str]:
        import tempfile

        # Save input audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = tmp.name
            waveform = src_audio["waveform"].squeeze(0).numpy()
            import soundfile as sf
            sf.write(temp_path, waveform, src_audio["sample_rate"])

        try:
            # Initialize handlers
            dit_handler, llm_handler = self.initialize_handlers(
                checkpoint_dir=checkpoint_dir,
                config_path=config_path,
                lm_model_path=lm_model_path,
                device=device,
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

            # Convert tensor to numpy
            audio_np = audio_tensor.cpu().numpy().T

            # Prepare ComfyUI audio format
            audio_output = {
                "waveform": torch.from_numpy(audio_np).unsqueeze(0),
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
                "query": ("STRING", {"default": "", "multiline": True}),
                "checkpoint_dir": ("STRING", {"default": ""}),
                "lm_model_path": ("STRING", {"default": "acestep-5Hz-lm-1.7B"}),
                "config_path": ("STRING", {"default": "acestep-v15-turbo"}),
                "batch_size": ("INT", {"default": 2, "min": 1, "max": 8}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFffffffff, "control_after_generate": True}),
                "inference_steps": ("INT", {"default": 8, "min": 1, "max": 64}),
                "device": (["auto", "cuda", "cpu", "mps", "xpu"], {"default": "auto"}),
            },
            "optional": {
                "instrumental": ("BOOLEAN", {"default": False}),
                "vocal_language": (["auto", "en", "zh", "ja", "ko", "es", "fr", "de", "ru", "pt", "it", "bn"], {"default": "auto"}),
                "thinking": ("BOOLEAN", {"default": True}),
                "audio_format": (["flac", "mp3", "wav"], {"default": "flac"}),
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
    ) -> Tuple[Dict[str, Any], str, str, str]:
        # Initialize handlers
        dit_handler, llm_handler = self.initialize_handlers(
            checkpoint_dir=checkpoint_dir,
            config_path=config_path,
            lm_model_path=lm_model_path,
            device=device,
        )

        # Step 1: Create sample from description
        sample_result = create_sample(
            llm_handler=llm_handler,
            query=query,
            instrumental=instrumental,
            vocal_language=None if vocal_language == "auto" else vocal_language,
            temperature=0.85,
        )

        if not sample_result.success:
            raise RuntimeError(f"Sample creation failed: {sample_result.error}")

        # Step 2: Generate music using the sample
        params = GenerationParams(
            task_type="text2music",
            caption=sample_result.caption,
            lyrics=sample_result.lyrics,
            bpm=sample_result.bpm,
            duration=sample_result.duration,
            keyscale=sample_result.keyscale,
            vocal_language=sample_result.language,
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

        # Convert tensor to numpy
        audio_np = audio_tensor.cpu().numpy().T

        # Prepare ComfyUI audio format
        audio_output = {
            "waveform": torch.from_numpy(audio_np).unsqueeze(0),
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
                "caption": ("STRING", {"default": "", "multiline": True}),
                "lyrics": ("STRING", {"default": "", "multiline": True}),
                "checkpoint_dir": ("STRING", {"default": ""}),
                "lm_model_path": ("STRING", {"default": "acestep-5Hz-lm-1.7B"}),
                "device": (["auto", "cuda", "cpu", "mps", "xpu"], {"default": "auto"}),
            },
            "optional": {
                "user_metadata": ("STRING", {"default": "{}"}),
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


class ACE_STEP_UNDERSTAND(ACE_STEP_BASE):
    """Understand and analyze audio"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_codes": ("STRING", {"default": "", "multiline": True}),
                "checkpoint_dir": ("STRING", {"default": ""}),
                "lm_model_path": ("STRING", {"default": "acestep-5Hz-lm-1.7B"}),
                "device": (["auto", "cuda", "cpu", "mps", "xpu"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("caption", "lyrics", "bpm", "duration", "keyscale", "language")
    FUNCTION = "understand"
    CATEGORY = "Audio/ACE-Step"

    def understand(
        self,
        audio_codes: str,
        checkpoint_dir: str,
        lm_model_path: str,
        device: str,
    ) -> Tuple[str, str, int, float, str, str]:
        # Initialize handlers
        dit_handler, llm_handler = self.initialize_handlers(
            checkpoint_dir=checkpoint_dir,
            config_path="acestep-v15-turbo",  # Dummy
            lm_model_path=lm_model_path,
            device=device,
        )

        # Understand music
        result = understand_music(
            llm_handler=llm_handler,
            audio_codes=audio_codes,
            temperature=0.85,
        )

        if not result.success:
            raise RuntimeError(f"Understanding failed: {result.error}")

        return (
            result.caption,
            result.lyrics,
            result.bpm or 0,
            result.duration or 0.0,
            result.keyscale,
            result.language,
        )


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ACE_STEP_TextToMusic": ACE_STEP_TEXT_TO_MUSIC,
    "ACE_STEP_Cover": ACE_STEP_COVER,
    "ACE_STEP_Repaint": ACE_STEP_REPAINT,
    "ACE_STEP_SimpleMode": ACE_STEP_SIMPLE_MODE,
    "ACE_STEP_FormatSample": ACE_STEP_FORMAT_SAMPLE,
    "ACE_STEP_Understand": ACE_STEP_UNDERSTAND,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ACE_STEP_TextToMusic": "ACE-Step Text to Music",
    "ACE_STEP_Cover": "ACE-Step Cover",
    "ACE_STEP_Repaint": "ACE-Step Repaint",
    "ACE_STEP_SimpleMode": "ACE-Step Simple Mode",
    "ACE_STEP_FormatSample": "ACE-Step Format Sample",
    "ACE_STEP_Understand": "ACE-Step Understand",
}
