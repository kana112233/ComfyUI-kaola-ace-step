"""
ACE-Step Common Module

Shared configuration, utilities, and base classes for ACE-Step nodes.
"""

import os
import folder_paths
from typing import Dict, List, Any, Optional, Tuple, NamedTuple


# =============================================================================
# Model Types (for node connections)
# =============================================================================

class ACEStepModel(NamedTuple):
    """ACE-Step model container for passing between nodes"""
    dit_handler: Any  # AceStepHandler instance
    llm_handler: Any  # LLMHandler instance (can be None)
    checkpoint_dir: str
    config_path: str
    lm_model_path: str
    device: str


class ACEStepLM(NamedTuple):
    """ACE-Step Language Model container for passing between nodes"""
    llm_handler: Any  # LLMHandler instance
    checkpoint_dir: str
    lm_model_path: str
    device: str


# =============================================================================
# Model Configuration
# =============================================================================

ACESTEP_MODEL_NAME = "Ace-Step1.5"

# Register model folder with ComfyUI (for standard format)
def _register_model_paths():
    """Register ACE-Step model directory with ComfyUI"""
    models_dir = folder_paths.models_dir
    model_dir = os.path.join(models_dir, ACESTEP_MODEL_NAME)
    folder_paths.add_model_folder_path(ACESTEP_MODEL_NAME, model_dir)

_register_model_paths()


def get_acestep_checkpoints() -> List[str]:
    """Get available ACE-Step checkpoint directories

    Returns both relative path (for backward compatibility) and full path.
    """
    paths = folder_paths.get_folder_paths(ACESTEP_MODEL_NAME)
    result = []

    # Add relative path first for backward compatibility with old workflows
    result.append(ACESTEP_MODEL_NAME)

    for p in paths:
        if os.path.exists(p):
            result.append(p)

    return result if result else [ACESTEP_MODEL_NAME]


def get_acestep_models() -> List[str]:
    """Get available ACE-Step model names

    Returns list with "None" as first option to allow skipping LM loading.
    """
    paths = folder_paths.get_folder_paths(ACESTEP_MODEL_NAME)
    models = ["None"]  # Add None as first option for skipping LM

    for p in paths:
        if os.path.exists(p):
            # Scan for DiT models (acestep-v15-*)
            for item in os.listdir(p):
                item_path = os.path.join(p, item)
                if os.path.isdir(item_path) and item.startswith("acestep-v15-"):
                    models.append(item)
                # Also include LM models
                if os.path.isdir(item_path) and item.startswith("acestep-5Hz-lm-"):
                    models.append(item)

    # Default fallback models
    defaults = ["acestep-5Hz-lm-1.7B", "acestep-v15-turbo"]
    for d in defaults:
        if d not in models:
            models.append(d)

    return models


def get_available_peft_loras() -> List[str]:
    """Get available PEFT LoRA models"""
    lora_paths = []
    search_paths = [
        os.path.join(folder_paths.models_dir, "loras"),
        os.path.join(folder_paths.models_dir, ACESTEP_MODEL_NAME, "loras")
    ]

    base_dir = os.path.abspath(folder_paths.models_dir)

    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue
        for root, dirs, files in os.walk(search_path):
            if "adapter_config.json" in files:
                abs_root = os.path.abspath(root)
                try:
                    rel_path = os.path.relpath(abs_root, base_dir)
                    rel_path = rel_path.replace("\\", "/")
                    lora_paths.append(rel_path)
                except ValueError:
                    lora_paths.append(abs_root)

    if not lora_paths:
        return ["None"]

    return sorted(list(set(lora_paths)))


# =============================================================================
# Common Parameters
# =============================================================================

DEVICES = ["auto", "cuda", "cpu", "mps", "xpu"]
DOWNLOAD_SOURCES = ["auto", "huggingface", "modelscope"]
AUDIO_FORMATS = ["flac", "mp3", "wav"]
# Vocal language options for music generation
LANGUAGES = ["auto", "en", "zh", "ja", "ko", "fr", "de", "es", "it", "ru", "pt", "nl", "pl", "tr", "vi", "cs", "fa", "id", "uk", "hu", "ar", "sv", "ro", "el", "th", "unknown"]
QUANTIZATION_OPTIONS = ["None", "int8_weight_only"]

# Musical key and scale options for music generation
# Format: "{root} {quality}" where root is the note and quality is major/minor
KEYSCALE_OPTIONS = [
    "auto",  # Let LM auto-detect/generate
    *[f"{root} {quality}"
      for quality in ["major", "minor"]
      for root in ["C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B"]]
]

# Time signature options (represents beats per measure)
TIMESIGNATURE_OPTIONS = ["auto", "2", "3", "4", "6"]  # 2/4, 3/4, 4/4, 6/8
