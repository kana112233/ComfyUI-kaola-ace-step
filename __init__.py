"""
ACE-Step 1.5 Music Generation Nodes for ComfyUI
"""

from .nodes import (
    ACE_STEP_TEXT_TO_MUSIC,
    ACE_STEP_COVER,
    ACE_STEP_REPAINT,
    ACE_STEP_SIMPLE_MODE,
    ACE_STEP_FORMAT_SAMPLE,
    ACE_STEP_UNDERSTAND,
)

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

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
