# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ComfyUI-Kaola-ACE-Step is a collection of custom nodes for ComfyUI that integrate ACE-Step 1.5, a commercial-grade music generation model. This is a Python package that extends ComfyUI's audio generation capabilities.

## Critical Requirements

### Python Version
- **Python 3.11 is REQUIRED** - ACE-Step is NOT compatible with Python 3.12 or 3.13
- If ComfyUI uses a different Python version, a separate Python 3.11 environment must be set up

### Core Dependencies
- torch>=2.0.0
- soundfile>=0.12.0 (used instead of torchaudio for robustness)
- peft>=0.12.0 (for LoRA support)
- ACE-Step 1.5 (must be installed separately)

## Installation and Setup

### Standard Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/kana112233/ComfyUI-kaola-ace-step.git
cd ComfyUI-kaola-ace-step
pip install -r requirements.txt
```

### ACE-Step Installation
```bash
git clone https://github.com/ACE-Step/ACE-Step-1.5.git
cd ACE-Step-1.5
pip install -e .
```

### Model Download
Models must be placed in `ComfyUI/models/Ace-Step1.5/` with specific subdirectories:
```bash
huggingface-cli download ACE-Step/Ace-Step1.5 --local-dir ComfyUI/models/Ace-Step1.5
```

Required directory structure:
```
ComfyUI/models/Ace-Step1.5/
├── acestep-v15-turbo/      # DiT model (fast)
├── acestep-v15-base/       # DiT model (quality, optional)
├── acestep-5Hz-lm-1.7B/    # Language model
├── vae/                    # VAE model
└── Qwen3-Embedding-0.6B/   # Text encoder
```

## Architecture

### Key Components

1. **`__init__.py`** - Package initialization with critical TLS fix for ComfyUI + Conda compatibility
2. **`nodes.py`** - Main node implementations (all node classes)
3. **`acestep_wrapper.py`** - Compatibility wrapper for ACE-Step initialization
4. **`acestep_repo/`** - Local ACE-Step source code

### Node Classes (in nodes.py)

All nodes inherit from `ACE_STEP_BASE` which provides handler management:

- `ACE_STEP_TEXT_TO_MUSIC` - Generate music from text
- `ACE_STEP_COVER` - Style transfer on audio
- `ACE_STEP_REPAINT` - Local audio editing
- `ACE_STEP_SIMPLE_MODE` - Natural language to music
- `ACE_STEP_FORMAT_SAMPLE` - Format and enhance input
- `ACE_STEP_CREATE_SAMPLE` - Generate music description from query
- `ACE_STEP_UNDERSTAND` - Analyze audio
- `ACE_STEP_LORA_LOADER` - Load PEFT LoRA models

### Critical Monkey Patches

The codebase includes several important monkey patches in `nodes.py`:

1. **TLS/OpenMP Fix** (lines 15-35) - Prevents "Inconsistency detected by ld.so" error in ComfyUI + Conda environments
2. **Audio Processing Patch** (lines 82-125) - Replaces torchaudio with soundfile for better compatibility
3. **Codebook Selection Patch** (lines 236-295) - Fixes multi-codebook flattening issue
4. **LLM Parameter Bridging** (lines 128-233) - Ensures vocal_language and sampling params are properly passed through

### Handler Initialization Pattern

Nodes use `initialize_handlers()` method which:
- Auto-detects device (CUDA > MPS > XPU > CPU)
- Initializes DiT handler via `_initialize_dit_service_direct()`
- Initializes LLM handler with PyTorch backend (not vLLM to avoid conflicts)
- Applies LoRA if provided
- Handles quantization and compilation options

## Common Development Tasks

### Running ComfyUI with these nodes
```bash
cd ComfyUI
python main.py
```

### Testing a workflow
Load example workflows from `examples/` directory in ComfyUI.

### Debugging
Enable verbose logging by checking console output. Nodes print detailed information about:
- Device detection
- Handler initialization
- LoRA loading
- Generation parameters

## Important Patterns

### Device Auto-Detection
```python
device = self.auto_detect_device()  # Returns: "cuda" | "mps" | "xpu" | "cpu"
```

### Audio Format Convention
- Input: ComfyUI AUDIO format `{"waveform": [batch, channels, samples], "sample_rate": int}`
- Output: Same format, with `audio_path` and `metadata` strings

### Temporary File Handling
Nodes that process audio save input to temp files and clean up in `finally` blocks.

### Parameter Validation
All nodes use `create_generation_params()` helper which filters kwargs to match upstream's `GenerationParams` signature, preventing errors when upstream API changes.

## Known Issues and Solutions

### "Inconsistency detected by ld.so"
- Caused by OpenMP library conflicts
- Fixed by `_force_load_libgomp()` in `__init__.py`

### LoRA + Quantization Incompatibility
- PEFT is incompatible with torchao quantization
- Code automatically disables quantization when LoRA is loaded (see `initialize_handlers()`)

### vLLM Backend Conflicts
- Using "pt" (PyTorch) backend for LLM to avoid process group conflicts with ComfyUI

## Node Parameters Reference

### Common Parameters
- `checkpoint_dir`: Path to model directory (leave empty for default `ComfyUI/models/Ace-Step1.5/`)
- `config_path`: Model config (`acestep-v15-turbo` for fast, `acestep-v15-base` for quality)
- `lm_model_path`: Language model path (`acestep-5Hz-lm-1.7B` recommended)
- `device`: Computing platform (`auto` for auto-detection)
- `seed`: Random seed (-1 for random)
- `inference_steps`: Diffusion steps (turbo: 8, base: 32-64)
- `thinking`: Enable LLM Chain-of-Thought reasoning

### Quality/Speed Trade-offs
- **Fastest**: turbo config, 8 steps, thinking=False
- **Balanced**: turbo config, 8-12 steps, thinking=True, shift=3.0
- **Quality**: base config, 32-64 steps, thinking=True, use_adg=True

## LoRA Support

LoRA models should be PEFT format directories containing `adapter_config.json`:
- Place in `ComfyUI/models/loras/` or `ComfyUI/models/Ace-Step1.5/loras/`
- Use `ACE_STEP_LoRALoader` node to load
- Connect `lora_info` output to any generation node's `lora_info` input
- Adjust `strength` parameter (0.0-5.0)

## macOS Specific Notes

- MPS (Metal Performance Shaders) is auto-detected on Apple Silicon
- For Mac M4 Max: Ensure MPS is available for GPU acceleration
- Use `device="auto"` for automatic detection
