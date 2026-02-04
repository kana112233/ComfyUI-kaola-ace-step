# ComfyUI-Kaola-ACE-Step

ComfyUI custom nodes for [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) - Commercial-grade music generation.

## Features

- 🎵 **Text to Music** - Generate music from text
- 🎭 **Cover Generation** - Style transfer
- 🎨 **Audio Repaint** - Local audio editing
- 💡 **Simple Mode** - Natural language to music
- 📝 **Format Sample** - Enhance user input
- 🔍 **Understand Audio** - Analyze audio codes

## Quick Start

### 1. Install ACE-Step

```bash
git clone https://github.com/ACE-Step/ACE-Step-1.5.git
cd ACE-Step-1.5
uv sync
```

### 2. Install ComfyUI Nodes

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/kana112233/ComfyUI-kaola-ace-step.git
cd ComfyUI-kaola-ace-step
pip install -r requirements.txt
```

### 3. Download Models

```bash
# Using huggingface-cli
huggingface-cli download ACE-Step/Ace-Step1.5 --local-dir ComfyUI/models/acestep

# Or using hfd (faster)
pip install hfd
hfd ACE-Step/Ace-Step1.5 --local-dir ComfyUI/models/acestep
```

### 4. Restart ComfyUI

## Node Settings

| Parameter | Value |
|----------|-------|
| `checkpoint_dir` | Leave empty (uses `ComfyUI/models/acestep/`) |
| `config_path` | `acestep-v15-turbo` (fast) or `acestep-v15-base` (quality) |
| `lm_model_path` | `acestep-5Hz-lm-1.7B` (recommended) |
| `device` | `auto` (auto-detects MPS/CUDA/CPU) |

## Model Directory Structure

```
ComfyUI/models/acestep/
├── acestep-v15-turbo/
├── acestep-5Hz-lm-1.7B/
├── vae/
└── Qwen3-Embedding-0.6B/
```

## Workflow Examples

See [examples/](examples/) directory for ready-to-use workflows.

## Nodes

| Node | Description |
|------|-------------|
| **ACE_STEP_TextToMusic** | Generate music from text |
| **ACE_STEP_Cover** | Style transfer |
| **ACE_STEP_Repaint** | Local audio editing |
| **ACE_STEP_SimpleMode** | Natural language generation |
| **ACE_STEP_FormatSample** | Format and enhance input |
| **ACE_STEP_Understand** | Analyze audio codes |

## Requirements

- **Python**: 3.11 (required for ACE-Step)
- **GPU**: 6GB+ VRAM recommended
- **Disk**: ~8GB for models

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- [ACE-Step GitHub](https://github.com/ace-step/ACE-Step-1.5)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
