# ACE-Step 1.5 ComfyUI Nodes

ComfyUI custom nodes for [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5), a powerful open-source music generation model.

## Features

- **Text to Music**: Generate music from text descriptions
- **Cover**: Transform existing audio while maintaining structure
- **Repaint**: Regenerate specific segments of audio
- **Simple Mode**: Generate music from natural language with automatic metadata
- **Format Sample**: Enhance and format user input
- **Understand**: Analyze audio semantic codes

## Installation

### Prerequisites

1. Install ACE-Step:
```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install ACE-Step
git clone https://github.com/ace-step/ACE-Step-1.5.git
cd ACE-Step-1.5
uv sync
```

2. Download models:
```bash
uv run acestep-download
```

### Install ComfyUI Nodes

1. Clone this repository or copy the `ace_step` folder to your ComfyUI `custom_nodes` directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/comfyui-ace-step-nodes.git ace_step
```

2. Install Python dependencies:
```bash
pip install torch torchvision torchaudio soundfile
```

3. Link or copy ACE-Step to the node directory:
```bash
cd ace_step
ln -s /path/to/ACE-Step-1.5 ./acestep_repo
```

## Usage

### Text to Music

Generate music from text description with optional parameters:

- **caption**: Text description of the music
- **lyrics**: Optional lyrics
- **duration**: Target duration in seconds (10-600)
- **bpm**: Beats per minute (0 = auto)
- **keyscale**: Musical key (empty = auto)
- **instrumental**: Generate instrumental music
- **inference_steps**: Number of denoising steps (8 for turbo, 32-64 for base)
- **seed**: Random seed (-1 for random)

### Cover

Transform existing audio:

- **src_audio**: Input audio
- **caption**: Desired style transformation
- **audio_cover_strength**: How much to follow original (0.0-1.0)

### Repaint

Regenerate specific segments:

- **src_audio**: Input audio
- **caption**: Description for repainted section
- **repainting_start**: Start time in seconds
- **repainting_end**: End time in seconds (-1 for end)

### Simple Mode

Generate from natural language:

- **query**: Natural language description
- **vocal_language**: Optional language constraint
- **instrumental**: Generate instrumental

### Format Sample

Format and enhance input:

- **caption**: Raw caption
- **lyrics**: Raw lyrics
- **user_metadata**: Optional JSON with constraints (e.g., `{"bpm": 120}`)

### Understand

Analyze audio codes:

- **audio_codes**: 5Hz audio semantic codes
- Returns: caption, lyrics, BPM, duration, key, language

## Configuration

### Checkpoint Directory

Set the path to your ACE-Step checkpoints (default: `./checkpoints`).

### Model Selection

- **config_path**: DiT model to use
  - `acestep-v15-turbo` (default, fast)
  - `acestep-v15-base` (higher quality)
  - `acestep-v15-sft` (SFT model)

- **lm_model_path**: LM model to use
  - `acestep-5Hz-lm-0.6B` (lightweight, 6-12GB VRAM)
  - `acestep-5Hz-lm-1.7B` (balanced, 12-16GB VRAM, default)
  - `acestep-5Hz-lm-4B` (best quality, 16GB+ VRAM)

### Device

- **cuda**: NVIDIA GPU (recommended)
- **cpu**: CPU (slower)
- **mps**: Apple Silicon GPU

## Tips

1. **Quality vs Speed**: Use turbo model with 8 steps for speed, base model with 32-64 steps for quality
2. **Memory**: Reduce batch size if you encounter OOM errors
3. **Metadata**: Use Simple Mode for automatic metadata, or Text to Music for manual control
4. **Seeds**: Set fixed seeds for reproducible results
5. **LM**: Disable thinking (`thinking=False`) for faster generation if you have precise parameters

## Examples

### Electronic Dance Track
```
caption: "upbeat electronic dance music with heavy bass and synthesizer leads"
bpm: 128
duration: 45
instrumental: true
```

### Pop Ballad
```
caption: "emotional pop ballad with piano and strings"
lyrics: [Verse 1]...[Chorus]...
vocal_language: en
bpm: 72
duration: 180
```

### Cover Generation
```
caption: "jazz piano arrangement with swing feel"
audio_cover_strength: 0.7
```

## Troubleshooting

**"ACE-Step is not installed"**: Install ACE-Step and link it to `acestep_repo/`

**"Out of memory"**: Reduce batch_size or use a smaller LM model

**"Failed to initialize handlers"**: Check checkpoint_dir path and model availability

**Poor quality**: Increase inference_steps, use base model, or adjust guidance_scale

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

Based on [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) by ACE Studio and StepFun.

## Links

- [ACE-Step GitHub](https://github.com/ace-step/ACE-Step-1.5)
- [ACE-Step Hugging Face](https://huggingface.co/ACE-Step)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
