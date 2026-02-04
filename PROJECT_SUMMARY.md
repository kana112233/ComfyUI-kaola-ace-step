# ACE-Step 1.5 ComfyUI Nodes - Project Summary

## Overview

This project provides ComfyUI custom nodes for [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5), an open-source music generation model that achieves commercial-grade quality while running on consumer hardware.

## What is ACE-Step 1.5?

ACE-Step 1.5 is a hybrid music generation model combining:
- **Language Model (LM)**: Acts as a planner, generating blueprints from user queries
- **Diffusion Transformer (DiT)**: Generates high-quality audio from LM blueprints

**Key Features**:
- Ultra-fast generation (2s on A100, 10s on RTX 3090 for full songs)
- Low VRAM requirement (<4GB for basic use)
- Supports 10 seconds to 10 minutes of audio
- 1000+ instruments and styles
- Multi-language lyrics support (50+ languages)
- Advanced features: cover generation, repaint, track separation, vocal-to-BGM

## Node Package Structure

```
ace_step/
├── __init__.py              # Package initialization, node mappings
├── nodes.py                 # All node implementations
├── README.md                # User-facing documentation
├── USAGE.md                 # Detailed usage guide with examples
├── requirements.txt         # Python dependencies
├── pyproject.toml          # Project metadata and build config
├── install.sh              # Automated installation script
├── Makefile                # Development commands
├── config.example.json     # Configuration example
├── workflow_example.json   # Sample ComfyUI workflow
├── LICENSE                 # MIT License
├── .gitignore             # Git ignore patterns
└── PROJECT_SUMMARY.md     # This file
```

## Included Nodes

### 1. ACE_STEP_TextToMusic
Generate music from text descriptions with full parameter control.

**Inputs**: caption, lyrics, duration, bpm, keyscale, etc.
**Outputs**: audio tensor, file path, metadata

### 2. ACE_STEP_Cover
Transform existing audio to new style while maintaining structure.

**Inputs**: source audio, caption, cover strength
**Outputs**: transformed audio, path, metadata

### 3. ACE_STEP_Repaint
Regenerate specific segments of audio.

**Inputs**: source audio, caption, start/end times
**Outputs**: repainted audio, path, metadata

### 4. ACE_STEP_SimpleMode
Generate from natural language with automatic metadata.

**Inputs**: query, instrumental flag, vocal language
**Outputs**: audio, path, metadata, sample info

### 5. ACE_STEP_FormatSample
Format and enhance user-provided caption and lyrics.

**Inputs**: caption, lyrics, user metadata (JSON)
**Outputs**: formatted caption, lyrics, full metadata

### 6. ACE_STEP_Understand
Analyze audio semantic codes to extract metadata.

**Inputs**: audio codes string
**Outputs**: caption, lyrics, BPM, duration, key, language

## Technical Details

### Architecture

The nodes use a handler-based architecture:
- `ACE_STEP_BASE`: Base class managing handler initialization
- `AceStepHandler`: DiT (Diffusion Transformer) handler
- `LLMHandler`: Language Model handler for reasoning

### Key Dependencies

- **torch**: PyTorch for tensor operations
- **soundfile**: Audio I/O
- **acestep**: ACE-Step library (installed via uv)

### GPU Requirements

- **Minimum**: 6GB VRAM (with 0.6B LM model)
- **Recommended**: 12-16GB VRAM (with 1.7B LM model)
- **Optimal**: 16GB+ VRAM (with 4B LM model)

## Installation

### Quick Install

```bash
cd /path/to/ComfyUI/custom_nodes
git clone <this-repo> ace_step
cd ace_step
chmod +x install.sh
./install.sh
```

### Manual Install

1. Clone ACE-Step repository
2. Install dependencies with `uv sync`
3. Download models with `uv run acestep-download`
4. Install Python requirements
5. Link or copy ACE-Step to `acestep_repo/`

## Usage Examples

### Example 1: Simple Generation
```
ACE-Step Text to Music
├── caption: "relaxing piano music"
├── duration: 60
└── Execute
```

### Example 2: Cover Generation
```
Load Audio → ACE-Step Cover
├── caption: "jazz piano version"
└── audio_cover_strength: 0.8
```

### Example 3: Natural Language
```
ACE-Step Simple Mode
└── query: "energetic K-pop dance track with catchy hooks"
```

## Configuration

### Model Selection

**DiT Models**:
- `acestep-v15-turbo`: Fast, 8 steps (default)
- `acestep-v15-base`: High quality, 32-64 steps
- `acestep-v15-sft`: SFT model

**LM Models**:
- `acestep-5Hz-lm-0.6B`: Lightweight, 6-12GB VRAM
- `acestep-5Hz-lm-1.7B`: Balanced, 12-16GB VRAM (default)
- `acestep-5Hz-lm-4B`: Best quality, 16GB+ VRAM

### Quality Presets

**Fast**: Turbo model, 8 steps, no LM
**Balanced**: Turbo model, 8 steps, LM enabled
**Quality**: Base model, 64 steps, LM enabled, ADG

## Common Use Cases

1. **Text-to-Music**: Generate background music, songs, soundtracks
2. **Style Transfer**: Convert songs to different genres
3. **Audio Editing**: Fix or enhance specific sections
4. **Inspiration**: Get musical ideas from natural language
5. **Stem Separation**: Extract individual instruments (base model)

## Limitations

- Minimum 10 second generation duration
- Maximum 600 seconds (10 minutes)
- Repaint/Lego/Extract/Complete features only work with base model
- CPU mode is significantly slower

## Troubleshooting

**Problem**: "ACE-Step is not installed"
**Solution**: Run `./install.sh` to install ACE-Step

**Problem**: Out of memory errors
**Solution**: Reduce batch_size or use smaller LM model

**Problem**: Poor quality results
**Solution**: Increase inference_steps, use base model

**Problem**: Results don't match prompt
**Solution**: Be more specific in caption, increase guidance_scale

## Performance

**Generation Times** (RTX 3090):
- Turbo model (8 steps): ~10s for full song
- Base model (32 steps): ~40s for full song
- Base model (64 steps): ~80s for full song

**Memory Usage**:
- DiT only: ~2GB VRAM
- + 0.6B LM: ~4GB VRAM
- + 1.7B LM: ~8GB VRAM
- + 4B LM: ~12GB VRAM

## Future Enhancements

Potential improvements:
- [ ] Add support for Lego/Extract/Complete tasks (base model)
- [ ] Add LoRA training node
- [ ] Add quality scoring node
- [ ] Add LRC (lyric timestamp) generation
- [ ] Add batch processing optimizations
- [ ] Add progress callbacks for long generations

## Contributing

Contributions welcome! Areas for improvement:
- Additional node types
- Performance optimizations
- Better error handling
- Documentation improvements
- Example workflows

## References

- [ACE-Step GitHub](https://github.com/ace-step/ACE-Step-1.5)
- [ACE-Step Hugging Face](https://huggingface.co/ACE-Step)
- [ACE-Step Technical Report](https://arxiv.org/abs/2602.00744)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## License

MIT License - See LICENSE file for details.

## Acknowledgments

Based on ACE-Step 1.5 by ACE Studio and StepFun.

---

**Created**: 2026-02-04
**Version**: 1.0.0
**ACE-Step Version**: 1.5
