# ACE-Step ComfyUI Nodes - Usage Guide

## Quick Start

1. **Install Dependencies**
   ```bash
   chmod +x install.sh
   ./install.sh
   ```

2. **Configure Checkpoint Path**
   - In any ACE-Step node, set `checkpoint_dir` to the full path:
   - Example: `/path/to/comfyui-nodes/ace_step/acestep_repo/checkpoints`

3. **Generate Your First Music**
   - Add "ACE-Step Text to Music" node
   - Enter a caption like: "relaxing piano music"
   - Set `duration` to 30 seconds
   - Execute the workflow

## Node Reference

### 1. Text to Music

**Purpose**: Generate music from text description

**Key Parameters**:
- `caption` (required): Music description
  - Example: "upbeat electronic dance music with heavy bass"
- `lyrics` (optional): Lyrics text
  - Use `[Instrumental]` for instrumental tracks
- `duration`: Length in seconds (10-600)
- `bpm`: Tempo (0 = auto-detect)
- `keyscale`: Musical key (empty = auto)
  - Examples: "C Major", "Am", "F# minor"
- `instrumental`: Force instrumental generation
- `inference_steps`: Quality/speed tradeoff
  - Turbo model: 1-20 (recommended 8)
  - Base model: 1-200 (recommended 32-64)
- `seed`: Random seed (-1 for random)
- `thinking`: Enable LM for better metadata
- `lm_temperature`: Creativity (0.0-2.0)

**Workflow Example**:
```
Text to Music → SaveAudio
              → TextNode (path)
              → TextNode (metadata)
```

### 2. Cover

**Purpose**: Transform existing audio to new style

**Key Parameters**:
- `src_audio`: Input audio from ComfyUI
- `caption`: Target style description
  - Example: "jazz piano arrangement with swing feel"
- `audio_cover_strength`: Fidelity to original
  - 1.0: Strong adherence
  - 0.5: Balanced
  - 0.1: Loose interpretation

**Workflow Example**:
```
LoadAudio → ACE-Step Cover → SaveAudio
                        ↓
                   TextNode (metadata)
```

### 3. Repaint

**Purpose**: Regenerate specific segment of audio

**Key Parameters**:
- `src_audio`: Input audio
- `caption`: Description for repainted section
- `repainting_start`: Start time (seconds)
- `repainting_end`: End time (seconds, -1 for end)

**Use Cases**:
- Fix problematic sections
- Add variations
- Smooth transitions

**Workflow Example**:
```
LoadAudio → ACE-Step Repaint → SaveAudio
                          ↓
                     TextNode (metadata)
```

### 4. Simple Mode

**Purpose**: Generate from natural language (auto-metadata)

**Key Parameters**:
- `query`: Natural language description
  - Example: "a soft Bengali love song for a quiet evening"
- `vocal_language`: Optional language constraint
  - Examples: "en", "zh", "ja", "bn"
- `instrumental`: Generate instrumental

**Outputs**:
- `audio`: Generated audio
- `audio_path`: File path
- `metadata`: Generation metadata
- `sample_info`: LM-generated full sample details

**Workflow Example**:
```
Simple Mode → SaveAudio
           → TextNode (sample_info)
```

### 5. Format Sample

**Purpose**: Format and enhance user input

**Key Parameters**:
- `caption`: Raw caption
- `lyrics`: Raw lyrics
- `user_metadata`: JSON with constraints
  - Example: `{"bpm": 120, "duration": 60}`

**Outputs**:
- `formatted_caption`: Enhanced caption
- `formatted_lyrics`: Formatted lyrics
- `formatted_metadata`: Full metadata JSON

**Workflow Example**:
```
TextNode → Format Sample → TextNode (formatted)
```

### 6. Understand

**Purpose**: Analyze audio semantic codes

**Key Parameters**:
- `audio_codes`: 5Hz audio semantic codes
  - Obtained from previous generation's metadata

**Outputs**:
- `caption`: Detected caption
- `lyrics`: Detected lyrics
- `bpm`: Detected BPM
- `duration`: Detected duration
- `keyscale`: Detected key
- `language`: Detected language

## Common Workflows

### Workflow 1: Simple Text to Music

```
ACE-Step Text to Music
├── caption: "epic cinematic trailer music"
├── duration: 60
├── bpm: 0 (auto)
└── inference_steps: 8
        ↓
    SaveAudio
```

### Workflow 2: Music with Lyrics

```
ACE-Step Text to Music
├── caption: "pop ballad with emotional vocals"
├── lyrics: "[Verse 1]\nWalking down...\n\n[Chorus]\nI'm moving on..."
├── vocal_language: "en"
├── bpm: 72
└── duration: 180
        ↓
    SaveAudio
```

### Workflow 3: Style Transfer (Cover)

```
LoadAudio (original song)
        ↓
ACE-Step Cover
├── caption: "orchestral symphonic arrangement"
└── audio_cover_strength: 0.7
        ↓
    SaveAudio
```

### Workflow 4: Batch Generation

```
ACE-Step Text to Music
├── caption: "ambient meditation music"
├── batch_size: 4
├── seed: -1 (random)
└── thinking: true
        ↓
    [4 audio outputs]
```

### Workflow 5: Natural Language to Music

```
ACE-Step Simple Mode
├── query: "energetic K-pop dance track with catchy hooks"
└── vocal_language: "ko"
        ↓
    SaveAudio + Sample Info
```

### Workflow 6: Multi-Stage Generation

```
TextNode (raw caption) → Format Sample → Enhanced Caption
                                              ↓
                                        Text to Music → SaveAudio
```

## Parameter Tuning Guide

### Quality vs Speed

**Fast (Turbo Model)**:
- `config_path`: `acestep-v15-turbo`
- `inference_steps`: 8
- `thinking`: false (if you have metadata)

**Balanced**:
- `config_path`: `acestep-v15-turbo`
- `inference_steps`: 8-12
- `thinking`: true
- `shift`: 3.0

**High Quality (Base Model)**:
- `config_path`: `acestep-v15-base`
- `inference_steps`: 32-64
- `thinking`: true
- `use_adg`: true
- `guidance_scale`: 7.0-9.0

### Memory Optimization

**Low VRAM (<8GB)**:
- `batch_size`: 1
- `lm_model_path`: `acestep-5Hz-lm-0.6B` (or disable LM)
- `thinking`: false

**Medium VRAM (8-16GB)**:
- `batch_size`: 2-4
- `lm_model_path`: `acestep-5Hz-lm-1.7B`

**High VRAM (16GB+)**:
- `batch_size`: 8
- `lm_model_path`: `acestep-5Hz-lm-4B`

### Style Control

**More Creative**:
- `lm_temperature`: 0.9-1.1
- `seed`: -1 (random)
- `thinking`: true

**More Consistent**:
- `lm_temperature`: 0.7-0.85
- `seed`: fixed value
- `thinking`: false (with manual metadata)

**Prompt Adherence**:
- `guidance_scale`: 7.0-9.0 (base model only)
- `shift`: 3.0 (turbo model)

## Tips and Tricks

1. **Caption Writing**
   - Be specific: "upbeat electronic dance music" > "good music"
   - Include mood, genre, instruments
   - Avoid contradictions

2. **Duration**
   - Instrumental: 30-180s works well
   - With lyrics: Use auto (-1) or 180-300s
   - Long form: Up to 600s

3. **Seeds**
   - Use fixed seeds for reproducibility
   - Use -1 for different results each time
   - Check metadata for actual seed used

4. **Metadata**
   - Use Simple Mode for automatic metadata
   - Use Format Sample to refine your input
   - Disable thinking if you have precise metadata

5. **Batch Generation**
   - Generate 2-8 variations at once
   - Seeds will be auto-generated if not specified
   - Each output includes its seed in metadata

6. **Troubleshooting**
   - OOM: Reduce batch_size or use smaller LM
   - Poor quality: Increase steps, use base model
   - Doesn't match prompt: Be more specific, increase guidance
   - Too slow: Use turbo model, reduce steps

## Advanced Usage

### Custom Timesteps

For expert users, customize the denoising schedule:
```python
# In nodes.py, modify GenerationParams
timesteps=[0.97, 0.76, 0.615, 0.5, 0.395, 0.28, 0.18, 0.085, 0]
```

### Understanding Your Audio

1. Generate music with `thinking: true`
2. Extract `audio_codes` from metadata
3. Use Understand node to analyze

### Looping Workflows

Use ComfyUI's iteration features to:
- Generate multiple variations
- Test different captions
- Batch process with different seeds

## Examples

### Example 1: Electronic Dance Track
```
caption: "upbeat electronic dance music with heavy bass and synthesizer leads"
bpm: 128
duration: 45
instrumental: true
inference_steps: 8
```

### Example 2: Acoustic Folk
```
caption: "gentle acoustic folk music with guitar and flute"
bpm: 90
duration: 120
keyscale: "G Major"
instrumental: true
```

### Example 3: Pop Song
```
caption: "catchy pop song with modern production"
lyrics: "[Verse 1]\n...lyrics...\n\n[Chorus]\n...chorus..."
vocal_language: "en"
bpm: 110
duration: 210
thinking: true
```

### Example 4: Jazz Cover
```
src_audio: <load audio>
caption: "smooth jazz trio arrangement with piano solo"
audio_cover_strength: 0.7
```

### Example 5: Repaint Section
```
src_audio: <load audio>
caption: "epic orchestral hit with rising tension"
repainting_start: 30.0
repainting_end: 40.0
```

## Support

- [ACE-Step GitHub](https://github.com/ace-step/ACE-Step-1.5)
- [ACE-Step Docs](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en)
- [ComfyUI GitHub](https://github.com/comfyanonymous/ComfyUI)

## License

MIT License - See LICENSE for details.
