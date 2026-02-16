# ACE-Step Base Model Features

These features are **ONLY available with the Base model** (NOT supported with Turbo model):

- **Extract**: Extract specific instrument/vocal tracks from mixed audio
- **Lego**: Add or modify specific instrument tracks based on audio context
- **Complete**: Complete missing tracks from partial audio input

## Model Requirements

| Model | Directory | Required |
|-------|-----------|----------|
| Base Model | `models/Ace-Step1.5/acestep-v15-base/` | Yes |
| LM Model | `models/Ace-Step1.5/acestep-5Hz-lm-1.7B/` | Yes |

## Model Download

### Option 1: Hugging Face CLI

```bash
# Download base model
huggingface-cli download AceStep/ACE-Stepv1.5 --local-dir models/Ace-Step1.5/acestep-v15-base

# Download LM model (if not included)
huggingface-cli download AceStep/ACE-Stepv1.5-LM-1.7B --local-dir models/Ace-Step1.5/acestep-5Hz-lm-1.7B
```

### Option 2: ModelScope

```bash
# Download base model
modelscope download --model AceStep/ACE-Stepv1.5 --local_dir models/Ace-Step1.5/acestep-v15-base
```

### Option 3: Manual Download

Download from:
- https://huggingface.co/AceStep/ACE-Stepv1.5
- https://www.modelscope.cn/models/AceStep/ACE-Stepv1.5

Extract to `ComfyUI/models/Ace-Step1.5/acestep-v15-base/`

## Workflow Usage

1. **Load Example Workflow**: In ComfyUI, drag and drop one of the example JSON files:
   - `examples/extract.json` - Extract vocals from a song
   - `examples/lego.json` - Add drums to existing audio
   - `examples/complete.json` - Complete missing instruments

2. **Basic Workflow Structure**:
   ```
   LoadAudio → ACE_STEP_Extract/Lego/Complete → PreviewAudio
   ```

3. **First Run**: Model loading takes time on first run. Subsequent runs will be faster.

## Nodes Reference

### ACE_STEP_Extract

Extract specific instrument/vocal tracks from mixed audio.

**Inputs:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `src_audio` | AUDIO | required | Source audio to extract from |
| `track_name` | ENUM | vocals | Track to extract |
| `seed` | INT | -1 | Random seed (-1 for random) |
| `inference_steps` | INT | 50 | Diffusion steps (20-100) |
| `guidance_scale` | FLOAT | 7.0 | CFG scale (1.0-15.0) |
| `use_adg` | BOOL | False | Adaptive Dual Guidance |
| `cfg_interval_start` | FLOAT | 0.0 | CFG start ratio (0.0-1.0) |
| `cfg_interval_end` | FLOAT | 1.0 | CFG end ratio (0.0-1.0) |
| `audio_format` | ENUM | flac | Output format (flac/mp3/wav) |
| `checkpoint_dir` | ENUM | acestep-v15-base | Model directory |
| `lm_model_path` | ENUM | acestep-5Hz-lm-1.7B | Language model |
| `device` | ENUM | auto | Compute device |

**Available Tracks:**
`vocals`, `backing_vocals`, `drums`, `bass`, `guitar`, `keyboard`, `percussion`, `strings`, `synth`, `fx`, `brass`, `woodwinds`

**Outputs:**
- `audio`: Extracted audio (AUDIO)
- `audio_path`: Path to saved audio file (STRING)
- `metadata`: Generation metadata JSON (STRING)

---

### ACE_STEP_Lego

Add or modify specific instrument tracks based on existing audio context.

**Inputs:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `src_audio` | AUDIO | required | Source audio as context |
| `track_name` | ENUM | drums | Track to generate |
| `caption` | STRING | "" | Style description (optional) |
| `seed` | INT | -1 | Random seed (-1 for random) |
| `inference_steps` | INT | 50 | Diffusion steps (20-100) |
| `guidance_scale` | FLOAT | 7.0 | CFG scale (1.0-15.0) |
| `repainting_start` | FLOAT | 0.0 | Start time for region (seconds) |
| `repainting_end` | FLOAT | -1.0 | End time for region (-1 = until end) |
| `use_adg` | BOOL | False | Adaptive Dual Guidance |
| `cfg_interval_start` | FLOAT | 0.0 | CFG start ratio |
| `cfg_interval_end` | FLOAT | 1.0 | CFG end ratio |
| `audio_format` | ENUM | flac | Output format |
| `checkpoint_dir` | ENUM | acestep-v15-base | Model directory |
| `lm_model_path` | ENUM | acestep-5Hz-lm-1.7B | Language model |
| `device` | ENUM | auto | Compute device |

**Outputs:**
- `audio`: Generated audio with new track (AUDIO)
- `audio_path`: Path to saved audio file (STRING)
- `metadata`: Generation metadata JSON (STRING)

---

### ACE_STEP_Complete

Complete missing tracks from partial audio input.

**Inputs:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `src_audio` | AUDIO | required | Source audio to complete |
| **Track Switches** | | | |
| `add_drums` | BOOL | True | Add drums track |
| `add_bass` | BOOL | True | Add bass track |
| `add_guitar` | BOOL | False | Add guitar track |
| `add_keyboard` | BOOL | False | Add keyboard/piano track |
| `add_strings` | BOOL | True | Add strings track |
| `add_synths` | BOOL | False | Add synthesizer track |
| `add_percussion` | BOOL | False | Add percussion track |
| `add_brass` | BOOL | False | Add brass track |
| `add_woodwinds` | BOOL | False | Add woodwinds track |
| `add_backing_vocals` | BOOL | False | Add backing vocals track |
| `add_fx` | BOOL | False | Add FX/sound effects track |
| `add_vocals` | BOOL | False | Add vocals track |
| **Vocal Settings** | | | |
| `vocal_language` | ENUM | unknown | Language for vocals |
| `lyrics` | STRING | "" | Lyrics text (multiline) |
| **Quality Settings** | | | |
| `caption` | STRING | "" | Style description |
| `seed` | INT | -1 | Random seed |
| `inference_steps` | INT | 50 | Diffusion steps |
| `guidance_scale` | FLOAT | 7.0 | CFG scale |
| `use_adg` | BOOL | False | Adaptive Dual Guidance |
| `cfg_interval_start` | FLOAT | 0.0 | CFG start ratio |
| `cfg_interval_end` | FLOAT | 1.0 | CFG end ratio |
| `audio_format` | ENUM | flac | Output format |
| `checkpoint_dir` | ENUM | acestep-v15-base | Model directory |
| `lm_model_path` | ENUM | acestep-5Hz-lm-1.7B | Language model |
| `device` | ENUM | auto | Compute device |

**Available Vocal Languages:**
`unknown`, `zh` (Chinese), `en` (English), `ja` (Japanese), `yue` (Cantonese), `ko` (Korean), `fr` (French), `de` (German), `es` (Spanish), `it` (Italian), `ru` (Russian), `pt` (Portuguese)

**Outputs:**
- `audio`: Completed audio with all tracks (AUDIO)
- `audio_path`: Path to saved audio file (STRING)
- `metadata`: Generation metadata JSON (STRING)

## Tips

1. **Quality**: Higher `inference_steps` (70-100) gives better quality but slower generation.
2. **Guidance**: `guidance_scale` 7.0 is a good default. Increase for more prompt adherence.
3. **ADG**: Enable `use_adg` for potentially better separation quality.
4. **Seed**: Use fixed seed for reproducible results.
5. **GPU Memory**: Base model requires significant VRAM. Use `device=cpu` if GPU memory is insufficient.
