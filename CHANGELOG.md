
# Changelog

All notable changes to the ACE-Step ComfyUI Nodes project will be documented in this file.

## [v0.6.0] - 2026-02-16

### Added
- **ACE_STEP_ClearVRAM Node**: Clear GPU VRAM and ACE-Step models.
  - Clears ACE-Step DiT and LLM handlers.
  - Uses ComfyUI's memory management API.
  - Multiple rounds of GC and cache cleanup.
  - Shows detailed memory usage logs.
- **Example Workflow**: Added `examples/extract-clean-VARM.json` demonstrating ClearVRAM usage after Extract.

## [v0.5.0] - 2026-02-16

### Added
- **ACE_STEP_Extract Node**: Extract instrument/vocal tracks from mixed audio. See `docs/base_features.md`.
- **ACE_STEP_Lego Node**: Add or modify instrument tracks in existing audio. See `docs/base_features.md`.
- **ACE_STEP_Complete Node**: Complete missing tracks from partial audio. See `docs/base_features.md`.
- **Documentation**: Added `docs/base_features.md` with detailed usage instructions.

## [v0.4.0] - 2026-02-15

### Added
- **ACE_STEP_Captioner Node**:
  - Professional-grade music captioning model based on Qwen2.5-Omni-7B.
  - Generates detailed, structured descriptions of audio content.
  - **Features**:
    - Musical Style Analysis (genres, sub-genres, stylistic influences)
    - Instrument Recognition (1000+ instrument types)
    - Structure & Progression Analysis (intro, verse, chorus, bridge, etc.)
    - Timbre Description (tonal qualities, textures, sonic characteristics)
  - **Performance**: Accuracy surpasses Gemini Pro 2.5 in music description tasks.
  - Inputs: `audio`, `model_id`, `device`, `dtype`, `custom_prompt`, generation parameters.
  - Outputs: `caption` (concise), `style_tags` (comma-separated), `full_description` (detailed).
- **Example Workflow**: Added `examples/captioner.json` to demonstrate the usage of the captioner node.

## [v0.3.0] - 2026-02-14

### Added
- **ACE_STEP_Transcriber Node**:
  - Implemented a new node for Automatic Speech Recognition (ASR) using the Qwen2.5-Omni model.
  - Supports loading the model from local path or HuggingFace/ModelScope.
  - **Quality Improvements**:
    - **Language Selection**: Added support for multiple languages (En, Zh, Ja, Ko, Fr, De, Es, It, Ru, Pt).
    - **Generation Control**: Added `temperature` and `repetition_penalty`.
    - **Prompting**: Implemented Chat Template.
    - **Precision**: Added `dtype` selection.
  - Inputs: `audio`, `model_id`, `device`, `chunk_length_s`, `return_timestamps`.
  - Outputs: `transcription` (STRING).
- **Example Workflow**: Added `examples/transcriber.json` to demonstrate the usage of the transcriber node.
