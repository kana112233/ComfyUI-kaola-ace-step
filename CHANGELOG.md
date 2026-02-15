
# Changelog

All notable changes to the ACE-Step ComfyUI Nodes project will be documented in this file.

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
