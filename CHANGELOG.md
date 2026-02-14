
# Changelog

All notable changes to the ACE-Step ComfyUI Nodes project will be documented in this file.

## [2026-02-14]

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
