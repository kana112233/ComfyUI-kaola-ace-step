"""
ACE-Step Captioner Node

Music captioning model for generating detailed descriptions of audio content.
Model: https://huggingface.co/ACE-Step/acestep-captioner

Usage is the same as Qwen2.5 Omni-7B.
Recommended prompt: "*Task* Describe this audio in detail"
"""

import os
import torch
import numpy as np
import librosa
import folder_paths
import comfy.utils
import comfy.model_management
from transformers.generation.streamers import BaseStreamer
from typing import Dict, Any, Tuple, Optional

# Streamer for ComfyUI progress bar and interruption
class ComfyStreamer(BaseStreamer):
    def __init__(self, pbar):
        self.pbar = pbar

    def put(self, value):
        self.pbar.update(1)
        comfy.model_management.throw_exception_if_processing_interrupted()

    def end(self):
        pass


# Constants
ACESTEP_MODEL_NAME = "Ace-Step1.5"
DEFAULT_CAPTIONER_MODEL = "ACE-Step/acestep-captioner"


def get_acestep_captioner_models():
    """Get available captioner models from local paths and online."""
    models = []

    # 1. Check for 'acestep-captioner' folder in models/
    captioner_dir = os.path.join(folder_paths.models_dir, "acestep-captioner")
    if os.path.exists(captioner_dir):
        # If it contains config.json, it's the model
        if os.path.exists(os.path.join(captioner_dir, "config.json")):
            models.append("acestep-captioner")
        else:
            # List subdirectories (e.g. models/acestep-captioner/v1)
            for name in os.listdir(captioner_dir):
                if os.path.isdir(os.path.join(captioner_dir, name)):
                    models.append(os.path.join("acestep-captioner", name))

    # 2. Check inside Ace-Step1.5 folder
    acestep_dir = os.path.join(folder_paths.models_dir, ACESTEP_MODEL_NAME)
    if os.path.exists(acestep_dir):
        candidates = ["captioner", "acestep-captioner"]
        for c in candidates:
            if os.path.exists(os.path.join(acestep_dir, c)):
                models.append(os.path.join(ACESTEP_MODEL_NAME, c))

    # 3. Always include the default online model ID
    if DEFAULT_CAPTIONER_MODEL not in models:
        models.append(DEFAULT_CAPTIONER_MODEL)

    return sorted(list(set(models)))


class ACE_STEP_CAPTIONER:
    """
    ACE-Step Captioner Node

    A professional-grade music captioning model that generates detailed,
    structured descriptions of audio content.

    Features:
    - Musical Style Analysis (genres, sub-genres, stylistic influences)
    - Instrument Recognition (1000+ instrument types)
    - Structure & Progression Analysis (intro, verse, chorus, bridge, etc.)
    - Timbre Description (tonal qualities, textures, sonic characteristics)

    Performance: Accuracy surpasses Gemini Pro 2.5 in music description tasks.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Input audio to caption/describe."}),
                "model_id": (get_acestep_captioner_models(), {
                    "default": DEFAULT_CAPTIONER_MODEL,
                    "tooltip": "Select the captioner model. Can be a local path or HuggingFace ID."
                }),
                "device": (["auto", "cuda", "cpu", "mps"], {
                    "default": "auto",
                    "tooltip": "Inference device. Use 'auto' or 'mps' for Mac."
                }),
                "dtype": (["auto", "float16", "float32"], {
                    "default": "auto",
                    "tooltip": "Model precision. 'auto' uses float16 for CUDA and float32 for CPU/MPS."
                }),
            },
            "optional": {
                "custom_prompt": ("STRING", {
                    "default": "*Task* Describe this audio in detail",
                    "multiline": True,
                    "tooltip": "Custom prompt for captioning. Default is the recommended prompt from ACE-Step."
                }),
                "max_new_tokens": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 4096,
                    "tooltip": "Maximum number of tokens to generate. Increase for longer descriptions."
                }),
                "temperature": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Sampling temperature. Lower values (0.1-0.3) are more deterministic and accurate."
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Nucleus sampling: cumulative probability threshold."
                }),
                "top_k": ("INT", {
                    "default": 50,
                    "min": 0,
                    "max": 1000,
                    "tooltip": "Top-K sampling. 0 = disabled. Lower values can improve accuracy."
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.1,
                    "min": 1.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Penalty for repeating tokens. Increase if output gets stuck in loops."
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffff,
                    "tooltip": "Random seed for reproducible results. 0 for random."
                }),
                "chunk_length_s": ("FLOAT", {
                    "default": 30.0,
                    "min": 0.0,
                    "max": 300.0,
                    "step": 1.0,
                    "tooltip": "Audio chunk length in seconds for processing long audio."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("caption", "style_tags", "full_description")
    FUNCTION = "caption"
    CATEGORY = "Audio/ACE-Step"
    OUTPUT_NODE = True

    def caption(
        self,
        audio: Dict[str, Any],
        model_id: str,
        device: str,
        dtype: str,
        custom_prompt: str = "*Task* Describe this audio in detail",
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        seed: int = 0,
        chunk_length_s: float = 30.0,
    ) -> Tuple[str, str, str]:
        """
        Generate a detailed caption/description of the input audio.

        Returns:
            caption: A concise summary caption
            style_tags: Comma-separated style/instrument tags
            full_description: The complete detailed description
        """
        print(f"ACE_STEP_CAPTIONER: Captioning with model {model_id} on {device} ({dtype})")

        # Set random seed for reproducibility
        if seed != 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            print(f"ACE_STEP_CAPTIONER: Using seed {seed}")

        # 1. Device Setup
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        print(f"ACE_STEP_CAPTIONER: Using device: {device}")

        # 2. Path Resolution
        load_path = model_id
        if not os.path.exists(load_path) and not os.path.isabs(load_path):
            potential_paths = [
                os.path.join(folder_paths.models_dir, load_path),
                os.path.join(folder_paths.models_dir, ACESTEP_MODEL_NAME, load_path),
            ]
            for p in potential_paths:
                if os.path.exists(p):
                    load_path = p
                    break

        print(f"ACE_STEP_CAPTIONER: Resolved model path: {load_path}")

        # 2.5 Memory Management
        if device == "cuda":
            torch.cuda.empty_cache()

        # 3. Model Loading
        from transformers import pipeline, AutoProcessor

        try:
            print(f"ACE_STEP_CAPTIONER: Loading pipeline from {load_path}...")

            # Determine dtype
            torch_dtype = torch.float32  # default
            if dtype == "auto":
                torch_dtype = torch.float16 if device == "cuda" else torch.float32
            elif dtype == "float16":
                torch_dtype = torch.float16
            elif dtype == "float32":
                torch_dtype = torch.float32

            # Load pipeline
            # trust_remote_code=True is CRITICAL for Qwen2.5-Omni based models
            caption_pipeline = pipeline(
                "automatic-speech-recognition",
                model=load_path,
                device=device,
                trust_remote_code=True,
                torch_dtype=torch_dtype
            )

            model = caption_pipeline.model

            # CRITICAL: Disable audio generation to prevent OOM
            # Qwen2.5-Omni tries to generate audio (DiT) by default during generation
            if hasattr(model.config, "disable_audio_generation"):
                model.config.disable_audio_generation = True

            if hasattr(model, "generation_config"):
                if hasattr(model.generation_config, "disable_audio_generation"):
                    model.generation_config.disable_audio_generation = True

            # Monkeypatch token2wav to prevent OOM if flags are ignored
            if hasattr(model, 'token2wav'):
                print("ACE_STEP_CAPTIONER: Monkeypatching model.token2wav to prevent audio generation and OOM.")

                class DummyModule(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.register_buffer("dummy_param", torch.tensor(0.0))

                    def forward(self, *args, **kwargs):
                        return torch.tensor([], device=self.dummy_param.device)

                    @property
                    def dtype(self):
                        return self.dummy_param.dtype

                model.token2wav = DummyModule().to(device)

            # Load processor separately
            processor = AutoProcessor.from_pretrained(load_path, trust_remote_code=True)

            print("ACE_STEP_CAPTIONER: Model and Processor loaded successfully.")

        except Exception as e:
            raise RuntimeError(f"Failed to load Captioner model {model_id}: {e}")

        # 4. Audio Preparation
        waveform = audio['waveform']
        sample_rate = audio['sample_rate']

        # Convert to numpy and mono
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()

        if waveform.ndim == 3:
            waveform = waveform[0]  # [channels, samples]

        # Mix to mono if stereo
        if waveform.ndim > 1 and waveform.shape[0] > 1:
            waveform = np.mean(waveform, axis=0)
        elif waveform.ndim > 1:
            waveform = waveform[0]

        waveform = waveform.astype(np.float32)

        # Resample to 16kHz if needed
        TARGET_SR = 16000
        if sample_rate != TARGET_SR:
            print(f"ACE_STEP_CAPTIONER: Resampling audio from {sample_rate}Hz to {TARGET_SR}Hz")
            try:
                waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=TARGET_SR)
                sample_rate = TARGET_SR
            except Exception as e:
                print(f"ACE_STEP_CAPTIONER: Resampling failed: {e}")
                raise e

        # Calculate audio duration
        audio_duration = len(waveform) / sample_rate
        print(f"ACE_STEP_CAPTIONER: Audio duration: {audio_duration:.1f}s")

        # Determine chunking strategy
        OVERLAP_S = 5
        chunk_samples = int(chunk_length_s * sample_rate)
        overlap_samples = int(OVERLAP_S * sample_rate)
        needs_chunking = audio_duration > chunk_length_s

        if needs_chunking:
            step_samples = chunk_samples - overlap_samples
            num_chunks = int(np.ceil((len(waveform) - overlap_samples) / step_samples))
            print(f"ACE_STEP_CAPTIONER: Splitting into {num_chunks} chunks (chunk={chunk_length_s}s, overlap={OVERLAP_S}s)")
        else:
            num_chunks = 1
            print(f"ACE_STEP_CAPTIONER: Processing as single chunk")

        # 5. Inference
        try:
            # Build prompt
            instruction = custom_prompt.strip() if custom_prompt.strip() else "*Task* Describe this audio in detail"
            print(f"ACE_STEP_CAPTIONER: Instruction: {instruction}")

            # Build text prompt with chat template
            if hasattr(processor, "apply_chat_template"):
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio": "placeholder"},
                            {"type": "text", "text": instruction}
                        ]
                    }
                ]
                text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            else:
                text_prompt = f"<|im_start|>system\nYou are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

            print(f"ACE_STEP_CAPTIONER: Generation params: temp={temperature}, top_p={top_p}, top_k={top_k}")

            # Helper function to caption a single chunk
            def caption_chunk(audio_chunk, chunk_idx=None):
                inputs = processor(
                    text=[text_prompt],
                    audio=audio_chunk,
                    sampling_rate=sample_rate,
                    return_tensors="pt"
                )

                inputs = {k: v.to(device) for k, v in inputs.items()}
                if torch_dtype == torch.float16 and "input_features" in inputs:
                    inputs["input_features"] = inputs["input_features"].to(dtype=torch.float16)

                pbar = comfy.utils.ProgressBar(max_new_tokens)
                streamer = ComfyStreamer(pbar)

                with torch.no_grad():
                    generation_output = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        streamer=streamer,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k if top_k > 0 else None,
                        repetition_penalty=repetition_penalty,
                        do_sample=True if temperature > 0 else False,
                        return_audio=False
                    )

                generated_ids = generation_output
                full_output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                # Extract assistant's response
                caption = full_output
                assistant_marker = "assistant"
                if assistant_marker in full_output:
                    parts = full_output.split(assistant_marker)
                    if len(parts) > 1:
                        caption = parts[-1].strip()

                for marker in ["system", "user"]:
                    if f"\n{marker}" in caption:
                        caption = caption.split(f"\n{marker}")[0].strip()

                return caption

            # Process chunks
            all_captions = []

            if needs_chunking:
                step_samples = chunk_samples - overlap_samples
                for i in range(num_chunks):
                    start = i * step_samples
                    end = min(start + chunk_samples, len(waveform))
                    chunk = waveform[start:end]

                    chunk_duration = len(chunk) / sample_rate
                    print(f"ACE_STEP_CAPTIONER: Processing chunk {i+1}/{num_chunks} ({chunk_duration:.1f}s)")

                    chunk_result = caption_chunk(chunk, chunk_idx=i)
                    all_captions.append(chunk_result)
                    print(f"ACE_STEP_CAPTIONER: Chunk {i+1} result: {chunk_result[:80]}...")

                # Merge captions
                # For captioning, we take the most detailed description
                # Usually the first chunk has the most complete analysis
                full_description = all_captions[0]
                if len(all_captions) > 1:
                    # Append additional details from other chunks
                    for i, cap in enumerate(all_captions[1:], 1):
                        # Only add unique content
                        if cap not in full_description:
                            full_description += f"\n\n[Section {i+1}]: {cap}"
            else:
                print("ACE_STEP_CAPTIONER: Running inference...")
                full_description = caption_chunk(waveform)

            # Extract style tags from description
            style_tags = self._extract_style_tags(full_description)

            # Generate concise caption (first sentence or summary)
            caption = self._generate_concise_caption(full_description)

            print(f"ACE_STEP_CAPTIONER: Final caption: {caption[:100]}...")
            return (caption, style_tags, full_description)

        except Exception as e:
            print(f"ACE_STEP_CAPTIONER: Inference failed: {e}")
            raise e

    def _extract_style_tags(self, description: str) -> str:
        """Extract style/instrument tags from the description."""
        tags = []

        # Common music style keywords
        style_keywords = [
            # Genres
            "ambient", "techno", "house", "drum and bass", "synthwave", "downtempo",
            "rock", "alternative", "indie", "post-rock", "progressive", "psychedelic",
            "pop", "synth-pop", "electropop", "dream pop", "art pop",
            "classical", "orchestral", "chamber", "minimalist", "cinematic",
            "jazz", "fusion", "smooth", "bebop", "modal",
            "hip-hop", "trap", "boom bap", "lo-fi", "cloud rap",
            "folk", "indie folk", "acoustic", "singer-songwriter",
            "electronic", "edm", "idm", "breakbeat",
            "r&b", "soul", "funk", "disco",
            "metal", "heavy metal", "death metal", "black metal",
            "punk", "post-punk", "new wave",
            "reggae", "dub", "dancehall",
            "country", "blues", "gospel",
            # Instruments
            "piano", "guitar", "acoustic guitar", "electric guitar", "bass",
            "drums", "percussion", "synthesizer", "synth", "strings", "violin",
            "cello", "saxophone", "trumpet", "flute", "keyboard", "organ",
            "harp", "mandolin", "banjo", "ukulele", "accordion",
            # Vocal styles
            "male vocals", "female vocals", "choir", "harmonies", "backing vocals",
            # Mood/Character
            "melancholic", "upbeat", "energetic", "calm", "peaceful", "dark",
            "bright", "warm", "cold", "ethereal", "atmospheric", "groovy",
            # Tempo
            "slow", "mid-tempo", "fast", "upbeat",
        ]

        desc_lower = description.lower()
        for keyword in style_keywords:
            if keyword in desc_lower and keyword not in [t.lower() for t in tags]:
                tags.append(keyword.title())

        return ", ".join(tags[:10]) if tags else "Music"

    def _generate_concise_caption(self, description: str) -> str:
        """Generate a concise caption from the full description."""
        # Take the first sentence or up to 200 characters
        sentences = description.split('. ')
        if sentences:
            first_sentence = sentences[0].strip()
            if len(first_sentence) > 200:
                return first_sentence[:200] + "..."
            return first_sentence + ("." if not first_sentence.endswith('.') else "")
        return description[:200] + "..." if len(description) > 200 else description
