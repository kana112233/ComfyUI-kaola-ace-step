
import os
import torch
import numpy as np
import librosa
import folder_paths # Ensure this is accessible. It is provided by ComfyUI context.

import comfy.utils
import comfy.model_management
from transformers.generation.streamers import BaseStreamer

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

# We check for ACESTEP availability mainly for consistency.
try:
    import acestep
    ACESTEP_AVAILABLE = True
except ImportError:
    ACESTEP_AVAILABLE = False

def get_acestep_transcriber_models():
    # If acestep packge is missing, we still might want to allow downloading the model if it's just transformers
    # But for consistency with other nodes, we can check availability if we wanted to enforce it.
    
    models = []
    
    # 1. Check for standard 'acestep-transcriber' folder in models/
    transcriber_dir = os.path.join(folder_paths.models_dir, "acestep-transcriber")
    if os.path.exists(transcriber_dir):
        # If it contains config.json, it's the model
        if os.path.exists(os.path.join(transcriber_dir, "config.json")):
            models.append("acestep-transcriber")
        else:
            # List subdirectories (e.g. models/acestep-transcriber/v1)
            for name in os.listdir(transcriber_dir):
                if os.path.isdir(os.path.join(transcriber_dir, name)):
                     models.append(os.path.join("acestep-transcriber", name))

    # 2. Check inside Ace-Step1.5 folder
    acestep_dir = os.path.join(folder_paths.models_dir, ACESTEP_MODEL_NAME)
    if os.path.exists(acestep_dir):
        candidates = ["transcriber", "Qwen2.5-Omni", "acestep-transcriber"]
        for c in candidates:
            if os.path.exists(os.path.join(acestep_dir, c)):
                 models.append(os.path.join(ACESTEP_MODEL_NAME, c))

    # 3. Always include the default online model ID
    default_id = "kana112233/ComfyUI-kaola-ace_step"
    if default_id not in models:
        models.append(default_id)
        
    return sorted(list(set(models)))

class ACE_STEP_TRANSCRIBER:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Input audio to transcribe."}),
                "model_id": (get_acestep_transcriber_models(), {"default": get_acestep_transcriber_models()[0], "tooltip": "Select the Qwen2.5-Omni model. Can be a local path (in models/acestep-transcriber) or a HuggingFace ID."}),
                "device": (["cuda", "cpu", "mps", "auto"], {"default": "auto", "tooltip": "Inference device. Use 'auto' or 'mps' for Mac."}),
                "dtype": (["auto", "float16", "float32"], {"default": "auto", "tooltip": "Model precision. 'auto' uses float16 for CUDA and float32 for CPU/MPS."}),
                "language": (["auto", "en", "zh", "ja", "ko", "fr", "de", "es", "it", "ru", "pt"], {"default": "auto", "tooltip": "Target language for transcription. 'auto' uses default prompt."}),
                "chunk_length_s": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 300.0, "step": 1.0, "tooltip": "Audio chunk length in seconds for processing."}),
                "return_timestamps": (["true", "false", "word"], {"default": "false", "tooltip": "Whether to return timestamps. 'word' for word-level timestamps, 'true' for segment-level."}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Custom prompt to override built-in language prompts. e.g. 'Transcribe the audio to Chinese:'"}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.1, "tooltip": "Sampling temperature. Lower values are more deterministic."}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.1, "tooltip": "Penalty for repeating tokens. Increase if output gets stuck in loops."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed for reproducible results. 0 for random."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("transcription",)
    FUNCTION = "transcribe"
    CATEGORY = "ACE_STEP"

    def transcribe(self, audio, model_id, device, dtype, language, chunk_length_s, return_timestamps, custom_prompt="", temperature=0.2, repetition_penalty=1.1, seed=0):
        print(f"ACE_STEP_TRANSCRIBER: Transcribing with model {model_id} on {device} ({dtype})")

        # Set random seed for reproducibility
        if seed != 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            print(f"ACE_STEP_TRANSCRIBER: Using seed {seed}")
        
        # 1. Device Setup
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        print(f"ACE_STEP_TRANSCRIBER: Using device: {device}")

        # 2. Path Resolution
        load_path = model_id
        # If it's not an absolute path and not existing, try to find it in models dir
        if not os.path.exists(load_path) and not os.path.isabs(load_path):
            potential_paths = [
                os.path.join(folder_paths.models_dir, load_path),
                os.path.join(folder_paths.models_dir, ACESTEP_MODEL_NAME, load_path),
            ]
            for p in potential_paths:
                if os.path.exists(p):
                    load_path = p
                    break
        
        print(f"ACE_STEP_TRANSCRIBER: Resolved model path: {load_path}")

        # 2.5 Memory Management
        if device == "cuda":
            torch.cuda.empty_cache()

        # 3. Model Loading
        # We use pipeline to load because it handles the custom Qwen2.5-Omni configuration reliably
        # where AutoModel sometimes fails with 'Incompatible safetensors'.
        from transformers import pipeline, AutoProcessor

        try:
            print(f"ACE_STEP_TRANSCRIBER: Loading pipeline from {load_path}...")
            
            # Determine dtype
            torch_dtype = torch.float32 # default
            if dtype == "auto":
                 torch_dtype = torch.float16 if device == "cuda" else torch.float32
            elif dtype == "float16":
                 torch_dtype = torch.float16
            elif dtype == "float32":
                 torch_dtype = torch.float32
            
            # Load pipeline
            # trust_remote_code=True is CRITICAL for Qwen2.5-Omni
            asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model=load_path,
                device=device,
                trust_remote_code=True,
                torch_dtype=torch_dtype
            )
            
            
            model = asr_pipeline.model
            
            # CRITICAL: Disable audio generation to prevent OOM
            # Qwen2.5-Omni tries to generate audio (DiT) by default during generation
            # We only want text transcription.
            if hasattr(model.config, "disable_audio_generation"):
                model.config.disable_audio_generation = True
            
            # Also try setting it on generation_config if checked there
            if hasattr(model, "generation_config"):
                 if hasattr(model.generation_config, "disable_audio_generation"):
                    model.generation_config.disable_audio_generation = True

            # Monkeypatch token2wav to ABSOLUTELY prevent OOM if the flags are ignored
            # The model calls self.token2wav(...) which triggers the DiT generation.
            # We must replace it with a proper nn.Module, not a lambda, to satisfy PyTorch checks.
            if hasattr(model, 'token2wav'):
                 print("ACE_STEP_TRANSCRIBER: Monkeypatching model.token2wav to prevent audio generation and OOM.")
                 
                 class DummyModule(torch.nn.Module):
                     def __init__(self):
                         super().__init__()
                         # Register buffer to track device/dtype
                         self.register_buffer("dummy_param", torch.tensor(0.0))

                     def forward(self, *args, **kwargs):
                         # Return empty tensor instead of None to satisfy .float() check in generate()
                         return torch.tensor([], device=self.dummy_param.device)
                     
                     @property
                     def dtype(self):
                        return self.dummy_param.dtype
                 
                 model.token2wav = DummyModule().to(device)

            # Load processor separately to access features
            processor = AutoProcessor.from_pretrained(load_path, trust_remote_code=True)
            
            print("ACE_STEP_TRANSCRIBER: Model and Processor loaded successfully.")

        except Exception as e:
            raise RuntimeError(f"Failed to load ASR model {model_id}: {e}")

        # 4. Audio Preparation
        # ComfyUI audio: {'waveform': [1, channels, samples], 'sample_rate': int}
        waveform = audio['waveform']
        sample_rate = audio['sample_rate']
        
        # Convert to numpy and mono
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
        
        # waveform shape is [batch, channels, samples]. 
        if waveform.ndim == 3:
            waveform = waveform[0] # [channels, samples]
        
        # Mix to mono if stereo
        if waveform.ndim > 1 and waveform.shape[0] > 1:
            waveform = np.mean(waveform, axis=0)
        elif waveform.ndim > 1:
             waveform = waveform[0]
            
        # Ensure float32
        waveform = waveform.astype(np.float32)

        # Resample to 16kHz if needed (WhisperFeatureExtractor requires 16000Hz)
        TARGET_SR = 16000
        if sample_rate != TARGET_SR:
            print(f"ACE_STEP_TRANSCRIBER: Resampling audio from {sample_rate}Hz to {TARGET_SR}Hz")
            try:
                waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=TARGET_SR)
                sample_rate = TARGET_SR
            except Exception as e:
                print(f"ACE_STEP_TRANSCRIBER: Resampling failed: {e}")
                raise e

        # Calculate audio duration
        audio_duration = len(waveform) / sample_rate
        print(f"ACE_STEP_TRANSCRIBER: Audio duration: {audio_duration:.1f}s")

        # Determine chunking strategy
        OVERLAP_S = 5  # 5 second overlap between chunks
        chunk_samples = int(chunk_length_s * sample_rate)
        overlap_samples = int(OVERLAP_S * sample_rate)

        # Only chunk if audio is longer than chunk_length
        needs_chunking = audio_duration > chunk_length_s

        if needs_chunking:
            # Calculate number of chunks
            step_samples = chunk_samples - overlap_samples
            num_chunks = int(np.ceil((len(waveform) - overlap_samples) / step_samples))
            print(f"ACE_STEP_TRANSCRIBER: Splitting into {num_chunks} chunks (chunk={chunk_length_s}s, overlap={OVERLAP_S}s)")
        else:
            num_chunks = 1
            print(f"ACE_STEP_TRANSCRIBER: Processing as single chunk")

        # 5. Inference
        try:
            # Construct Prompt with Chat Template
            # Official recommended prompt from ACE-Step: "Transcribe this audio in detail"
            # See: https://huggingface.co/ACE-Step/acestep-transcriber
            instruction = "Transcribe this audio in detail"

            lang_map = {
                "zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean",
                "fr": "French", "de": "German", "es": "Spanish", "it": "Italian",
                "ru": "Russian", "pt": "Portuguese"
            }

            if custom_prompt.strip():
                instruction = custom_prompt.strip()
            elif language in lang_map:
                # Use official prompt format with language specification
                instruction = f"Transcribe this audio in detail into {lang_map[language]}."

            # Build text prompt
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

            print(f"ACE_STEP_TRANSCRIBER: Using prompt: '{text_prompt[:80]}...'")
            print(f"ACE_STEP_TRANSCRIBER: Generation params: temp={temperature}, rep_penalty={repetition_penalty}")

            # Helper function to transcribe a single chunk
            def transcribe_chunk(audio_chunk, chunk_idx=None):
                inputs = processor(
                    text=[text_prompt],
                    audio=audio_chunk,
                    sampling_rate=sample_rate,
                    return_tensors="pt"
                )

                inputs = {k: v.to(device) for k, v in inputs.items()}
                if torch_dtype == torch.float16 and "input_features" in inputs:
                    inputs["input_features"] = inputs["input_features"].to(dtype=torch.float16)

                max_new_tokens = 512
                pbar = comfy.utils.ProgressBar(max_new_tokens)
                streamer = ComfyStreamer(pbar)

                with torch.no_grad():
                    generation_output = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        streamer=streamer,
                        temperature=temperature,
                        repetition_penalty=repetition_penalty,
                        do_sample=True if temperature > 0 else False,
                        return_audio=False
                    )

                generated_ids = generation_output
                full_output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                # Extract assistant's response
                transcription = full_output
                assistant_marker = "assistant"
                if assistant_marker in full_output:
                    parts = full_output.split(assistant_marker)
                    if len(parts) > 1:
                        transcription = parts[-1].strip()

                for marker in ["system", "user"]:
                    if f"\n{marker}" in transcription:
                        transcription = transcription.split(f"\n{marker}")[0].strip()

                return transcription

            # Process chunks
            all_transcriptions = []

            if needs_chunking:
                step_samples = chunk_samples - overlap_samples
                for i in range(num_chunks):
                    start = i * step_samples
                    end = min(start + chunk_samples, len(waveform))
                    chunk = waveform[start:end]

                    chunk_duration = len(chunk) / sample_rate
                    print(f"ACE_STEP_TRANSCRIBER: Processing chunk {i+1}/{num_chunks} ({chunk_duration:.1f}s)")

                    chunk_result = transcribe_chunk(chunk, chunk_idx=i)
                    all_transcriptions.append(chunk_result)
                    print(f"ACE_STEP_TRANSCRIBER: Chunk {i+1} result: {chunk_result[:50]}...")

                # Merge transcriptions
                # Simple merge: concatenate with newlines, remove duplicate headers
                merged = []
                seen_header = False
                for trans in all_transcriptions:
                    lines = trans.split('\n')
                    for line in lines:
                        # Skip duplicate language headers after first chunk
                        if line.strip().startswith('# Languages') or line.strip().startswith('# Lyrics'):
                            if not seen_header or line.strip() != '# Lyrics':
                                merged.append(line)
                                seen_header = True
                        else:
                            merged.append(line)

                transcription = '\n'.join(merged)
            else:
                print("ACE_STEP_TRANSCRIBER: Running inference...")
                transcription = transcribe_chunk(waveform)

            print(f"ACE_STEP_TRANSCRIBER: Final result: {transcription[:100]}...")
            return (transcription,)

        except Exception as e:
            print(f"ACE_STEP_TRANSCRIBER: Inference failed: {e}")
            raise e
