
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
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("transcription",)
    FUNCTION = "transcribe"
    CATEGORY = "ACE_STEP"

    def transcribe(self, audio, model_id, device, dtype, language, chunk_length_s, return_timestamps, custom_prompt="", temperature=0.2, repetition_penalty=1.1):
        print(f"ACE_STEP_TRANSCRIBER: Transcribing with model {model_id} on {device} ({dtype})")
        
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

        # Resample to 16kHz if needed (Qwen2.5-Omni/Whisper requires 16000)
        TARGET_SR = 16000
        if sample_rate != TARGET_SR:
            print(f"ACE_STEP_TRANSCRIBER: Resampling audio from {sample_rate}Hz to {TARGET_SR}Hz")
            try:
                waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=TARGET_SR)
                sample_rate = TARGET_SR
            except Exception as e:
                print(f"ACE_STEP_TRANSCRIBER: Resampling failed: {e}")
                raise e

        # 5. Inference
        try:
            print("ACE_STEP_TRANSCRIBER: Running inference...")
            
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
            
            # Use apply_chat_template if available for correct special token formatting
            if hasattr(processor, "apply_chat_template"):
                # Qwen2.5-Omni processor expects content to be a list of dictionaries
                # e.g. [{"type": "text", "text": "..."}]
                messages = [
                    {
                        "role": "system", 
                        "content": [{"type": "text", "text": "You are a helpful assistant."}]
                    },
                    {
                        "role": "user", 
                        "content": [{"type": "text", "text": instruction}]
                    }
                ]
                
                # Qwen2.5-Omni processor usually handles <|audio_bos|><|AUDIO|><|audio_eos|> automatically
                # when passing 'audio' argument to processor(). 
                # We just need the text conversation format.
                text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            else:
                # Fallback manual formatting
                text_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

            print(f"ACE_STEP_TRANSCRIBER: Using prompt: '{text_prompt}'")
            print(f"ACE_STEP_TRANSCRIBER: Generation params: temp={temperature}, rep_penalty={repetition_penalty}")

            # Use processor to prepare inputs
            inputs = processor(
                text=[text_prompt], 
                audio=waveform, 
                sampling_rate=sample_rate, 
                return_tensors="pt"
            )
            
            # Move to device and cast to correct dtype
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # Note: feature extractor usually returns float32, but model might expect half if loaded in half
            if torch_dtype == torch.float16 and "input_features" in inputs:
                 inputs["input_features"] = inputs["input_features"].to(dtype=torch.float16)

            # Setup Streamer for Progress Bar
            max_new_tokens = 512
            pbar = comfy.utils.ProgressBar(max_new_tokens)
            streamer = ComfyStreamer(pbar)

            # Generate
            # max_new_tokens can be adjustable, but 256 or 512 is safe for standard sentences
            # return_audio=False: Only return text, skip audio generation (saves memory and time)
            with torch.no_grad():
                generation_output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    streamer=streamer,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    do_sample=True if temperature > 0 else False,
                    return_audio=False  # Official Qwen2.5-Omni param: only return text
                )

            # With return_audio=False, output should be just the token ids
            generated_ids = generation_output

            # Decode full output first
            full_output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Extract only the assistant's response
            # The prompt ends with "<|im_start|>assistant\n" and we want everything after that
            transcription = full_output

            # Try to extract content after the assistant marker
            # Format: system\n...\nuser\n...\nassistant\n<ACTUAL_RESPONSE>
            assistant_marker = "assistant"
            if assistant_marker in full_output:
                parts = full_output.split(assistant_marker)
                if len(parts) > 1:
                    # Take the last part (after the final "assistant")
                    transcription = parts[-1].strip()

            # Also try to remove any remaining role markers that might have been generated
            # (in case the model continued generating more conversation turns)
            for marker in ["system", "user"]:
                if f"\n{marker}" in transcription:
                    transcription = transcription.split(f"\n{marker}")[0].strip()
            
            print(f"ACE_STEP_TRANSCRIBER: Result: {transcription[:50]}...")
            return (transcription,)
            
        except Exception as e:
            print(f"ACE_STEP_TRANSCRIBER: Inference failed: {e}")
            raise e
