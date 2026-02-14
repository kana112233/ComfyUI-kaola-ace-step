
import os
import torch
import numpy as np
import librosa
import folder_paths # Ensure this is accessible. It is provided by ComfyUI context.

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
                "chunk_length_s": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 300.0, "step": 1.0, "tooltip": "Audio chunk length in seconds for processing."}),
                "return_timestamps": (["true", "false", "word"], {"default": "false", "tooltip": "Whether to return timestamps. 'word' for word-level timestamps, 'true' for segment-level."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("transcription",)
    FUNCTION = "transcribe"
    CATEGORY = "ACE_STEP"

    def transcribe(self, audio, model_id, device, chunk_length_s, return_timestamps):
        print(f"ACE_STEP_TRANSCRIBER: Transcribing with model {model_id} on {device}")
        
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

        # 3. Model Loading
        # We use pipeline to load because it handles the custom Qwen2.5-Omni configuration reliably
        # where AutoModel sometimes fails with 'Incompatible safetensors'.
        from transformers import pipeline, AutoProcessor

        try:
            print(f"ACE_STEP_TRANSCRIBER: Loading pipeline from {load_path}...")
            
            # Load pipeline
            # trust_remote_code=True is CRITICAL for Qwen2.5-Omni
            asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model=load_path,
                device=device,
                trust_remote_code=True
            )
            
            model = asr_pipeline.model
            
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
            # librosa.resample expects [channels, samples] or [samples]
            # Our waveform is [samples] (mono) at this point
            try:
                waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=TARGET_SR)
                sample_rate = TARGET_SR
            except Exception as e:
                print(f"ACE_STEP_TRANSCRIBER: Resampling failed: {e}")
                raise e

        # 5. Inference
        try:
            print("ACE_STEP_TRANSCRIBER: Running inference...")
            
            # Qwen2.5-Omni requires a text prompt.
            prompt = "Transcribe the audio:"
            
            # Use processor to prepare inputs
            inputs = processor(
                text=[prompt], 
                audio=waveform, 
                sampling_rate=sample_rate, 
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            # max_new_tokens can be adjustable, but 256 or 512 is safe for standard sentences
            generated_ids = model.generate(**inputs, max_new_tokens=512)
            
            # Decode
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            print(f"ACE_STEP_TRANSCRIBER: Result: {transcription[:50]}...")
            return (transcription,)
            
        except Exception as e:
            print(f"ACE_STEP_TRANSCRIBER: Inference failed: {e}")
            raise e
