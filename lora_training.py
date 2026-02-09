"""
LoRA Training Module for ACE-Step ComfyUI Nodes

This module provides training utilities that are compatible with ComfyUI's worker thread model.
Uses ComfyUILoRATrainer which is a standalone implementation without monkey patches.
"""

import torch

# Try relative import first (when used as package), fall back to absolute
try:
    from .comfyui_trainer import ComfyUILoRATrainer
except ImportError:
    from comfyui_trainer import ComfyUILoRATrainer


def apply_all_training_patches():
    """
    Apply ComfyUI-compatible training "patches" to ACE-Step.

    Actually replaces the training method with our standalone trainer.
    No actual monkey patching of ACE-Step internals needed!
    """
    try:
        import acestep.training.trainer as trainer_module

        # Save original for reference
        _original_train_from_preprocessed = trainer_module.LoRATrainer.train_from_preprocessed

        def patched_train_from_preprocessed(
            self, tensor_dir, training_state=None, resume_from=None
        ):
            """Use our standalone ComfyUI trainer instead of original."""
            self.is_training = True

            try:
                # Create our standalone trainer
                trainer = ComfyUILoRATrainer(
                    dit_handler=self.dit_handler,
                    lora_config=self.lora_config,
                    training_config=self.training_config,
                )
                # Run training
                yield from trainer.train_from_preprocessed(
                    tensor_dir=tensor_dir,
                    training_state=training_state,
                    resume_from=resume_from,
                )
            finally:
                self.is_training = False

        # Replace the method
        trainer_module.LoRATrainer.train_from_preprocessed = patched_train_from_preprocessed
        print("[MonkeyPatch] LoRA training replaced with ComfyUI-compatible standalone trainer")
        return True

    except Exception as e:
        print(f"[MonkeyPatch] Could not apply training patches: {e}")
        import traceback
        traceback.print_exc()
        return False


def apply_torchaudio_soundfile_patch():
    """
    Patch torchaudio.load to use soundfile instead.

    This avoids libtorchcodec issues when loading audio in training.
    """
    try:
        import torchaudio
        import soundfile as sf

        # Save original
        _original_torchaudio_load = torchaudio.load

        def patched_torchaudio_load(filepath, *args, **kwargs):
            """Load audio using soundfile instead of torchaudio."""
            try:
                # Read with soundfile
                data, sr = sf.read(filepath, always_2d=False)
                # Convert to tensor
                waveform = torch.from_numpy(data).float()
                # Add channel dimension if needed
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                elif waveform.dim() == 2:
                    waveform = waveform.transpose(0, 1)  # (samples, channels) -> (channels, samples)
                else:
                    raise ValueError(f"Unexpected audio shape: {waveform.shape}")
                return waveform, sr
            except Exception as e:
                print(f"[patched_torchaudio_load] Error loading {filepath}: {e}")
                # Fallback to original torchaudio.load if soundfile fails
                try:
                    return _original_torchaudio_load(filepath, *args, **kwargs)
                except Exception as e2:
                    print(f"[patched_torchaudio_load] Original torchaudio also failed: {e2}")
                    raise

        # Replace torchaudio.load
        torchaudio.load = patched_torchaudio_load
        print("[MonkeyPatch] torchaudio.load patched to use soundfile for training")
        return True

    except ImportError as e:
        print(f"[MonkeyPatch] Could not patch torchaudio.load: {e}")
        return False
