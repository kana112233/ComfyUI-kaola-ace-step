"""
LoRA Training Module for ACE-Step ComfyUI Nodes

This module provides training utilities that are compatible with ComfyUI's worker thread model.
It uses fp32 precision instead of bf16 to avoid threading issues.
"""

import os
import time
from typing import Generator, Tuple, Dict, Optional
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, SequentialLR


class ComfyUITrainer:
    """
    Trainer class that works with ComfyUI's execution model.
    Uses fp32 precision to avoid BFloat16 threading issues.
    """

    def __init__(self, dit_handler, lora_config, training_config):
        """
        Initialize the trainer.

        Args:
            dit_handler: ACE-Step DiT handler with loaded model
            lora_config: LoRA configuration object
            training_config: Training configuration object
        """
        self.dit_handler = dit_handler
        self.lora_config = lora_config
        self.training_config = training_config
        self.device = dit_handler.device  # Get device from handler
        self.module = None
        self.is_training = False

    def train_from_preprocessed(
        self,
        tensor_dir: str,
        training_state: Optional[Dict] = None,
        resume_from: Optional[str] = None,
    ) -> Generator[Tuple[int, float, str], None, None]:
        """
        Main training loop without Fabric, using fp32 precision.

        This is a ComfyUI-compatible version that avoids BFloat16 issues.

        Args:
            tensor_dir: Directory containing preprocessed .pt files
            training_state: Optional state dict for stopping control
            resume_from: Optional path to checkpoint directory to resume from

        Yields:
            Tuples of (step, loss, status_message)
        """
        from acestep.training.trainer import PreprocessedLoRAModule
        from acestep.training.data_module import PreprocessedTensorDataset
        from acestep.training.lora_utils import save_lora_weights

        # Validate tensor directory
        tensor_path = Path(tensor_dir)
        if not tensor_path.exists():
            yield 0, 0.0, f"❌ Tensor directory not found: {tensor_dir}"
            return

        # Create output directory
        os.makedirs(self.training_config.output_dir, exist_ok=True)

        # Create training module FIRST (critical step that was missing)
        print("[ComfyUITrainer] Creating PreprocessedLoRAModule...")
        self.module = PreprocessedLoRAModule(
            model=self.dit_handler.model,
            lora_config=self.lora_config,
            training_config=self.training_config,
            device=self.dit_handler.device,
            dtype=self.dit_handler.dtype,
        )

        # Create dataset and dataloader
        dataset = PreprocessedTensorDataset(tensor_dir)

        if len(dataset) == 0:
            yield 0, 0.0, "❌ No valid samples found in tensor directory"
            return

        train_loader = DataLoader(
            dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=self.training_config.num_workers,
            pin_memory=self.training_config.pin_memory,
        )

        yield 0, 0.0, f"📂 Loaded {len(dataset)} preprocessed samples ({len(train_loader)} batches)"

        # Count trainable parameters
        trainable_params = [p for p in self.module.model.parameters() if p.requires_grad]

        if not trainable_params:
            yield 0, 0.0, "❌ No trainable parameters found!"
            return

        param_count = sum(p.numel() for p in trainable_params)
        yield 0, 0.0, f"🎯 Training {param_count:,} parameters"

        # Create optimizer
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )

        # Calculate total steps
        total_steps = (
            len(train_loader) * self.training_config.max_epochs // self.training_config.gradient_accumulation_steps
        )
        warmup_steps = min(self.training_config.warmup_steps, max(1, total_steps // 10))

        # Create scheduler
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        main_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, total_steps - warmup_steps),
            T_mult=1,
            eta_min=self.training_config.learning_rate * 0.01,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )

        # Use fp32 precision (NOT bf16 - ComfyUI worker threads don't support it)
        print(f"[ComfyUITrainer] Using fp32 precision (bf16 disabled for ComfyUI compatibility)")
        self.module.model.train()
        self.module.model.to(self.device)

        global_step = 0
        accumulation_step = 0
        accumulated_loss = 0.0

        for epoch in range(self.training_config.max_epochs):
            epoch_loss = 0.0
            num_batches = 0
            epoch_start_time = time.time()

            for batch in train_loader:
                # Check for stop signal
                if training_state and training_state.get("should_stop", False):
                    yield global_step, accumulated_loss / max(accumulation_step, 1), "⏹️ Training stopped"
                    return

                # Forward pass
                loss = self.module.training_step(batch)
                loss = loss / self.training_config.gradient_accumulation_steps
                loss.backward()

                accumulated_loss += loss.item()
                accumulation_step += 1

                # Optimizer step with gradient accumulation
                if accumulation_step >= self.training_config.gradient_accumulation_steps:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params, self.training_config.max_grad_norm
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % self.training_config.log_every_n_steps == 0:
                        avg_loss = accumulated_loss / accumulation_step
                        yield (
                            global_step,
                            avg_loss,
                            f"Epoch {epoch+1}/{self.training_config.max_epochs}, Step {global_step}/{total_steps}, Loss: {avg_loss:.4f}",
                        )

                    epoch_loss += accumulated_loss
                    num_batches += 1
                    accumulated_loss = 0.0
                    accumulation_step = 0

            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            yield (
                global_step,
                avg_epoch_loss,
                f"✅ Epoch {epoch+1}/{self.training_config.max_epochs} complete in {epoch_time:.1f}s, Loss: {avg_epoch_loss:.4f}",
            )

            # Save checkpoint
            if (epoch + 1) % self.training_config.save_every_n_epochs == 0:
                checkpoint_dir = os.path.join(
                    self.training_config.output_dir, "checkpoints", f"epoch_{epoch+1}"
                )
                save_lora_weights(self.module.model, checkpoint_dir)
                yield global_step, avg_epoch_loss, f"💾 Checkpoint saved to {checkpoint_dir}"

        # Save final LoRA
        final_path = os.path.join(self.training_config.output_dir, "final")
        save_lora_weights(self.module.model, final_path)
        final_loss = avg_epoch_loss
        yield global_step, final_loss, f"✅ Training complete! LoRA saved to {final_path}"


def apply_comfyui_training_patches():
    """
    Apply ComfyUI-compatible training patches to ACE-Step.

    This patches the LoRATrainer to use our ComfyUITrainer instead of
    the original training logic that has BFloat16 compatibility issues.
    """
    try:
        import acestep.training.trainer as trainer_module

        # Save original train_from_preprocessed
        _original_train_from_preprocessed = trainer_module.LoRATrainer.train_from_preprocessed

        def patched_train_from_preprocessed(
            self, tensor_dir, training_state=None, resume_from=None
        ):
            """Patched training that uses ComfyUITrainer."""
            # Set training flag
            self.is_training = True

            try:
                # Create our trainer instance
                trainer = ComfyUITrainer(
                    dit_handler=self.dit_handler,
                    lora_config=self.lora_config,
                    training_config=self.training_config,
                )
                # Delegate to our trainer
                yield from trainer.train_from_preprocessed(
                    tensor_dir=tensor_dir,
                    training_state=training_state,
                    resume_from=resume_from,
                )
            finally:
                self.is_training = False

        # Apply the patch
        trainer_module.LoRATrainer.train_from_preprocessed = patched_train_from_preprocessed
        print("[MonkeyPatch] LoRA training patched to use ComfyUITrainer with fp32 precision")
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
        import numpy as np

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


def apply_all_training_patches():
    """
    Apply all training-related patches for ComfyUI compatibility.
    """
    print("[MonkeyPatch] Applying training patches for ComfyUI compatibility...")
    apply_torchaudio_soundfile_patch()
    apply_comfyui_training_patches()
