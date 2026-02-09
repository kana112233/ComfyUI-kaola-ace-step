"""
ComfyUI-compatible LoRA Trainer

This is a standalone trainer that works with ComfyUI's execution model.
It copies and modifies the ACE-Step training logic to avoid BFloat16 issues.
"""

import os
import time
from typing import Generator, Tuple, Dict, Optional
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, SequentialLR


class ComfyUILoRATrainer:
    """
    Standalone LoRA trainer for ComfyUI.

    No monkey patches needed - all training logic is self-contained.
    Uses fp32/bfloat16 hybrid approach for ComfyUI compatibility.
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
        self.device = dit_handler.device
        self.module = None
        self.is_training = False

    def _create_lora_module(self):
        """Create the LoRA training module."""
        from acestep.training.trainer import PreprocessedLoRAModule

        self.module = PreprocessedLoRAModule(
            model=self.dit_handler.model,
            lora_config=self.lora_config,
            training_config=self.training_config,
            device=self.dit_handler.device,
            dtype=self.dit_handler.dtype,
        )

    def _training_step(self, batch):
        """
        Single training step with ComfyUI-compatible logging.

        Modified from ACE-Step original to fix bfloat16 numpy issues.
        """
        from acestep.training.trainer import sample_discrete_timestep

        # Get tensors from batch
        target_latents = batch["target_latents"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        encoder_hidden_states = batch["encoder_hidden_states"].to(self.device)
        encoder_attention_mask = batch["encoder_attention_mask"].to(self.device)
        context_latents = batch["context_latents"].to(self.device)

        bsz = target_latents.shape[0]

        # Detect model dtype
        model_dtype = next(self.module.model.parameters()).dtype

        # Flow matching: sample noise x1 and interpolate with data x0
        x1 = torch.randn_like(target_latents, dtype=model_dtype)
        x0 = target_latents

        # Sample timesteps
        t, r = sample_discrete_timestep(bsz, self.device, model_dtype)
        t_ = t.unsqueeze(-1).unsqueeze(-1)

        # Interpolate: x_t = t * x1 + (1 - t) * x0
        xt = t_ * x1 + (1.0 - t_) * x0

        # Determine autocast dtype
        _device_type = self.device if isinstance(self.device, str) else self.device.type
        _autocast_dtype = torch.float16 if _device_type == "mps" else model_dtype

        # Forward pass with autocast
        with torch.autocast(device_type=_device_type, dtype=_autocast_dtype):
            decoder_outputs = self.module.model.decoder(
                hidden_states=xt,
                timestep=t,
                timestep_r=r,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                context_latents=context_latents,
            )

            # Flow matching loss
            flow = x1 - x0
            diffusion_loss = F.mse_loss(decoder_outputs[0], flow)

        # Store loss for logging (convert to float for stability)
        self.module.training_losses.append(diffusion_loss.detach().float().item())

        # Return loss with gradients preserved
        return diffusion_loss

    def train_from_preprocessed(
        self,
        tensor_dir: str,
        training_state: Optional[Dict] = None,
        resume_from: Optional[str] = None,
    ) -> Generator[Tuple[int, float, str], None, None]:
        """
        Main training loop.

        Args:
            tensor_dir: Directory containing preprocessed .pt files
            training_state: Optional state dict for stopping control
            resume_from: Optional path to checkpoint directory to resume from

        Yields:
            Tuples of (step, loss, status_message)
        """
        from acestep.training.data_module import PreprocessedTensorDataset
        from acestep.training.lora_utils import save_lora_weights

        # Validate tensor directory
        tensor_path = Path(tensor_dir)
        if not tensor_path.exists():
            yield 0, 0.0, f"❌ Tensor directory not found: {tensor_dir}"
            return

        # Create output directory
        os.makedirs(self.training_config.output_dir, exist_ok=True)

        # Create training module
        print("[ComfyUILoRATrainer] Creating PreprocessedLoRAModule...")
        self._create_lora_module()

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

        # Setup training
        print(f"[ComfyUILoRATrainer] Starting training loop...")
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

                # Forward pass using our custom training_step
                loss = self._training_step(batch)
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
        yield global_step, avg_epoch_loss, f"✅ Training complete! LoRA saved to {final_path}"
