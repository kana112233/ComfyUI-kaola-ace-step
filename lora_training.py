"""
LoRA Training Module for ACE-Step ComfyUI Nodes

Minimal patching approach - use original ACE-Step training code,
only fix the bfloat16 numpy conversion issue in logging.
"""

import torch


def apply_training_fixes():
    """
    Apply minimal fixes to ACE-Step training for ComfyUI compatibility.

    Only patches:
    1. torchaudio.load -> soundfile (fixes libtorchcodec errors)
    2. sample_discrete_timestep -> force float32 (fixes dtype mismatch)
    3. training_step -> pure fp32 without autocast (fixes gradient issues)

    Everything else uses original ACE-Step code.
    """
    # Fix 1: Patch torchaudio to use soundfile
    try:
        import torchaudio
        import soundfile as sf

        _original_torchaudio_load = torchaudio.load

        def patched_torchaudio_load(filepath, *args, **kwargs):
            try:
                data, sr = sf.read(filepath, dtype='float32')
                if data.ndim == 1:
                    audio = torch.from_numpy(data).unsqueeze(0)
                else:
                    audio = torch.from_numpy(data.T)
                return audio, sr
            except Exception:
                return _original_torchaudio_load(filepath, *args, **kwargs)

        torchaudio.load = patched_torchaudio_load
        print("[ComfyUI-ACE-Step] Patched torchaudio.load -> soundfile")
    except ImportError:
        pass

    # Fix 1.5: Patch sample_discrete_timestep to return float32
    try:
        from acestep.training.trainer import sample_discrete_timestep

        _original_sample_timestep = sample_discrete_timestep

        def patched_sample_timestep(bsz, device, dtype):
            """Force float32 timesteps for ComfyUI compatibility."""
            return _original_sample_timestep(bsz, device, torch.float32)

        sample_discrete_timestep = patched_sample_timestep
        print("[ComfyUI-ACE-Step] Patched sample_discrete_timestep -> float32")
    except Exception as e:
        print(f"[ComfyUI-ACE-Step] Could not patch sample_discrete_timestep: {e}")

    # Fix 2: Patch training_step to use fp32 instead of autocast (ComfyUI compatibility)
    try:
        from acestep.training.trainer import PreprocessedLoRAModule

        _original_training_step = PreprocessedLoRAModule.training_step

        def patched_training_step(self, batch):
            """Training step using pure fp32 (no autocast) for ComfyUI compatibility."""
            import torch.nn.functional as F
            from acestep.training.trainer import sample_discrete_timestep

            # Get tensors - convert to fp32
            target_latents = batch["target_latents"].to(self.device).float()
            attention_mask = batch["attention_mask"].to(self.device)
            encoder_hidden_states = batch["encoder_hidden_states"].to(self.device).float()
            encoder_attention_mask = batch["encoder_attention_mask"].to(self.device)
            context_latents = batch["context_latents"].to(self.device).float()

            bsz = target_latents.shape[0]

            # No autocast - use pure fp32 operations
            x1 = torch.randn_like(target_latents, dtype=torch.float32)
            x0 = target_latents

            # Sample timesteps in float32
            t, r = sample_discrete_timestep(bsz, self.device, torch.float32)
            t_ = t.unsqueeze(-1).unsqueeze(-1)

            # Interpolate
            xt = t_ * x1 + (1.0 - t_) * x0

            # Forward pass - no autocast
            decoder_outputs = self.model.decoder(
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

            # Store for logging
            self.training_losses.append(diffusion_loss.item())

            # Return fp32 loss with gradients
            return diffusion_loss

        PreprocessedLoRAModule.training_step = patched_training_step
        print("[ComfyUI-ACE-Step] Patched training_step (pure fp32, no autocast)")

    except Exception as e:
        print(f"[ComfyUI-ACE-Step] Could not patch training_step: {e}")
        import traceback
        traceback.print_exc()

    print("[ComfyUI-ACE-Step] Training fixes applied - using original ACE-Step training logic")
