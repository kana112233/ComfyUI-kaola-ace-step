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
    2. logging bfloat16 -> float32 conversion (fixes numpy errors)

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

    # Fix 2: Patch logging to handle bfloat16 tensors
    try:
        from acestep.training.trainer import PreprocessedLoRAModule
        import logging

        _original_training_step = PreprocessedLoRAModule.training_step

        def patched_training_step(self, batch):
            """Original training_step with fixed logging."""
            import torch.nn.functional as F
            from acestep.training.trainer import sample_discrete_timestep
            from acestep.training import trainer

            # Get tensors
            target_latents = batch["target_latents"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            encoder_hidden_states = batch["encoder_hidden_states"].to(self.device)
            encoder_attention_mask = batch["encoder_attention_mask"].to(self.device)
            context_latents = batch["context_latents"].to(self.device)

            bsz = target_latents.shape[0]

            # Use autocast (like original)
            _device_type = self.device if isinstance(self.device, str) else self.device.type
            _autocast_dtype = torch.float16 if _device_type == "mps" else torch.bfloat16

            with torch.autocast(device_type=_device_type, dtype=_autocast_dtype):
                x1 = torch.randn_like(target_latents)
                x0 = target_latents
                t, r = sample_discrete_timestep(bsz, self.device, torch.bfloat16)
                t_ = t.unsqueeze(-1).unsqueeze(-1)
                xt = t_ * x1 + (1.0 - t_) * x0

                decoder_outputs = self.model.decoder(
                    hidden_states=xt,
                    timestep=t,
                    timestep_r=r,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    context_latents=context_latents,
                )

                flow = x1 - x0
                diffusion_loss = F.mse_loss(decoder_outputs[0], flow)

            # FIX: Convert to float AFTER autocast, but for logging only
            loss_item = diffusion_loss.float().item()
            self.training_losses.append(loss_item)

            # Return loss WITHOUT .float() - keeps it in autocast dtype with gradients
            return diffusion_loss

        PreprocessedLoRAModule.training_step = patched_training_step
        print("[ComfyUI-ACE-Step] Patched training_step logging (keeps original autocast)")

    except Exception as e:
        print(f"[ComfyUI-ACE-Step] Could not patch training_step: {e}")
        import traceback
        traceback.print_exc()

    print("[ComfyUI-ACE-Step] Training fixes applied - using original ACE-Step training logic")
