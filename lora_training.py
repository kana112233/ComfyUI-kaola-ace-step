
import torch
import torch.nn.functional as F

# -------------------------------------------------------------------------
# HELPER: Turbo shift=3.0 discrete timesteps (8 steps)
# -------------------------------------------------------------------------
TURBO_SHIFT3_TIMESTEPS = [1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75, 0.6428571428571429, 0.5, 0.3]

def sample_discrete_timestep(bsz, device, dtype):
    """Sample timesteps from discrete turbo shift=3 schedule."""
    indices = torch.randint(0, len(TURBO_SHIFT3_TIMESTEPS), (bsz,), device=device)
    timesteps = torch.tensor([TURBO_SHIFT3_TIMESTEPS[i] for i in indices], device=device, dtype=dtype)
    return timesteps, indices

def patched_training_step(self, batch):
    """
    Robust training step that handles:
    1. Gradients enabled
    2. Correct dtype casting (bfloat16 vs float32)
    3. Model unwrapping
    4. Explicit forward pass bypassing external libraries
    """
    # FORCE ENABLE GRADIENTS
    torch.set_grad_enabled(True)
    
    # Debug Info (Once)
    if not getattr(self, '_debug_logged', False):
        print(f"[ACE_STEP DEBUG] patched_training_step start")
        print(f"  Grad Enabled: {torch.is_grad_enabled()}")
        print(f"  Model Training Mode: {self.model.training}")
        if hasattr(self.model, 'decoder'):
            print(f"  Decoder Training Mode: {self.model.decoder.training}")
        self._debug_logged = True

    try:
        # 1. Prepare data with correct dtype/device
        # We use self.device and self.dtype (bfloat16 usually)
        target_latents = batch["target_latents"].to(self.device).to(self.dtype)
        attention_mask = batch["attention_mask"].to(self.device)
        encoder_hidden_states = batch["encoder_hidden_states"].to(self.device).to(self.dtype)
        encoder_attention_mask = batch["encoder_attention_mask"].to(self.device)
        context_latents = batch["context_latents"].to(self.device).to(self.dtype)
        
        bsz = target_latents.shape[0]
        
        # 2. Flow matching noise sampling
        x1 = torch.randn_like(target_latents)
        x0 = target_latents
        
        # 3. Sample timesteps
        # CRITICAL FIX: sample_discrete_timestep might return float32 if not careful
        # We ensure 't' matches self.dtype (bfloat16)
        t, _ = sample_discrete_timestep(bsz, self.device, self.dtype)
        t_ = t.unsqueeze(-1).unsqueeze(-1)
        
        # 4. Interpolate
        xt = t_ * x1 + (1.0 - t_) * x0
        
        # 5. Connect to decoder
        # Handle both wrapped and unwrapped cases by checking for decoder attribute
        decoder_model = self.model.decoder if hasattr(self.model, 'decoder') else self.model
        
        # 6. Forward Pass
        # CRITICAL FIX: Ensure timestep is cast if the model expects something specific
        # But usually pass it as self.dtype
        decoder_outputs = decoder_model(
            hidden_states=xt,
            timestep=t,      # bfloat16
            timestep_r=t,    # bfloat16
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            context_latents=context_latents,
        )
        
        # 7. Calculate Loss
        # Flow matching loss: predict the flow field v = x1 - x0
        flow = x1 - x0
        diffusion_loss = F.mse_loss(decoder_outputs[0], flow)
        
        # 8. Return float32 loss for stability
        return diffusion_loss.float()
        
    except Exception as e:
        print(f"[ACE_STEP ERROR] Exception in patched_training_step: {e}")
        import traceback
        traceback.print_exc()
        raise e

# Legacy compatibility function
# Since we now override training_step directly in SafePreprocessedLoRAModule,
# we don't need to patch the external library.
def apply_all_training_patches():
    print("[ACE_STEP] Using inline training loop (patched_training_step). Legacy patches skipped.")
