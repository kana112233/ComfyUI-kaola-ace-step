
import os
import torch
import torch.nn.functional as F
from acestep.training import trainer as trainer_mod

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
        
        # CRITICAL FIX for LoRA + Gradient Checkpointing:
        # If the model uses gradient checkpointing, the input must require grad 
        # to allow gradients to flow back through the checkpoints.
        xt.requires_grad_(True)
        
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
        
        # Debug graph connection
        if not getattr(self, '_grad_fn_logged', False):
            print(f"[ACE_STEP DEBUG] Loss grad_fn: {diffusion_loss.grad_fn}")
            print(f"  Input xt requires_grad: {xt.requires_grad}")
            self._grad_fn_logged = True
            
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


# -------------------------------------------------------------------------
# ROBUST TRAINING CLASSES (Fix for ComfyUI no_grad environment)
# -------------------------------------------------------------------------
class SafePreprocessedLoRAModule(trainer_mod.PreprocessedLoRAModule):
    """Subclass that forces gradients enabled during training step."""
    def _unwrap_compiled_model(self, model):
        """Recursively unwrap compiled/optimized models to get the raw nn.Module."""
        # Unwrap torch.compile / OptimizedModule
        if hasattr(model, "_orig_mod"):
            return self._unwrap_compiled_model(model._orig_mod)
        
        # Unwrap other potential wrappers (like DDP, etc)
        if hasattr(model, "module"):
            return self._unwrap_compiled_model(model.module)
        
        # Unwrap _forward_module (common in some compile backends)
        if hasattr(model, "_forward_module"):
            return self._unwrap_compiled_model(model._forward_module)
            
        return model

    def training_step(self, batch):
        # FORCE ENABLE GRADIENTS
        torch.set_grad_enabled(True)
        
        # Ensure we are working with the raw model, not a compiled one
        # Compiled models might have captured a no_grad graph
        if not getattr(self, '_unwrapped', False):
            self.model = self._unwrap_compiled_model(self.model)
            if hasattr(self.model, 'decoder'):
                self.model.decoder = self._unwrap_compiled_model(self.model.decoder)
            self._unwrapped = True

        # FORCE TRAIN MODE
        self.model.train()
        if hasattr(self.model, 'decoder'):
            self.model.decoder.train()
        
        # Forward using robustness patch (this fixes dtype and no_grad issues)
        loss = patched_training_step(self, batch)
        
        return loss

class SafeLoRATrainer(trainer_mod.LoRATrainer):
    """Subclass that uses SafePreprocessedLoRAModule."""
    def train_from_preprocessed(self, tensor_dir, training_state=None, resume_from=None):
        self.is_training = True
        try:
            # Validate tensor directory
            if not os.path.exists(tensor_dir):
                yield 0, 0.0, f"❌ Tensor directory not found: {tensor_dir}"
                return
            
            # Create SAFE training module
            self.module = SafePreprocessedLoRAModule(
                model=self.dit_handler.model,
                lora_config=self.lora_config,
                training_config=self.training_config,
                device=self.dit_handler.device,
                dtype=self.dit_handler.dtype,
            )
            
            # Everything else is the same as original, but we can't easily call super() 
            # because it hardcodes the class instantiation.
            # So we delegate to the _train methods which use self.module
            
            # Create data module
            from acestep.training.data_module import PreprocessedDataModule
            data_module = PreprocessedDataModule(
                tensor_dir=tensor_dir,
                batch_size=self.training_config.batch_size,
                num_workers=self.training_config.num_workers,
                pin_memory=self.training_config.pin_memory,
            )
            
            # Setup data
            data_module.setup('fit')
            
            if len(data_module.train_dataset) == 0:
                yield 0, 0.0, "❌ No valid samples found in tensor directory"
                return
            
            dataset_size = len(data_module.train_dataset)
            yield 0, 0.0, f"📂 Loaded {dataset_size} preprocessed samples"
            
            # Calculate total steps for Progress Bar
            # steps = (dataset_size // batch_size) * epochs + (epochs if remainder) ... strictly:
            # steps_per_epoch = len(train_loader) 
            # But we don't have loader yet. data_module.train_dataloader() creates it.
            # Lightning/Fabric handles this, but we can estimate.
            
            batch_size = self.training_config.batch_size
            grad_accum = self.training_config.gradient_accumulation_steps
            max_epochs = self.training_config.max_epochs
            
            # steps_per_epoch = math.ceil(dataset_size / batch_size)
            import math
            steps_per_epoch = math.ceil(dataset_size / batch_size)
            total_steps = (steps_per_epoch * max_epochs) // grad_accum
            
            yield 0, 0.0, f"TOTAL_STEPS: {total_steps}"

            # Disconnnect from ComfyUI's VRAM management temporarily implies we handle it? 
            # No, just run the loop.
            
            if trainer_mod.LIGHTNING_AVAILABLE:
                yield from self._train_with_fabric(data_module, training_state, resume_from)
            else:
                yield from self._train_basic(data_module, training_state)
                
        except Exception as e:
            trainer_mod.logger.exception("Training failed")
            yield 0, 0.0, f"❌ Training failed: {str(e)}"
        finally:
            self.is_training = False
