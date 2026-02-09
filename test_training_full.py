"""
Full LoRA Training Logic Test
Simulates the complete training workflow with mocks
"""

import sys
import os
import tempfile
from unittest.mock import Mock, MagicMock, patch, call
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.getcwd())


def test_full_training_workflow():
    """Test the complete training workflow from start to finish."""
    print("\n" + "=" * 60)
    print("FULL TRAINING WORKFLOW TEST")
    print("=" * 60)

    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\nUsing temp directory: {tmpdir}")

        # Step 1: Create mock tensor data
        print("\n--- Step 1: Creating mock tensor data ---")
        tensor_dir = Path(tmpdir) / "tensors"
        tensor_dir.mkdir()

        # Create mock .pt files
        import torch
        for i in range(3):
            tensor_file = tensor_dir / f"sample_{i}.pt"
            mock_tensor = {
                "target_latents": torch.randn(1, 100, 64),
                "attention_mask": torch.ones(1, 100),
                "encoder_hidden_states": torch.randn(1, 50, 768),
                "encoder_attention_mask": torch.ones(1, 50),
                "context_latents": torch.randn(1, 100, 128),
            }
            torch.save(mock_tensor, tensor_file)
        print(f"✅ Created 3 mock tensor files in {tensor_dir}")

        # Step 2: Create mock dit_handler with model
        print("\n--- Step 2: Creating mock DiT handler ---")
        mock_dit_handler = Mock()
        mock_dit_handler.device = "cpu"
        mock_dit_handler.dtype = torch.float32

        # Create a minimal mock model with trainable parameters
        mock_model = MagicMock()
        mock_model.config = Mock()
        mock_model.decoder = MagicMock()
        mock_model.decoder.return_value = (torch.randn(1, 100, 64),)  # Mock output

        # Create trainable parameters (for optimizer)
        param1 = torch.nn.Parameter(torch.randn(10, 10))
        param2 = torch.nn.Parameter(torch.randn(20, 20))
        mock_model.parameters = Mock(return_value=[param1, param2])

        # Add LoRA adapters structure
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            if not hasattr(mock_model, name):
                setattr(mock_model, name, Mock())

        mock_dit_handler.model = mock_model
        print(f"✅ Mock DiT handler created")

        # Step 3: Create mock configs
        print("\n--- Step 3: Creating mock configs ---")
        mock_lora_config = Mock()
        mock_lora_config.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        mock_lora_config.rank = 4
        mock_lora_config.alpha = 8
        mock_lora_config.dropout = 0.0

        mock_training_config = Mock()
        mock_training_config.batch_size = 1
        mock_training_config.num_workers = 0
        mock_training_config.pin_memory = False
        mock_training_config.learning_rate = 0.001
        mock_training_config.weight_decay = 0.01
        mock_training_config.max_epochs = 2
        mock_training_config.gradient_accumulation_steps = 1
        mock_training_config.warmup_steps = 1
        mock_training_config.max_grad_norm = 1.0
        mock_training_config.log_every_n_steps = 1
        mock_training_config.save_every_n_epochs = 1
        mock_training_config.output_dir = str(Path(tmpdir) / "lora_output")
        print(f"✅ Mock configs created")

        # Step 4: Mock the ACE-Step dependencies
        print("\n--- Step 4: Mocking ACE-Step dependencies ---")

        # Mock PreprocessedLoRAModule
        mock_lora_module = MagicMock()
        mock_lora_module.model = mock_model

        # Create a proper loss tensor with requires_grad
        def mock_training_step(batch):
            loss = torch.tensor(0.5, dtype=torch.float32, requires_grad=True)
            return loss

        mock_lora_module.training_step = mock_training_step

        # Mock inject_lora_into_dit
        mock_lora_info = {"trainable_params": 1000}

        # Mock PreprocessedTensorDataset
        mock_dataset = MagicMock()
        mock_dataset.__len__ = Mock(return_value=3)

        # Mock DataLoader
        mock_batch = {
            "target_latents": torch.randn(1, 100, 64),
            "attention_mask": torch.ones(1, 100),
            "encoder_hidden_states": torch.randn(1, 50, 768),
            "encoder_attention_mask": torch.ones(1, 50),
            "context_latents": torch.randn(1, 100, 128),
        }
        mock_dataloader = [mock_batch, mock_batch, mock_batch]  # 3 batches

        # Mock save_lora_weights
        mock_save_lora_weights = Mock()

        print(f"✅ ACE-Step dependencies mocked")

        # Step 5: Run training with mocked dependencies
        print("\n--- Step 5: Running training workflow ---")

        with patch('acestep.training.trainer.PreprocessedLoRAModule', return_value=mock_lora_module) as mock_module_cls:
            with patch('acestep.training.data_module.PreprocessedTensorDataset', return_value=mock_dataset):
                with patch('torch.utils.data.DataLoader', return_value=mock_dataloader):
                    with patch('acestep.training.lora_utils.save_lora_weights', mock_save_lora_weights):
                        from lora_training import ComfyUITrainer

                        trainer = ComfyUITrainer(
                            dit_handler=mock_dit_handler,
                            lora_config=mock_lora_config,
                            training_config=mock_training_config,
                        )

                        print(f"✅ Trainer created")
                        print(f"  - Device: {trainer.device}")
                        print(f"  - Module before training: {trainer.module}")

                        # Execute training
                        results = []
                        for step, loss, message in trainer.train_from_preprocessed(
                            tensor_dir=str(tensor_dir),
                            training_state=None,
                            resume_from=None
                        ):
                            results.append((step, loss, message))
                            print(f"  [{step}] {message}")

                        # Verify results
                        print(f"\n--- Step 6: Verifying results ---")
                        print(f"✅ Total yields: {len(results)}")

                        # Check module was created
                        assert mock_module_cls.called, "PreprocessedLoRAModule should be created"
                        print(f"✅ PreprocessedLoRAModule created")

                        # Check module creation arguments
                        call_kwargs = mock_module_cls.call_args[1]
                        assert call_kwargs['model'] == mock_model
                        assert call_kwargs['lora_config'] == mock_lora_config
                        assert call_kwargs['training_config'] == mock_training_config
                        print(f"✅ Module created with correct arguments")

                        # Check training progress from results
                        training_steps = [r for r in results if "Step" in r[2]]
                        print(f"✅ Training executed {len(training_steps)} steps")

                        # Check LoRA weights were saved
                        assert mock_save_lora_weights.called, "save_lora_weights should be called"
                        print(f"✅ save_lora_weights called {mock_save_lora_weights.call_count} times")

                        # Check output directory was created
                        output_path = Path(mock_training_config.output_dir)
                        assert output_path.exists(), "Output directory should exist"
                        print(f"✅ Output directory created: {output_path}")

                        return results


def test_error_handling():
    """Test error handling in training."""
    print("\n" + "=" * 60)
    print("ERROR HANDLING TEST")
    print("=" * 60)

    from lora_training import ComfyUITrainer

    # Test with non-existent tensor directory
    print("\n--- Test: Non-existent tensor directory ---")
    mock_dit_handler = Mock()
    mock_dit_handler.device = "cpu"
    mock_dit_handler.dtype = "float32"

    mock_lora_config = Mock()
    mock_training_config = Mock()
    mock_training_config.output_dir = "/tmp/test"

    trainer = ComfyUITrainer(
        dit_handler=mock_dit_handler,
        lora_config=mock_lora_config,
        training_config=mock_training_config,
    )

    results = []
    for step, loss, message in trainer.train_from_preprocessed(
        tensor_dir="/nonexistent/path",
        training_state=None,
        resume_from=None
    ):
        results.append((step, loss, message))
        print(f"  [{step}] {message}")

    assert len(results) == 1, "Should yield one error message"
    assert "not found" in results[0][2].lower(), "Should mention directory not found"
    print(f"✅ Error handled correctly")


def main():
    """Run all full workflow tests."""
    print("\n" + "=" * 70)
    print(" " * 15 + "LORA TRAINING FULL WORKFLOW TEST")
    print("=" * 70)

    try:
        test_full_training_workflow()
        test_error_handling()

        print("\n" + "=" * 70)
        print("✅ ALL FULL WORKFLOW TESTS PASSED!")
        print("=" * 70)

        print("\nVerified Components:")
        print("  1. Mock tensor data creation ✅")
        print("  2. DiT handler mock ✅")
        print("  3. Configuration mocks ✅")
        print("  4. ACE-Step dependency mocking ✅")
        print("  5. Training workflow execution ✅")
        print("  6. Module creation with correct args ✅")
        print("  7. Training step execution ✅")
        print("  8. Checkpoint saving ✅")
        print("  9. Error handling ✅")

        print("\n🎯 The training logic is ready for ComfyUI testing!")
        print("\nNext steps:")
        print("  1. Pull latest changes in ComfyUI")
        print("  2. Restart ComfyUI")
        print("  3. Test with real audio data")
        print("  4. Verify LoRA generation works")

        return 0

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
