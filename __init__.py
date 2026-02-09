import os
import ctypes
import sys
import glob

# --------------------------------------------------------------------------------
# Critical TLS Fix for ComfyUI + Conda
# The "Inconsistency detected by ld.so" error is caused by OpenMP library conflicts.
# We attempt to force-load libgomp via ctypes before any other library loads it.
# --------------------------------------------------------------------------------
def _force_load_libgomp():
    try:
        # 1. Try generic path
        ctypes.CDLL("libgomp.so.1", mode=ctypes.RTLD_GLOBAL)
    except OSError:
        # 2. Try to find it in the current conda environment (based on sys.executable)
        try:
            conda_prefix = os.environ.get("CONDA_PREFIX")
            if not conda_prefix and "envs" in sys.executable:
                 # Infer from python path if CONDA_PREFIX is missing
                 conda_prefix = sys.executable.split("/bin/python")[0]

            if conda_prefix:
                lib_paths = glob.glob(os.path.join(conda_prefix, "lib", "libgomp.so.1"))
                if lib_paths:
                    ctypes.CDLL(lib_paths[0], mode=ctypes.RTLD_GLOBAL)
        except Exception:
            pass

_force_load_libgomp()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# Patch torchaudio to use soundfile instead of libtorchcodec
# This fixes "Could not load libtorchcodec" errors in both training and data preparation
# --------------------------------------------------------------------------------
def _apply_torchaudio_soundfile_patch():
    """Patch torchaudio.load to use soundfile for compatibility."""
    try:
        import torchaudio
        import soundfile as sf
        import numpy as np

        # Save original
        _original_torchaudio_load = torchaudio.load

        def patched_torchaudio_load(filepath, *args, **kwargs):
            """Load audio using soundfile instead of torchaudio."""
            try:
                # Use soundfile to load audio
                data, sr = sf.read(filepath, dtype='float32')

                # Convert to torch tensor format: [channels, samples]
                if data.ndim == 1:
                    audio = torch.from_numpy(data).unsqueeze(0)
                else:
                    audio = torch.from_numpy(data.T)

                return audio, sr
            except Exception as e:
                print(f"[patched_torchaudio_load] Error loading {filepath}: {e}")
                # Fallback to original torchaudio.load if soundfile fails
                try:
                    return _original_torchaudio_load(filepath, *args, **kwargs)
                except Exception as e2:
                    print(f"[patched_torchaudio_load] Original torchaudio also failed: {e2}")
                    raise

        # Replace torchaudio.load globally
        torchaudio.load = patched_torchaudio_load
        print("[ComfyUI-ACE-Step] Patched torchaudio.load to use soundfile (fixes libtorchcodec errors)")
    except ImportError:
        pass  # soundfile not available, skip patch

_apply_torchaudio_soundfile_patch()
# --------------------------------------------------------------------------------

import torch
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
