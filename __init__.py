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

import torch
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
