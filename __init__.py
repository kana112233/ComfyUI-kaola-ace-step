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
# Transformers 5.x Compatibility Patch
# Transformers v5 removed 'layer_type_validation' from configuration_utils, 
# which is still required by the remote ACE-Step model code.
# --------------------------------------------------------------------------------
def _patch_transformers_v5():
    try:
        import transformers.configuration_utils
        if not hasattr(transformers.configuration_utils, "layer_type_validation"):
            # print("[ACE_STEP] Patching transformers.configuration_utils.layer_type_validation for v5 compatibility")
            def layer_type_validation(*args, **kwargs):
                return args[0] if args else None
            transformers.configuration_utils.layer_type_validation = layer_type_validation
    except (ImportError, Exception):
        pass

_patch_transformers_v5()
# --------------------------------------------------------------------------------

import torch
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
