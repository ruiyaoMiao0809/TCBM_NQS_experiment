"""
core/device.py
==============
GPU/CPU device selection utility.

Adapted for TCBM-NQS POC. Keeps interface compatible with
tcbm_cyberphysical_v2/core/device.py (if you merge back).
"""

import torch


def get_device(prefer: str = 'cuda') -> str:
    """
    Return device string 'cuda' or 'cpu' based on availability.

    Parameters
    ----------
    prefer : str, 'cuda' or 'cpu'
        If 'cuda' requested but CUDA is unavailable, falls back to 'cpu'
        with a warning. If 'cpu' requested, returns 'cpu'.

    Returns
    -------
    device : str, 'cuda' or 'cpu'
    """
    if prefer == 'cpu':
        return 'cpu'
    if torch.cuda.is_available():
        return 'cuda'
    import warnings
    warnings.warn("CUDA requested but not available; falling back to CPU.", RuntimeWarning)
    return 'cpu'


def device_info() -> dict:
    """Return a dict of GPU / CUDA info for logging."""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'pytorch_version': torch.__version__,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info['gpu_name'] = props.name
        info['gpu_total_memory_gb'] = props.total_memory / 1e9
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version()
    return info


if __name__ == '__main__':
    print("Device info:")
    for k, v in device_info().items():
        print(f"  {k}: {v}")
    print(f"\nDefault device: {get_device()}")
