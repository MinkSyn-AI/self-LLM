import shutil
import subprocess

import pytest


def _has_nvidia_smi() -> bool:
    return shutil.which("nvidia-smi") is not None


def _check_nvidia_smi() -> bool:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-L"], stderr=subprocess.STDOUT, timeout=5
        )
        return bool(out.strip())
    except Exception:
        return False


def test_gpu_ready():
    """Check if a GPU is available via PyTorch, otherwise fall back to nvidia-smi.

    Fails the test if neither PyTorch reports a usable CUDA device nor nvidia-smi
    reports any GPUs. TensorFlow checks removed to avoid that dependency.
    """
    # PyTorch
    try:
        import torch

        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            return
    except Exception:
        pass

    # nvidia-smi fallback
    if _has_nvidia_smi() and _check_nvidia_smi():
        return

    pytest.fail(
        "No GPU detected: PyTorch reports no GPU and nvidia-smi not present or returned no devices."
    )


if __name__ == "__main__":
    pytest.main([__file__])
