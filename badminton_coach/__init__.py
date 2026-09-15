"""badminton-coach: measure a badminton stroke from phone video, let Claude judge it.

Importing this package pre-loads CUDA/cuDNN shared libraries from the installed `torch` wheel before
`onnxruntime` touches the GPU. onnxruntime-gpu's CUDA execution provider links against cuDNN 9 at
runtime; on a system without a matching system-wide cuDNN, the libraries bundled in the `torch` cu12
wheel satisfy that dependency, but only if `torch` is imported (or `onnxruntime.preload_dlls()` is
called) before the first CUDA onnxruntime session is created. See README's GPU section.
"""

from __future__ import annotations

__version__ = "0.1.0"


def _preload_cuda_libs() -> None:
    """Best-effort: import torch (loads its bundled CUDA/cuDNN .so files) before onnxruntime needs them.

    Safe no-op if torch isn't installed or has no CUDA build; never raises.
    """
    try:
        import torch  # noqa: F401
    except Exception:
        pass

    try:
        import onnxruntime as ort

        preload = getattr(ort, "preload_dlls", None)
        if callable(preload):
            preload()
    except Exception:
        pass


_preload_cuda_libs()
