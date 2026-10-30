"""aibridge-core: the aibridge-exec provider layer, rewritten in Rust.

``aibridge_core.providers`` has the same API as ``aibridge_exec.providers``.
"""
from ._native import __version__
from .errors import ProviderError

__all__ = ["ProviderError", "__version__"]
