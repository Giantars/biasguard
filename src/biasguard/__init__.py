"""biasguard — a bias-resistant, event-driven backtesting framework.

The public API is intentionally small and grows one phase at a time. Import
submodules directly (e.g. ``from biasguard.data import load_csv``) rather than
relying on a large top-level namespace.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
