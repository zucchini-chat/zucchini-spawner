"""Zucchini platform plugin for Hermes Agent.

Directory-based plugin discovery (hermes_cli/plugins.py:1167-1272) imports
this package and calls ``register(ctx)``. We re-export from ``adapter.py``
to keep the entry point shallow.
"""

from .adapter import register

__all__ = ["register"]
