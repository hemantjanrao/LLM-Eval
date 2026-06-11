"""Compatibility helpers for fast-moving evaluation dependencies."""

from __future__ import annotations

import importlib.util
import sys
import types


class MissingOptionalIntegration:
    """Placeholder that explains when an optional provider integration is unavailable.

    Some framework releases import optional integrations at module import time. This
    class gives those imports a harmless symbol while still failing clearly if code
    actually tries to instantiate the unavailable provider.
    """

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        """Explain that the optional integration is intentionally not configured."""

        raise ImportError(
            "This optional integration is not installed in the learning environment."
        )


def install_ragas_vertexai_import_shim() -> None:
    """Install a temporary shim for Ragas 0.4 VertexAI import compatibility.

    Ragas 0.4 imports `langchain_community.chat_models.vertexai.ChatVertexAI` at
    import time, while modern LangChain moved that integration elsewhere. The lab
    examples use OpenAI-backed evaluators, so this shim only prevents the optional
    VertexAI import from breaking unrelated Ragas usage.
    """

    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return
    try:
        if importlib.util.find_spec(module_name) is not None:
            return
    except (ModuleNotFoundError, ValueError):
        return

    module = types.ModuleType(module_name)
    module.ChatVertexAI = MissingOptionalIntegration  # type: ignore[attr-defined]
    sys.modules[module_name] = module
