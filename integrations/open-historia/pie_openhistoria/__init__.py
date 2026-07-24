"""pie-openhistoria: an OpenAI-compatible HTTP adapter that serves Open Historia
(and any OpenAI `/chat/completions` client) from a Pie-hosted model.

See the package README for architecture, setup, and roadmap.

The server/backend modules pull in aiohttp and pie_client; they are imported
lazily (PEP 562) so `from pie_openhistoria import translate` — the pure mapping
layer — works with no runtime deps installed.
"""

__all__ = ["AdapterConfig", "build_app", "run"]
__version__ = "0.0.1"


def __getattr__(name: str):
    if name in ("AdapterConfig", "build_app", "run"):
        from . import server
        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
