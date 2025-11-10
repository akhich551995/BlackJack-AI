"""LLM adapter stub

This module used to provide a Gemini-specific adapter but the project has
migrated to OpenAI (OPENAI_API_KEY). The adapter is deprecated and kept as a
no-op placeholder to avoid import errors in older commits.
"""

def deprecated_adapter(*args, **kwargs):
    raise RuntimeError("Gemini adapter removed — use OpenAI (set OPENAI_API_KEY) and ensure CREWAI envs are configured to 'openai'.")
