from typing import Any

# Best-effort import of the HumanMessage class used by langchain_core
try:
    from langchain_core.messages.human import HumanMessage
except Exception:
    HumanMessage = None


class GeminiLLMAdapter:
    """Wrap a ChatGoogleGenerativeAI instance to accept plain strings/lists
    and convert them into the message-object shape expected by the
    installed langchain_core wrapper.

    This adapter forwards .generate(...) calls to the underlying LLM after
    normalizing inputs. It leaves the original response object untouched.
    """

    def __init__(self, llm: Any):
        self._llm = llm

    def _normalize(self, inp: Any) -> Any:
        # If caller sent a plain string, wrap into [[HumanMessage(content=...)]]
        if isinstance(inp, str):
            if HumanMessage is not None:
                return [[HumanMessage(content=inp)]]
            return [inp]

        # If caller sent a list of strings, convert each to its own message-list
        if isinstance(inp, list):
            if len(inp) == 0:
                return inp
            if isinstance(inp[0], str):
                if HumanMessage is not None:
                    return [[HumanMessage(content=s)] for s in inp]
                return inp
            # assume it's already the right message-object structure
            return inp

        # Fallback: return as-is
        return inp

    def generate(self, messages: Any, **kwargs) -> Any:
        msgs = self._normalize(messages)
        return self._llm.generate(msgs, **kwargs)

    def __call__(self, *args, **kwargs):
        # Support calling the adapter with a single string
        if args and isinstance(args[0], str):
            return self.generate(args[0])
        # Fallback to calling underlying llm if it's callable
        if callable(self._llm):
            return self._llm(*args, **kwargs)
        # Otherwise try to route to generate
        if args:
            return self.generate(args[0])
        return self.generate(kwargs.get('messages') or kwargs.get('input'))
