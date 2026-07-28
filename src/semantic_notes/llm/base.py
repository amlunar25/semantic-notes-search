from typing import Protocol


class LanguageModelError(RuntimeError):
    """
    Raised when language-model generation fails.
    """


class LanguageModel(Protocol):
    def generate(self, prompt: str) -> str:
        """
        Generate text from a completed prompt.
        """
        ...
