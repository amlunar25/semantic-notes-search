class FakeLanguageModel:
    def __init__(
        self,
        response: str = "Generated test answer.",
    ) -> None:
        self.response = response
        self.received_prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.received_prompts.append(prompt)
        return self.response
