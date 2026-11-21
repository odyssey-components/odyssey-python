"""Configuration errors safe to display at a process boundary."""


class ConfigurationError(ValueError):
    def __init__(self, category: str, detail: str = ""):
        self.category = category
        super().__init__(f"invalid configuration: {detail or category}")
