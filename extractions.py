from typing import Any

class Extractions:
    def __init__(self):
        self._extractions = {}

    def add_extraction(self, name: str, value: Any):
        # TODO What if already present?
        self._extractions[name] = value

    def update(self, extractions: "Extractions"):
        # TODO What if already present?
        self._extractions.update(extractions._extractions)

    def has_extraction(self, name: str) -> bool:
        return name in self._extractions
