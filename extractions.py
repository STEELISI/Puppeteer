from typing import Any, Dict


class Extractions:
    """Class holding a number of extractions."""
    def __init__(self) -> None:
        self._extractions: Dict[str, Any] = {}

    def __repr__(self) -> str:
        return repr(self._extractions)

    def add_extraction(self, name: str, value: Any) -> None:
        # TODO What if already present?
        self._extractions[name] = value

    def update(self, extractions: "Extractions") -> None:
        # TODO What if already present?
        self._extractions.update(extractions._extractions)

    def has_extraction(self, name: str) -> bool:
        return name in self._extractions
