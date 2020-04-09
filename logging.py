from typing import List, Optional


class Logger:

    INDENT_SIZE = 4
    _instance = None

    # Note: Having this information in a common object works as long as there are not puppeteers running react()
    # concurrently.
    def __init__(self) -> None:
        self._lines: List[str] = []
        self._indent_level = 0
        self._stack: List[int] = []

    def __new__(cls) -> "Logger":
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    @property
    def log(self) -> Optional[str]:
        if self._lines:
            return "\n".join(self._lines)
        else:
            return None

    def add(self, line: Optional[str]) -> None:
        if line is not None:
            indent = " " * self._indent_level * self.INDENT_SIZE
            self._lines.append(indent + line)

    def begin(self, header: str) -> None:
        self.add(header)
        self._indent_level += 1
        self._stack.append(len(self._lines))

    def end(self) -> None:
        if self._stack.pop() == len(self._lines):
            # Remove the header if there was nothing added.
            self._lines.pop()
        self._indent_level -= 1

    def clear(self) -> None:
        self._lines = []
        self._stack = []
        self._indent_level = 0
