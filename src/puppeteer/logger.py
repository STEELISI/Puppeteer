from typing import List, Optional


class Logger:
    """Logger with indentation.

    This is a line-based logger, keeping a list of log line strings provided
    to it through the add() method. It implements indentation levels through
    the begin() and end() methods, where begin() increases and end() decreases
    the indentation level.

    Example (without any interleaved "real" code):

        logger = Logger()
        logger.add("Starting the application.")
        logger.add("Some more text...")
        logger.begin("Sub-task")
        logger.add("Some text concerning the sub-task...")
        logger.add("Some more text concerning the sub-task...")
        logger.end()
        logger.add("Closing the application.")
        print(logger.log)

    Producing the following text output:

        Starting the application.
        Some more text...
        Sub-task
            Some text concerning the sub-task...
            Some more text concerning the sub-task...
        Closing the application.
    """

    INDENT_SIZE = 4
    _instance = None

    def __init__(self) -> None:
        """Initializes a new empty Logger."""
        self._lines: List[str] = []
        self._indent_level = 0
        self._stack: List[int] = []

    def __new__(cls) -> "Logger":
        """Creates a new Logger."""
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    @property
    def log(self) -> Optional[str]:
        """Returns the entire log string, containing all log lines."""
        if self._lines:
            return "\n".join(self._lines)
        else:
            return None

    def add(self, line: Optional[str]) -> None:
        """Add a log line.
        Args:
            line: Line text to add to the log, or None to skip adding anything.
        """
        if line is not None:
            indent = " " * self._indent_level * self.INDENT_SIZE
            self._lines.append(indent + line)

    def begin(self, header: str) -> None:
        """Increase indentation level, with header line.

        Args:
            header: Header line preceding the indented section.
        """
        self.add(header)
        self._indent_level += 1
        self._stack.append(len(self._lines))

    def end(self) -> None:
        """Decrease indentation level."""
        if self._stack.pop() == len(self._lines):
            # Remove the header if there was nothing added.
            self._lines.pop()
        self._indent_level -= 1

    def clear(self) -> None:
        """Reset logger to initial empty state."""
        self._lines = []
        self._stack = []
        self._indent_level = 0
