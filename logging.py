from typing import Optional


class Logging:
    # Having this information at class level works as long as there are not puppeteers running react() concurrently. It
    # would be nice to have the log state to an object, but the log object would have to be distributed to all objects
    # that want to log, and trigger detectors might be used in more than one log context.
    _lines = []
    _indent_level = 0
    _indent_size = 4
    _stack = []

    @property
    def log(self) -> Optional[str]:
        if Logging._lines:
            return "\n".join(Logging._lines)
        else:
            return None

    def _log(self, line: Optional[str]) -> None:
        if line is not None:
            indent = " " * Logging._indent_level * Logging._indent_size
            Logging._lines.append(indent + line)

    def _log_begin(self, header):
        self._log(header)
        Logging._indent_level += 1
        Logging._stack.append(len(Logging._lines))

    def _log_end(self):
        if Logging._stack.pop() == len(Logging._lines):
            # Remove the header if there was nothing added.
            Logging._lines.pop()
        Logging._indent_level -= 1

    def _clear_log(self) -> None:
        Logging._lines = []
        Logging._stack = []
        Logging._indent_level = 0
