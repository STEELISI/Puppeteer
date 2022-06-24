import abc
from typing import Set


class Observation(abc.ABC):
    """Observation of something that has happened since last time the Puppeteer was run."""
    pass

class IntentObservation(Observation):
    def __init__(self) -> None:
        """Initializes a new IntentObservation.
        """
        self._intents: Set[str] = set()

    def __str__(self) -> str:
        """Returns a string representation of this object."""
        return f"intents: {list(self._intents)}"

    def has_intent(self, intent: str) -> bool:
        """Returns True if the observation has the given intent.

        Args:
            intent: The name of the intent.

        Returns:
            True if the observation has the given intent.
        """
        return intent in self._intents
    def add_intent(self, intent: str) -> None:
        """Add an intent to this observation.

        Args:
            intent: The name of the intent.
        """
        self._intents.add(intent)

class MessageObservation(Observation):
    """A message received since last time the Puppeteer was run.

    The message object contains the message text and a list of intents detected in the message.

    Intents may be set by external analysis before the message is handed to the Puppeteer, and may represent the belief
    that the message expresses some intent. The intent mechanism may also be used more generally, to indicate some fact
    that has been inferred about the message. Intents are typically used by subclasses of TriggerDetector as a part of
    determining whether a trigger has occurred -- this is how they may affect Puppeteer behavior.
    """

    def __init__(self, text: str) -> None:
        """Initializes a new MessageObservation.

        Args:
            text: The message text.
        """
        self._text = text
        self._intents: Set[str] = set()

    def __str__(self) -> str:
        """Returns a string representation of this object."""
        if self._intents:
            return f"text: '{self._text}', intents: {list(self._intents)}"
        else:
            return f"text: '{self._text}',"

    @property
    def text(self) -> str:
        """Returns the message text."""
        return self._text
    
    def has_intent(self, intent: str) -> bool:
        """Returns True if the observation has the given intent.

        Args:
            intent: The name of the intent.

        Returns:
            True if the observation has the given intent.
        """
        return intent in self._intents

    def add_intent(self, intent: str) -> None:
        """Add an intent to this observation.

        Args:
            intent: The name of the intent.
        """
        self._intents.add(intent)
