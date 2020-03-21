import abc
from typing import List


class Observation(abc.ABC):
    """Observation of something that has happened since last time the Puppeteer was run."""
    pass


class MessageObservation(Observation):
    """A message received since last time the Puppeteer was run.

    The message object contains the message text and a list of intents detected in the message.

    Intents may be set by external analysis before the message is handed to the Puppeteer, and may represent the belief
    that the message expresses some intent. The intent mechanism may also be used more generally, to indicate some fact
    that has been inferred about the message. Intents are typically used by subclasses of TriggerDetector as a part of
    determining whether a trigger has occurred -- this is how they may affect Puppeteer behavior.
    """

    def __init__(self, text: str) -> None:
        self._text = text
        self._intents: List[str] = []
    
    @property
    def text(self) -> str:
        return self._text
    
    def has_intent(self, intent: str) -> bool:
        return intent in self._intents

    def add_intent(self, intent: str) -> None:
        return self._intents.append(intent)
