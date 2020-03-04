import abc


class Observation(abc.ABC):
    # Abstract base class for all types of observations used by the Puppeteer
    # to update its beliefs.
    # Corresponds to InputManager from the v0.1 description.
    pass


class MessageObservation(Observation):
    # An Observation class implementing a message that has been received. Can
    # subclass this for more specific message types with more specific information.
    
    def __init__(self, text: str):
        self._text = text
        self._intents = []
    
    @property
    def text(self) -> str:
        return self._text
    
    def has_intent(self, intent):
        return intent in self._intents

    def add_intent(self, intent):
        return self._intents.append(intent)


