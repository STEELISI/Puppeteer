from typing import Any, Dict


class Extractions:
    """Class holding a number of extractions for a conversation.
    
    Extractions are key-value pairs that represent facts about the world that have been extracted by the Puppeteer, or
    some other analysis outside of the Puppeteer. They are kept for the entire conversation, and entries may be added,
    modified, and removed, as the conversation goes along.
    
    New Puppeteer extractions are delegated to the TriggerDetector class and are made based on the observations, for
    each turn.
    
    External extractions may be made based on the same observations that the Puppeteer gets, before or after the call
    to the Puppeteer's react() method each turn, or they may use other sources of information not used by the
    Puppeteer.

    The typical way of handling extractions for a conversation is to have a single object per conversation, collecting
    all extractions that have been made over the whole conversation, as described above. This is not strictly required
    though. Extractions are provided as input to Puppeteer's react() method, which also returns new extractions made by
    its trigger detectors for the current turn. The typical procedure is to have a single Extractions object for the
    conversation, that is given to the react() method each turn, and updated with the new extractions made by react().
    Handling this is the responsibility of the surrounding code, and there is nothing that prevents other ways of
    handling extractions between turns.
    """
    def __init__(self) -> None:
        """Creates an empty set of extractions."""
        self._extractions: Dict[str, Any] = {}

    def __repr__(self) -> str:
        return repr(self._extractions)

    def add_extraction(self, name: str, value: Any) -> None:
        """Adds an extraction.

        Updates the extraction if there was an extraction already set for the name.
        """
        self._extractions[name] = value

    def remove_extraction(self, name: str) -> None:
        """Removes an extraction, if present."""
        if name in self._extractions:
            del self._extractions[name]

    def update(self, extractions: "Extractions") -> None:
        """Updates these extractions

        Uses extraction from argument if there are duplicate names.
        """
        self._extractions.update(extractions._extractions)

    def has_extraction(self, name: str) -> bool:
        """Returns true if this object has an extraction with the given name."""
        return name in self._extractions
