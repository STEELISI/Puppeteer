import requests
from typing import List, Mapping, Tuple
from ..observation import Observation, MessageObservation
from ..extractions import Extractions
from ..trigger_detector import TriggerDetector
from urlextract import URLExtract

extractor = URLExtract()

def extract_url(msg: str) -> Extractions:
    extractions = Extractions()
    extract_urls = extractor.find_urls(msg)
    if extract_urls:
        extractions.add_extraction("url", extract_urls)
    return extractions

class URLWebsiteTriggerDetector(TriggerDetector):

    def __init__(self, detector_name="url"):
        self._detector_name = detector_name

    @property
    def trigger_names(self) -> List[str]:
        return ["valid_url", "invalid_url"]

    def trigger_probabilities(self, observations: List[Observation], old_extractions: Extractions) -> Tuple[Mapping[str, float], Extractions]:
        # Assume we have only one observation
        if isinstance(observations[0], MessageObservation):
            extractions = extract_url(observations[0].text) #extracted url
            if extractions.has_extraction("url"):
                return ({"url": 1.0}, extractions)
            else:
                return ({}, extractions)
        else:
            return ({}, Extractions())

