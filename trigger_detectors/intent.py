from typing import Any, List, Mapping, Tuple

from observation import Observation, MessageObservation
from trigger_detector import TriggerDetector

class MessageIntentTriggerDetector(TriggerDetector):

    def __init__(self, intent_name: str, trigger_name: str):
        self._intent_name = intent_name
        self._trigger_name = trigger_name
    
    @property
    def trigger_names(self) -> List[str]:
        return [self._trigger_name]
    
    def trigger_probabilities(self, observations: List[Observation], old_extractions: Mapping[str, Any]) -> Tuple[Mapping[str, float], float, Mapping[str, Any]]:
        # Kickoff if we have payment intent in the observations
        for observation in observations:
            if isinstance(observation, MessageObservation):            
                if observation.has_intent(self._intent_name):
                    # Kickoff condition seen
                    return ({self._trigger_name: 1.0}, 0.0, {})
                else:
                    # No kickoff
                    return ({}, 1.0, {})
