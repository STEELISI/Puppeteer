import abc
import os
from typing import Any, List, Mapping, Tuple

from nlu import SnipsLoader, SpacyLoader
from observation import Observation, MessageObservation


class TriggerDetector(abc.ABC):
    # A trigger detector can take observations and return probabilities that
    # its triggers are "seen" in the observation. A trigger detector has a set
    # of triggers it is looking for.
    # Corresponds partly to TriggerManage from the v0.1 description.

    @abc.abstractproperty
    def trigger_names(self) -> List[str]:
        raise NotImplementedError()

    def load(self):
        # Default is that detectors don't need loading.
        pass
        
    @abc.abstractmethod
    def trigger_probabilities(self, observations: List[Observation], old_extractions: Mapping[str, Any]) -> Tuple[Mapping[str, float], float, Mapping[str, Any]]:
        raise NotImplementedError()


class SnipsTriggerDetector(TriggerDetector):
    # A trigger detector using one or more Snips engines to detect triggers in
    # observations.

    def __init__(self, paths: List[str], nlp, multi_engine=False):
       # TODO Is there a Snips convention for how to store its training data?
        self._engines = []
        self._trigger_names = []
        self._nlp = nlp
        self._paths_list = []
        
       # Preparer creation of our Snips engine or engines.
        if multi_engine:
            self._paths_list = [[p] for p in paths]
        else:
            self._paths_list = [paths]
    
    def load(self):
        for paths in self._paths_list:
            engine = SnipsLoader.engine(paths, self._nlp)
            self._engines.append(SnipsLoader.engine(paths, self._nlp))
            self._trigger_names.extend(engine.intent_names)
    
    @property
    def trigger_names(self) -> List[str]:
        return self._trigger_names

    def trigger_probabilities(self, observations: List[Observation], old_extractions: Mapping[str, Any]) -> Tuple[Mapping[str, float], float, Mapping[str, Any]]:
        texts = []
        for observation in observations:
            if isinstance(observation, MessageObservation):
                texts.append(observation.text)
        text = "\n".join(texts)

        trigger_map = {}
        for engine in self._engines:
            snips_results = engine.detect(text)
                        
            for intent, p, sen in snips_results:
                if 'NOT' not in intent:
                    # trigger_name = intent + '_intent'
                    # if intent + '_intent' not in trigger_map:
                    trigger_name = intent
                    if intent not in trigger_map:
                        trigger_map[trigger_name] = p
                    elif trigger_map[trigger_name] < p:
                        trigger_map[trigger_name] = p
        if trigger_map:
            non_event_prob = 1.0 - max(trigger_map.values())
        else:
            non_event_prob = 1.0
        return (trigger_map, non_event_prob, {})


class TriggerDetectorLoader:
    # TODO Have abstract superclass and let this be DefaultTriggerDetectorLoader?
    def __init__(self, default_snips_path=None):
        self._default_snips_path = default_snips_path
        self._snips_paths = {}
        self._registered = {}
        self._registered_by_agenda = {}
    
    # What to register? Class? Needs to allow parameters -- best to register
    # detectors, but have load method in trigger that loads it when needed.
    def register_detector(self, detector: TriggerDetector):
        for trigger_name in detector.trigger_names:
            self._registered[trigger_name] = detector
    
    def register_detector_for_agenda(self, agenda_name: str, detector: TriggerDetector):
        if agenda_name not in self._registered_by_agenda:
            self._registered_by_agenda[agenda_name] = {}
        for trigger_name in detector.trigger_names:
            self._registered_by_agenda[agenda_name][trigger_name] = detector
    
    def register_snips_path_for_agenda(self, agenda_name: str, snips_path: str):
        self._snips_paths[agenda_name] = snips_path
    
    def load(self, agenda_name: str, trigger_names: List[str], snips_multi_engine=False) -> List[TriggerDetector]:
        detectors = []
        snips_trigger_paths = []
        for trigger_name in trigger_names:
            if (agenda_name in self._registered_by_agenda and
                trigger_name in self._registered_by_agenda[agenda_name]):
                detector = self._registered_by_agenda[agenda_name][trigger_name]
                detector.load()
                detectors.append(detector)
            elif trigger_name in self._registered:
                detector = self._registered[trigger_name]
                detector.load()
                detectors.append(detector)
            else:
                # See if this is a standard Snips trigger.
                def lookfor(trigger_name, path):
                    for (root, dirs, files) in os.walk(path):
                        if trigger_name in dirs:
                            return os.path.join(root, trigger_name)
                    return None
                if agenda_name in self._snips_paths:
                    path = lookfor(trigger_name, self._snips_paths[agenda_name])
                    if path is not None:
                        snips_trigger_paths.append(path)
                elif self._default_snips_path is not None:
                    path = lookfor(trigger_name, self._default_snips_path)
                    if path is not None:
                        snips_trigger_paths.append(path)
                    else:
                        raise ValueError("Could not find detector for trigger: %s" % trigger_name)
        # Get standard Snips trigger detectors.
        if snips_trigger_paths:
            nlp = SpacyLoader.nlp()
            detector = SnipsTriggerDetector(snips_trigger_paths,
                                            nlp,
                                            multi_engine=snips_multi_engine)
            detector.load()
            detectors.append(detector)
        assert detectors
        return detectors




        