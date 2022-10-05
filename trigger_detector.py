import abc
import os
import errno
from typing import Dict, List, Optional, Tuple

from .extractions import Extractions
from .nli import NLIEngine
from .observation import Observation, MessageObservation

# See if this is a standard NLI trigger.
def lookfor_nli(trigger_name: str, agenda_name: str, rootpath: str) -> Optional[str]:
    filename = trigger_name + ".txt"
    agenda_training_data_path = os.path.join(rootpath, agenda_name)
    if not os.path.isdir(agenda_training_data_path):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), agenda_training_data_path)

    for f in os.listdir(agenda_training_data_path):
        if filename == f:
            return os.path.join(agenda_training_data_path, f)

    return None

class TriggerDetector(abc.ABC):
    """Class detecting triggers in observations.

    A TriggerDetector has the ability to detect one or more triggers in a set of observations. It may also make
    new extractions based on the observations.

    Triggers are defined by name, and the names used by the TriggerDetector need to match the names used by an agenda
    for the TriggerDetector to actually be used for the Agenda's triggers. The Agenda may have more than one trigger
    detector, and each of its triggers should be detected by at least one of its TriggerDetectors.

    This is an abstract class, to be subclassed by concrete trigger detector implementations.

    For more information about triggers, please refer to the documentation of class agenda.Trigger.
    For more information about observations, please refer to the documentation of class observation.Observation.
    For more information about extractions, please refer to the documentation of class extractions.Extractions.
    """

    @property
    @abc.abstractmethod
    def trigger_names(self) -> List[str]:
        """Returns the names of the triggers detected by this trigger detector."""
        raise NotImplementedError()

    def load(self) -> None:
        """Loads the trigger detector.
        
        If there is time- or memory-consuming initialization to do for the TriggerDetector, this initialization should
        be kept out of the constructor, and implemented here instead. This is to allow creation of all available
        types of trigger detectors, e.g., for registering in TriggerDetectorLoader, without having to spend time and/or
        memory on detectors that are never used.
        
        By default, this method does nothing, and it is not required for a subclass to override it, if there is no
        need. For TriggerDetector subclasses the do override this method, it is required that objects are loaded and
        ready to be used before they are added to an Agenda object. For trigger detectors provided to the Agenda
        through the TriggerDetectorLoader, this is done automatically so there is no need to load detectors before
        registering them with the loader.
        """
        pass
        
    @abc.abstractmethod
    # TODO Use class TriggerProbabilities (or similar) as output?
    def trigger_probabilities(self, observations: List[Observation],
                              old_extractions: Extractions) -> Tuple[Dict[str, float], Extractions]:
        """Returns trigger probabilities and extractions, based on observations and previous extractions.
        
        The returned probabilities are measures of how strongly the detector believes that different triggers are
        present in the observations. The previous extractions may also be taken into account in determining the
        probabilities. Probilities should be non-negative, but apart from that, there are no strong constraints on
        the probability values. Specifically, they are not required to sum to 1. The detector is not required to return
        values for all its triggers, it may even return an empty trigger probability dictionary.
        
        Args:
            observations: New observations made this turn.
            old_extractions: Extractions made during previous turns and, possibly, by external analysis in this turn.
        
        Returns:
            A tuple consisting of the following three elements:
            - A dictionary with trigger names as keys and corresponding probabilities as values.
            - The probability that no trigger was present.
            - New extractions made by the detector.
        """
        raise NotImplementedError()

class NLITriggerDetector:
    """Class detecting triggers in observations, using NLI.
    """

    def __init__(self, paths: Dict[str, str]) -> None:
        """Initializes a newly created NLITriggerEngine.
        
        Args:
            paths: This is a list of file paths pointing to the NLI premise data
        """
        self._engine: NLIEngine = NLIEngine(paths)
        self._trigger_names: List[str] = self._engine.trigger_names
        
    @property
    def trigger_names(self) -> List[str]:
        """Returns the names of the triggers detected by this trigger detector."""
        return list(self._trigger_names)

    def trigger_probabilities(self, observations: List[Observation],
                              old_extractions: Extractions) -> Tuple[Dict[str, float], Extractions]:
        """Returns trigger probabilities and extractions, based on observations and previous extractions.
        """
        messages = []
        for observation in observations:
            if isinstance(observation, MessageObservation):
                m = observation.text.strip()
                if m:
                    if m[-1] not in ['.', '?', '!']:
                        m += '.' # append a period
                    messages.append(m)
        message = " ".join(messages).lower() # make sure all text is lowercase

        trigger_map: Dict[str, float] = {}
        nli_results: List[Tuple[str, float, str]] = self._engine.detect(message)
        detected_trigger_name, max_score, detected_sent = max(nli_results, key=lambda x: x[1])
        """
        print("NLI detector: {}, detected_trigger_name: {}, max score: {}, detected_sent: {}". \
                format(self.trigger_names, detected_trigger_name, max_score, detected_sent))
        """

        # We only interested in the trigger with the highest score
        # Thus, we zero out other trigger's scores
        for trigger_name, score, _ in nli_results:
            if trigger_name == detected_trigger_name:
                trigger_map[trigger_name] = max_score
            else:
                trigger_map[trigger_name] = 0

        return trigger_map, Extractions()

class TriggerDetectorLoader:
    """Class loading trigger detectors from disk.
    
    This is a utility class that can be used to map trigger names to actual trigger detectors. It is specifically used
    by the Agenda.load() method to connect trigger detectors when agendas are loaded from file. This takes place in two
    steps:
    - First, trigger detectors are registered with the loader, telling the loader what detectors are available. The
      constructor and three register_xxx() methods are used for this.
    - Second, the load() method is used to find and load detectors for a given agenda with given trigger names.
    
    There are four different ways that we can tell the loader about a detector in the first step.
    - We can specify that a detector should be used by a certain agenda, using the register_detector_for_agenda()
      method. Note that trigger names still need to match, i.e., the agenda will only use a detector for one of its
      triggers if the detector does actually detect a trigger with that name, as defined by the trigger_names() method
      in TriggerDetector.
    - We can specify that a detector may be used by any agenda, using the register_detector() method.
    - We can specify that a certain agenda may use a SnipsTriggerDetector using intents under a certain path, using the
      register_snips_path_for_agenda() method.
    - We can specify that any agenda may use a SnipsTriggerDetector using intents under a certain path, using the
      default_snips_path keyword argument to the constructor.
    
    When looking for trigger detectors for a given agenda, in the load() method, we are preferring detectors that were
    registered using an earlier of the four methods listed above, over one registered using a later method. E.g., if
    both a detector registered using register_detector_for_agenda and one registered using default_snips_path match one
    of the agenda's triggers, the one registered with register_detector_for_agenda will be chosen.
    
    Trigger detectors often need to load and train on data on disk before they can actually be used for trigger
    detection. The load() method makes sure that this takes place, by calling the load() method on any TriggerDetector
    object that it returns.
    """
    def __init__(self, default_nli_path: str) -> None:
        """Initializes a newly created TriggerDetectorLoader.

        Args:
            default_snips_path: The default root path used to load SNIPS-based trigger detectors.
        """
        self._default_nli_path = default_nli_path
        self._nli_paths: Dict[str, str] = {}
        self._registered: Dict[str, TriggerDetector] = {}
        self._registered_by_agenda: Dict[str, Dict[str, TriggerDetector]] = {}
    
    def register_detector(self, detector: TriggerDetector) -> None:
        """Register a trigger detector that may be used by any agenda.

        Args:
            detector: The trigger detector.
        """
        for trigger_name in detector.trigger_names:
            #print('TDL: trigger_name: {}'.format(trigger_name))
            self._registered[trigger_name] = detector
    
    def register_detector_for_agenda(self, agenda_name: str, detector: TriggerDetector) -> None:
        """Register a trigger detector that may only be used by a single agenda.

        Args:
            agenda_name: The name of the agenda.
            detector: The trigger detector.
        """
        if agenda_name not in self._registered_by_agenda:
            self._registered_by_agenda[agenda_name] = {}
        for trigger_name in detector.trigger_names:
            self._registered_by_agenda[agenda_name][trigger_name] = detector
    
    def load(self, agenda_name: str,
             trigger_names: List[str]) -> List[TriggerDetector]:
        """Load trigger detectors for an agenda.

        Args:
            agenda_name: The name of the agenda.
            trigger_names: The names of the triggers to load.
            snips_multi_engine: If true, SnipsTriggerDetectors are loaded in multi-engine mode.

        Return:
            A list containing the loaded trigger detectors.
        """
        detectors = []
        nli_trigger_paths: Dict[str, str] = {} #{trigger_name: path}
        for trigger_name in trigger_names:
            #print(trigger_name)
            if (agenda_name in self._registered_by_agenda and
                    trigger_name in self._registered_by_agenda[agenda_name]):
                print("1) CUSTOM Trigger: {}, is registered for agenda, {}.".format(trigger_name, agenda_name))
                detector = self._registered_by_agenda[agenda_name][trigger_name]
                #detector.load()
                detectors.append(detector)
            elif trigger_name in self._registered:
                print("2) CUSTOM Trigger: {}, is registered for any agendas.".format(trigger_name))
                detector = self._registered[trigger_name]
                #detector.load()
                detectors.append(detector)
            else:
                # NLI standard detector
                path = lookfor_nli(trigger_name, agenda_name, self._default_nli_path)
                if path is not None:
                    nli_trigger_paths[trigger_name] = path
                else:
                    raise ValueError("Could not find any detector for trigger: %s" % trigger_name)

        # Get standard NLI trigger detectors.
        #print(nli_trigger_paths)
        if nli_trigger_paths:
            detector = NLITriggerDetector(nli_trigger_paths)
            if detector.trigger_names:
                # This detector is responsible to detect some triggers
                print("3) NLI Trigger(s): {}".format(detector.trigger_names))
                detectors.append(detector)

        # Return unique detectors
        return list(set(detectors))
