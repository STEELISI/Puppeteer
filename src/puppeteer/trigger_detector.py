import abc
import os
from typing import Dict, List, Optional, Tuple

from .extractions import Extractions
from .nlu import SnipsEngine, SpacyEngine
from .observation import MessageObservation, Observation


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
        pass  # pylint: disable=W0107

    @abc.abstractmethod
    # TODO Use class TriggerProbabilities (or similar) as output?
    def trigger_probabilities(
        self, observations: List[Observation], old_extractions: Extractions
    ) -> Tuple[Dict[str, float], float, Extractions]:
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


class SnipsTriggerDetector(TriggerDetector):
    """Class detecting triggers in observations, using Snips.

    A SnipsTriggerDetector uses the Snips NLU library to detect triggers, termed *intents* in Snips. Each trigger
    (intent) is learned based on sets of example sentences provided to the detector in text files. More specifically,
    for each intent, there is a folder with the same name as the intent and this folder contains two text files, each
    file with a number of sentences, each sentence on its own line in the file. One of the files contains positive
    examples, sentences where the trigger is present, and the other file contains negative example sentences. If the
    intent name is "xyz", the file with positive examples should be called "xyz.txt" and the file with negative
    examples should be called "NOTxyz.txt".

    The SnipsTriggerDetector trains one or several Snips engines on the intent data, for use in detecting triggers
    (intents) in observations. This can be done in two different modes. In single-engine mode, there is one engine
    trained on all the intents, and this engine is responsible for all of the trigger detection. In multi-engine mode,
    there is one engine per intent, trained only on the examples for that intent. The mode is chosen when a new trigger
    detector is created, through the constructor's multi_engine parameter.

    See the documentation in TriggerDetector for further information.
    """

    def __init__(
        self, paths: List[str], nlp: SpacyEngine, multi_engine: bool = False
    ) -> None:
        """Initializes a newly created SnipsTriggerEngine.

        Args:
            paths: This is a list of directory paths pointing to the training data. For each directory, the leafs of
                the sub-directory tree should be intent data formatted as outlined in the class documentation above.
                The created detector will detect all intents found under these paths.
            nlp: This is a SpacyEngine, used by the detector to perform internal tasks.
            multi_engine: This flags controls the choice between single-engine and multi-engine mode. The default is
                single-engine.
        """
        self._engines: List[SnipsEngine] = []
        self._trigger_names: List[str] = []
        self._nlp = nlp
        self._paths_list: List[List[str]] = []

        # Prepare creation of our Snips engine or engines.
        if multi_engine:
            self._paths_list = [[p] for p in paths]
        else:
            self._paths_list = [paths]

    def load(self) -> None:
        """Loads the trigger detector.

        See documentation of the corresponding method in TriggerDetector.
        """
        for paths in self._paths_list:
            engine = SnipsEngine.load(paths, self._nlp)
            self._engines.append(engine)
            self._trigger_names.extend(engine.intent_names)

    @property
    def trigger_names(self) -> List[str]:
        """Returns the names of the triggers detected by this trigger detector."""
        return list(self._trigger_names)

    def trigger_probabilities(
        self, observations: List[Observation], old_extractions: Extractions
    ) -> Tuple[Dict[str, float], float, Extractions]:
        """Returns trigger probabilities and extractions, based on observations and previous extractions.

        See documentation of the corresponding method in TriggerDetector.
        """
        texts = []
        for observation in observations:
            if isinstance(observation, MessageObservation):
                texts.append(observation.text)
        text = "\n".join(texts)

        trigger_map: Dict[str, float] = {}
        for engine in self._engines:
            snips_results = engine.detect(text)

            for intent, p, sen in snips_results:  # pylint: disable=W0612
                if "NOT" not in intent:
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
        return trigger_map, non_event_prob, Extractions()


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

    def __init__(self, default_snips_path: Optional[str] = None) -> None:
        """Initializes a newly created TriggerDetectorLoader.

        Args:
            default_snips_path: The default root path used to load SNIPS-based trigger detectors.
        """
        self._default_snips_path = default_snips_path
        self._snips_paths: Dict[str, str] = {}
        self._registered: Dict[str, TriggerDetector] = {}
        self._registered_by_agenda: Dict[str, Dict[str, TriggerDetector]] = {}

    def register_detector(self, detector: TriggerDetector) -> None:
        """Register a trigger detector that may be used by any agenda.

        Args:
            detector: The trigger detector.
        """
        for trigger_name in detector.trigger_names:
            self._registered[trigger_name] = detector

    def register_detector_for_agenda(
        self, agenda_name: str, detector: TriggerDetector
    ) -> None:
        """Register a trigger detector that may only be used by a single agenda.

        Args:
            agenda_name: The name of the agenda.
            detector: The trigger detector.
        """
        if agenda_name not in self._registered_by_agenda:
            self._registered_by_agenda[agenda_name] = {}
        for trigger_name in detector.trigger_names:
            self._registered_by_agenda[agenda_name][trigger_name] = detector

    def register_snips_path_for_agenda(self, agenda_name: str, snips_path: str) -> None:
        """Register a path to SNIPS intents used by a single agenda.

        Args:
            agenda_name: The name of the agenda.
            snips_path: The path to the SNIPS intents.
        """
        self._snips_paths[agenda_name] = snips_path

    def load(
        self, agenda_name: str, trigger_names: List[str], snips_multi_engine: bool = False
    ) -> List[TriggerDetector]:
        """Load trigger detectors for an agenda.

        Args:
            agenda_name: The name of the agenda.
            trigger_names: The names of the triggers to load.
            snips_multi_engine: If true, SnipsTriggerDetectors are loaded in multi-engine mode.

        Return:
            A list containing the loaded trigger detectors.
        """
        detectors = []
        snips_trigger_paths = []
        for trigger_name in trigger_names:
            if (
                agenda_name in self._registered_by_agenda
                and trigger_name in self._registered_by_agenda[agenda_name]
            ):
                detector = self._registered_by_agenda[agenda_name][trigger_name]
                detector.load()
                detectors.append(detector)
            elif trigger_name in self._registered:
                detector = self._registered[trigger_name]
                detector.load()
                detectors.append(detector)
            else:
                # See if this is a standard Snips trigger.
                def lookfor(dirname: str, rootpath: str) -> Optional[str]:
                    for (root, dirs, files) in os.walk(rootpath):  # pylint: disable=W0612
                        if dirname in dirs:
                            return os.path.join(root, dirname)
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
                        raise ValueError(
                            "Could not find detector for trigger: %s" % trigger_name
                        )
        # Get standard Snips trigger detectors.
        if snips_trigger_paths:
            nlp = SpacyEngine.load()
            detector = SnipsTriggerDetector(
                snips_trigger_paths, nlp, multi_engine=snips_multi_engine
            )
            detector.load()
            detectors.append(detector)

        # Return unique detectors
        return list(set(detectors))
