import json
from os import walk
from os.path import basename, join
from typing import Any, Dict, FrozenSet, Generator, List, Tuple, Union

from snips_nlu import SnipsNLUEngine  # type: ignore
from snips_nlu.default_configs import CONFIG_EN  # type: ignore
import spacy


class SpacyNLUEngine:
    """Wrapper around a Spacy model."""

    _engines: Dict[str, "SpacyNLUEngine"] = dict()

    def __init__(self, model: str) -> None:
        """Initializes a new SpacyNLUEngine using the given language model.

        This method is only intended for class-internal use. For external
        code getting access to a SpacyNLUEngine, the load() method should be
        used.

        Args:
            model: Name of the Spacy language model to use.
        """
        self._nlp = spacy.load(model)

    @classmethod
    def load(cls, model: str = 'en_core_web_lg') -> "SpacyNLUEngine":
        """Load a SpacyNLUEngine using the given language model.

        Args:
            model: Name of the Spacy language model to use.

        Returns:
            An engine using the specified language model.
        """
        if model not in cls._engines:
            cls._engines[model] = SpacyNLUEngine(model)
        return cls._engines[model]

    def get_sentences(self, text: str) -> List[str]:
        """Extract sentences from a text.

        Args:
            text: A text string to extract sentences from.

        Returns:
            List of extracted sentences.
        """
        sens = []
        for chunk in self._generate_data_chunks(text):
            sens.extend([s.text for s in self._nlp(chunk).sents])
        return sens

    def nent_extraction(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from a text.

        Args:
            text: A text string to extract named entities from.

        Returns:
            A dictionary of extracted named entities. Keys in the dictionary are
            names of types of named entities ("people", "orgs", etc.), and values
            are list of extracted entity names for the given type.
        """
        orgs: List[str] = []
        # Like orgs, but religious, political and nationality groups
        norgs: List[str] = []
        # Locs are Spacy's GPE (country, city etc.) and Spacy's LOCs (bodies of water, mountain ranges etc.)
        locs: List[str] = []
        people: List[str] = []
        products: List[str] = []
        money_amounts: List[str] = []
        # Well known events like WW2
        events: List[str] = []
        # Quantities - like weight, distance
        quants: List[str] = []

        map_spacy_to_ours = {
            'PERSON': people,
            'NORP': norgs,
            'FAC': locs,
            'ORG': orgs,
            'GPE': locs,
            'LOC': locs,
            'PRODUCT': products,
            'EVENT': events,
            'PERCENT': quants,
            'MONEY': money_amounts,
            'QUANTITY': quants
        }

        # Spacy can choke on large data, so chunk if we have to.
        for chunk in self._generate_data_chunks(text):
            doc = self._nlp(chunk)
            for ent in doc.ents:
                if ent.label_ in map_spacy_to_ours:
                    our_list = map_spacy_to_ours[ent.label_]
                    our_list.append(ent.text)

        return {'orgs': orgs, 'norgs': norgs, 'locs': locs, 'people': people, 'products': products,
                'money': money_amounts, 'events': events, 'quants': quants}

    @staticmethod
    def _generate_data_chunks(data: str, chunk_size: int = 2000) -> Generator[str, None, None]:
        """Divide a text into a sequence of text chunks.

        The method aims to make splits at boundary characters, not splitting
        sentences into different chunks.

        Args:
            data: Text to divide into chunks:
            chunk_size: Target size in characters of each chunk.

        Yields:
            The next chunk.
        """
        # Spacy can croak on large data.
        boundary_chars = ['.', '!', '?', '=', '*']
        sindex = 0
        eindex = chunk_size
        while sindex < len(data):
            if eindex < len(data):
                if eindex < len(data):
                    move_up_start = eindex
                    while eindex < len(data) and not data[eindex] in boundary_chars and eindex < move_up_start + 200:
                        eindex = eindex + 1
                    if eindex < len(data) - 2:
                        if data[eindex] in boundary_chars:
                            eindex = eindex + 1
                if eindex < len(data):
                    d = data[sindex:eindex]
                else:
                    d = data[sindex:]
            else:
                d = data[sindex:]
            sindex = eindex
            eindex = sindex + chunk_size
            yield d


class SnipsEngine:
    """Wrapper around a Snips engine."""

    _engines: Dict[FrozenSet[str], "SnipsEngine"] = dict()

    def __init__(self, engine: SnipsNLUEngine, intent_names: List[str], nlp: SpacyNLUEngine) -> None:
        """Initialize a new SnipsEngine.

        This method is only intended for class-internal use. For external
        code getting access to a SnipsEngine, the load() method should be
        used.

        Args:
            engine: The wrapped Snips engine.
            intent_names: The intent names the engine detects.
            nlp: A SpacyNLUEngine used internally to split text into sentences.
        """
        self._engine = engine
        self._intent_names = intent_names
        self._nlp = nlp

    @classmethod
    def load(cls, path_list: List[str], nlp: SpacyNLUEngine) -> Union["SnipsEngine", None]:
        """Load a SnpisEngine trained on data stored at the given path.

        Refer to the documentation of class SnipsTriggerDetector for details
        of how to organize the training data.

        Args:
            path_list: List of root paths where training data is located.
            nlp: A SpacyNLUEngine used internally to split text into sentences.

        Returns:
            An engine trained on the given data.
        """
        paths = frozenset(path_list)
        if paths not in cls._engines:
            filenames = []
            for path in paths:
                #print(1, path)
                for wpath, _, files in walk(path):
                    #print(2, wpath, files)
                    for filename in files:
                        #print(3, filename)
                        fullpath = join(wpath, filename)
                        filenames.append(fullpath)
            # The intent name is the name of the leaf folder
            intent_names = [basename(p) for p in paths]
            #print(paths, filenames, intent_names)
            engine = cls.train(filenames, intent_names, nlp)
            if engine:
                # if engine is not None: not all training data files are empty
                cls._engines[paths] = engine
        return cls._engines[paths] if paths in cls._engines else None

    @classmethod
    def train(cls, filenames: List[str], intent_names: List[str], nlp: SpacyNLUEngine) -> Union["SnipsEngine", None]:
        """Create and train a SnipsEngine on given data.

        Refer to the documentation of class SnipsTriggerDetector for details
        of how to organize the training data.

        Args:
            filenames: List of paths to files to use as training data.
            intent_names: The intent names the created engine detects.
            nlp: A SpacyNLUEngine used internally by the created engine to split text into sentences.

        Returns:
            An engine trained on the given data.
        """
        #print(filenames, intent_names)
        json_dict: Dict[str, Any] = {}
        json_dict["intents"] = {}
        json_dict["entities"] = {}
        json_dict["language"] = "en"
        for filename in filenames:
            texts = []
            with open(filename, "r") as f:
                filetxt = f.read()
            for txt in filetxt.split('\n'):
                if txt.strip() != "":
                    texts.append(txt.strip())

            # skip if it is an empty text file
            if len(texts) == 0:
                continue
                 
            skillname = filename.replace('.txt', '').replace('-', '')
            skillname = basename(skillname)
            json_dict["intents"][skillname] = {}
            json_dict["intents"][skillname]["utterances"] = []

            for txt in texts:
                udic: Dict[str, List[Dict[str, str]]] = {"data": []}
                udic["data"].append({"text": txt})
                json_dict["intents"][skillname]["utterances"].append(udic)
        #print(json_dict)
        '''
        with open("example.json", "w") as out_file:
            json.dump(json_dict, out_file, indent=4)
        '''
        # if all training files for the intents are empty => no engine
        if len(json_dict["intents"]) == 0:
            return None

        valid_intent_names: List[str] = list(json_dict["intents"].keys())
        json_data = json.loads(json.dumps(json_dict, sort_keys=False))

        engine = SnipsNLUEngine(config=CONFIG_EN)
        engine.fit(json_data)
        return cls(engine, valid_intent_names, nlp)

    @property
    def intent_names(self) -> List[str]:
        """Returns the intent names that this engine detects."""
        return self._intent_names

    def detect(self, text: str) -> List[Tuple[str, float, str]]:
        """Detect intents in the given text.

        Args:
            text: Text to detect intents in.

        Returns:
            A list of detections. Each detection is a tuple consisting of:
            - The name of the intent.
            - The probability of the detection. The exact interpretation of
              this probability is a bit unclear, but it can at least be
              viewed as a reasonable confidence measure.
            - The sentence in which the intent was detected.
        """
        intents = []
        sens = self._nlp.get_sentences(text)
        for sen in sens:
            results = self._engine.parse(sen)
            intent = results["intent"]["intentName"]
            p = results["intent"]["probability"]
            if intent is not None and intent != 'null':
                intents.append((intent, p, sen))
        return sorted(intents, key=lambda tup: tup[1], reverse=True)
