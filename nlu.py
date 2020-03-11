import json
from os import walk
from os.path import basename, join

from snips_nlu import SnipsNLUEngine # type: ignore
from snips_nlu.default_configs import CONFIG_EN # type: ignore

from spacy_helpers import generate_data_chunks, spacy_get_sentences, spacy_load


class SpacyLoader:
    # Implements lookup to make sure that we only load one copy of each Spacy
    # model.

    _nlp = dict()

    @classmethod
    def nlp(cls, model='en_core_web_lg'):
        if model not in cls._nlp:
            cls._nlp[model] = spacy_load(model=model)
        return cls._nlp[model]


class SnipsEngine:
    # Wrapper around a Snips engine.
    
    def __init__(self, engine, intent_names, nlp):
        self._engine = engine
        self._intent_names = intent_names
        self._nlp = nlp
    
    @classmethod
    def train(cls, filenames, intent_names, nlp):
        jsonDict = {}
        jsonDict["intents"] = {}
        for filename in filenames:
            skillname = filename.replace('.txt', '').replace('-', '')
            skillname = basename(skillname)
            jsonDict["intents"][skillname] = {}
            jsonDict["intents"][skillname]["utterances"] = []
            try:
                with open(filename, "r") as f:
                    filetxt = f.read()
            except Exception as e:
                raise(e)
            for txt in filetxt.split('\n'):
                if txt.strip() != "":
                    udic = {}
                    udic["data"] = []
                    udic["data"].append({"text": txt})
                    jsonDict["intents"][skillname]["utterances"].append(udic)
        jsonDict["entities"] = {}
        jsonDict["language"] = "en"
    
        engine = SnipsNLUEngine(config=CONFIG_EN)
        engine.fit(json.loads(json.dumps(jsonDict, sort_keys=False)))
        return cls(engine, intent_names, nlp)

    @property
    def intent_names(self):
        return self._intent_names
    
    def detect(self, text: str):
        intents = []
        for chunk in generate_data_chunks(text):
            sens = spacy_get_sentences(chunk, nlp=self._nlp)
            for sen in sens:
                results = self._engine.parse(sen)
                intent = results["intent"]["intentName"]
                p = results["intent"]["probability"]
                if intent != None and intent != 'null':
                    intents.append((intent, p, sen))
        return sorted(intents, key=lambda tup: tup[1], reverse=True)


class SnipsLoader:
    # Implements lookup to make sure that we only train one copy of each Snips
    # engine. An engine is defined by the set of paths for its training
    # directories (used as lookup key).

    _engines = dict()
    
    @classmethod
    def engine(cls, paths, nlp):
        paths = frozenset(paths)
        if paths not in cls._engines:
            filenames = []
            for path in paths:
                for wpath, _, files in walk(path):
                    for filename in files:
                        fullpath = join(wpath, filename)
                        filenames.append(fullpath)
            # The intent name is the name of the leaf folder
            intent_names = [basename(p) for p in paths]
            cls._engines[paths] = SnipsEngine.train(filenames, intent_names, nlp)
        return cls._engines[paths]


