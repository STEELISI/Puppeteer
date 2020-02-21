from files import filenames_from_dirname
from snips_helpers import snips_over_text_body, snips_train_from_txt
from spacy_helpers import spacy_load

class SpacyManager:
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
    
    def __init__(self, engine, nlp):
        self._engine = engine
        self._nlp = nlp
    
    def detect(self, text: str):
        return snips_over_text_body(text, self._nlp, self._engine) 


class SnipsManager:
    # Implements lookup to make sure that we only train one copy of each Snips
    # engine. An engine is defined by the common root path of its training
    # files (used as lookup key).

    _engines = dict()
    
    @classmethod
    def engine(cls, path, nlp):
        if path not in cls._engines:
            filenames = filenames_from_dirname(path)
            snips_engine = snips_train_from_txt(filenames)
            cls._engines[path] = SnipsEngine(snips_engine, nlp)
        return cls._engines[path]


