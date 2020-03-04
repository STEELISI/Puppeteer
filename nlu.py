import os

from files import filenames_from_dirname
from snips_helpers import snips_over_text_body, snips_train_from_txt
from spacy_helpers import spacy_load


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

    @property
    def intent_names(self):
        return self._intent_names
    
    def detect(self, text: str):
        return snips_over_text_body(text, self._nlp, self._engine) 


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
                filenames.extend(filenames_from_dirname(path))
            print(filenames)
            snips_engine = snips_train_from_txt(filenames)
            # The intent name is the name of the leaf folder
            intent_names = [os.path.basename(p) for p in paths]
            cls._engines[paths] = SnipsEngine(snips_engine, intent_names, nlp)
        return cls._engines[paths]


