import json
from os import walk
from os.path import basename, join

from snips_nlu import SnipsNLUEngine # type: ignore
from snips_nlu.default_configs import CONFIG_EN # type: ignore

# from snips_helpers import snips_over_text_body, snips_train_from_txt
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


"""
Extract data from a training file. Right now, this is just text.
"""
def data_from_file(filename, mode='r', replaceChars=False):
    with open(filename, mode) as f:
        data = f.read()
        if replaceChars:
            data = data.replace('\n', ' ').replace("*", ' ').replace('>', ' ').replace('=', '')
    return data


"""
Train a snips engine from json.
"""
def snips_train(json):
    engine = SnipsNLUEngine(config=CONFIG_EN)
    engine.fit(json)
    return engine


"""
Given a series of text files, train using filenames as intent names.
"""
def snips_train_from_txt(files):
    jsonDict = {}
    jsonDict["intents"] = {}
    for filename in files:
        skillname = filename.replace('.txt', '').replace('-', '')
        skillname = basename(skillname)
        jsonDict["intents"][skillname] = {}
        jsonDict["intents"][skillname]["utterances"] = []
        try:
            filetxt = data_from_file(filename, mode='r', replaceChars=False)
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

    # pprint(jsonDict)
    engine = snips_train(json.loads(json.dumps(jsonDict, sort_keys=False)))
    return engine

"""
Do snips intent extraction over a text body. 
"""
def snips_over_text_body(text, nlp, engine):
    intents = []
    for chunk in generate_data_chunks(text):
        sens = spacy_get_sentences(chunk, nlp=nlp)
        for sen in sens:
            intent, p = snips_intent(sen, engine)
            if intent != None and intent != 'null':
                intents.append((intent, p, sen))
    return sorted(intents, key=lambda tup: tup[1], reverse=True)

"""
Given an engine and some text, see if we can match to a known intent.
"""
def snips_intent(txt, engine):
    results = engine.parse(txt)
    return results["intent"]["intentName"], results["intent"]["probability"]


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
                for wpath, _, files in walk(path):
                    for filename in files:
                        fullpath = join(wpath, filename)
                        filenames.append(fullpath)
                #filenames.extend(filenames_from_dirname(path))
            snips_engine = snips_train_from_txt(filenames)
            # The intent name is the name of the leaf folder
            intent_names = [basename(p) for p in paths]
            cls._engines[paths] = SnipsEngine(snips_engine, intent_names, nlp)
        return cls._engines[paths]


