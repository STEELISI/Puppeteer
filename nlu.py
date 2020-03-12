import json
from os import walk
from os.path import basename, join

from snips_nlu import SnipsNLUEngine # type: ignore
from snips_nlu.default_configs import CONFIG_EN # type: ignore
import spacy


class SpacyEngine:
    """Wrapper around a Spacy model."""
    def __init__(self, model: str) -> None:
        self._nlp = spacy.load(model)

    def get_sentences(self, text: str):
        sens = []
        for chunk in self._generate_data_chunks(text):
            sens.extend([s.text for s in self._nlp(chunk).sents])
        return sens

    def nent_extraction(self, text):
        orgs = []
        # Like orgs, but religeous, political and nationality groups
        norgs = []
        # Locs are Spacy's GPE (country, city etc.) and Spacy's LOCs (bodies of water, mountain ranges etc.)
        locs = []
        people = []
        products = []
        money_amounts = []
        # Well known events like WW2
        events = []
        # Quantities - like weight, distance
        quants = []
            
        map_spacy_to_ours = {
            'PERSON': people,
            'NORP' : norgs,
            'FAC' : locs,
            'ORG' : orgs,
            'GPE' : locs,
            'LOC' : locs,
            'PRODUCT' : products,
            'EVENT' : events,
            'PERCENT' : quants,
            'MONEY' : money_amounts,
            'QUANTITY' : quants
        }
    
        # Spacy can choke on large data, so chunk if we have to.
        for chunk in self._generate_data_chunks(text):
            doc = self._nlp(chunk)
            for ent in doc.ents:
                #print(ent.text)
                #print(ent.label_)
                ## For now, punt on something that's all numbers.
                if ent.label_ in map_spacy_to_ours:
                    l = map_spacy_to_ours[ent.label_]
                    l.append(ent.text)
    
        return {'orgs':orgs, 'norgs':norgs, 'locs':locs, 'people':people, 'products':products, 'money':money_amounts, 'events':events, 'quants':quants}

    @staticmethod
    def _generate_data_chunks(data, chunk_size=2000, boundry_chars=['.','!','?','=','*']):
        # Spacy can croak on large data.
        sindex = 0
        eindex = chunk_size
        while(sindex < len(data)):
            if eindex < len(data):
                if eindex < len(data):
                    move_up_start = eindex
                    while(eindex < len(data) and not data[eindex] in boundry_chars and eindex < move_up_start + 200):
                        eindex = eindex + 1
                    if eindex < len(data)-2:
                        if data[eindex] in boundry_chars:
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


class SpacyLoader:
    """Loader for SpacyEngines."""
    # Implements lookup to make sure that we only load one copy of each Spacy
    # model.

    _nlp = dict()

    @classmethod
    def nlp(cls, model='en_core_web_lg'):
        if model not in cls._nlp:
            cls._nlp[model] = SpacyEngine(model)
        return cls._nlp[model]


class SnipsEngine:
    """Wrapper around a Snips engine."""

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
        sens = self._nlp.get_sentences(text)
        for sen in sens:
            results = self._engine.parse(sen)
            intent = results["intent"]["intentName"]
            p = results["intent"]["probability"]
            if intent != None and intent != 'null':
                intents.append((intent, p, sen))
        return sorted(intents, key=lambda tup: tup[1], reverse=True)


class SnipsLoader:
    """Loader for SnipsEngines."""
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


