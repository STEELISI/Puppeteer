import gzip
import os
import json
from snips_nlu import SnipsNLUEngine # type: ignore
from snips_nlu.default_configs import CONFIG_EN# type: ignore

# From this project
from spacy_helpers import generate_data_chunks, spacy_get_sentences


"""
Train a snips engine from json.
"""
def snips_train(json):
    engine = SnipsNLUEngine(config=CONFIG_EN)
    engine.fit(json)
    return engine

"""
Extract data from a training file. Right now, this is just text.
"""
def data_from_file(filename, mode='r', replaceChars=False):
    try:
        with gzip.open(filename, 'rb') as f:
            data = f.read().decode()
    except:
        try:
            with open(filename, mode) as f:
                data = f.read()
                if replaceChars:
                    data = data.replace('\n', ' ').replace("*", ' ').replace('>', ' ').replace('=', '')
        except Exception as e:
            raise(e)
    return data


"""
Given a series of text files, train using filenames as intent names.
"""
def snips_train_from_txt(files):
    jsonDict = {}
    jsonDict["intents"] = {}
    for filename in files:
        skillname = filename.replace('.txt', '').replace('-', '')
        skillname = os.path.basename(skillname)
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

