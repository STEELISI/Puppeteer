import os
import json
import logging
from snips_nlu import SnipsNLUEngine # type: ignore
from snips_nlu.default_configs import CONFIG_EN# type: ignore

_LOGGER = logging.getLogger(__name__)

# From this project
from spacy_helpers import (
    generate_data_chunks,
    spacy_get_sentences,
    spacy_load,
)

from snips_helpers import (
    snips_intent,
)


"""
Extract several flavors of Named Entities.
"""
def nent_extraction(text, nlp):
    if nlp == None:
        nlp = spacy_load()

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
    for chunk in generate_data_chunks(text):
        doc = nlp(chunk)
        for ent in doc.ents:
            #print(ent.text)
            #print(ent.label_)
            ## For now, punt on something that's all numbers.
            if ent.label_ in map_spacy_to_ours:
                l = map_spacy_to_ours[ent.label_]
                l.append(ent.text)

    return {'orgs':orgs, 'norgs':norgs, 'locs':locs, 'people':people, 'products':products, 'money':money_amounts, 'events':events, 'quants':quants}
    
def from_nent_extraction(text, nlp, from_engine):
    ents = []
    from_candidate_orgs = []
    from_candidate_people = []
    
    _LOGGER.info("Looking for from candidates in text.")
    
    for chunk in generate_data_chunks(text):
        sens = spacy_get_sentences(chunk, nlp=nlp)
        for sen in sens:
            doc = nlp(sen)
            sen_ents = []
            for ent in doc.ents:
                if ent.label_ != 'ORG':
                    sen = sen.replace(ent.text, ent.label_)
                    _LOGGER.info("New sentence is: %s" % sen)
                else:
                    sen = sen.replace(ent.text, 'corporation')
                sen_ents.append((ent.text, ent.label_))
                ents.append(ent.text)
                _LOGGER.info("TESTING for from ents: %s" % sen)
            intent, p = snips_intent(sen, from_engine)                
            if len(sen_ents) > 0 and intent != None and 'NOT' not in intent and p > .67:
                _LOGGER.info("FOUND FROM CLUE: %s" % sen)
                for t, label in sen_ents:
                    if label == 'PERSON':
                        from_candidate_people.append(t)
                    if label == 'ORG':
                        from_candidate_orgs.append(t)
    return(from_candidate_orgs, from_candidate_people)        
