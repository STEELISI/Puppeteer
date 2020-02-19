import spacy # type: ignore
import re
import spacy.tokenizer  # type: ignore

""" 
Load our spacy model.
"""
def spacy_load(model=None):
    if model == None:
        modelName = 'en_core_web_lg'
    else:
        modelName = model
    try:
        nlp = spacy.load(modelName)
    except Exception as e:   
        raise e
    return nlp
"""
Spacy can croak on large data.
"""
def generate_data_chunks(data, chunk_size=2000, boundry_chars=['.','!','?','=','*']):
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

"""
Take in a nlp object and modify the tokenizer to not cut off at urls.
"""
def spacy_modify_tokenizer_for_urls(nlp):
    prefix_re = re.compile(r'''^[\[\("']''')
    suffix_re = re.compile(r'''[\]\)"']$''')
    infix_re = re.compile(r'''[-~]''')
    simple_url_re = re.compile(r'''^https?://''')
    customTokenizer = spacy.tokenizer.Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                            suffix_search=suffix_re.search,
                            infix_finditer=infix_re.finditer,
                            token_match=simple_url_re.match)
    nlp.tokenizer = customTokenizer

"""
Return sentences.
"""
def spacy_get_sentences(text, nlp=None):
    if nlp == None:
        nlp = spacy_load('en_core_web_lg')
    doc = nlp(text)
    sens = []
    for s in list(doc.sents):
        sens.append(s.text)
    return sens
