from typing import List, Tuple, Dict
from .observation import MessageObservation
import re
import spacy
from .model import NLI_MODEL, NLI_TOKENIZER

model = NLI_MODEL
tokenizer = NLI_TOKENIZER

class SpacyNLIEngine:
    """ Wrapper around a Spacy model. """

    def __init__(self, model_name="en_core_web_sm"):
        self._model_name = model_name
        self._nlp = spacy.load(model_name)

    def get_sentences(self, msg: str):
        """ Split a massage by punctuations ('.', '!', '?')
        if the split sentence is compound (FANBOYS: For, And, Nor, But, Or, Yet, So)
        then segment it into multiple simple sentences
        a simple sentence is a sequence of word in which 
        its dependency tree contains at least one VERB token and one SUBJ token (nsubj, csubj)

        return: list of split sentences plus their simple sentences
        """
        sentences = set()
        split_msg = re.split("[" + ".!?" + "]+", msg)
        split_msg = [s.strip().lower() for s in split_msg if s.strip()]

        if len(split_msg) > 1: #the given message can be split by punctuations
            sentences.update(split_msg)

        for m in split_msg:
            text = self._nlp(m)
            conj_idx = []
            for token in text:
                if token.text in ["for", "and", "nor", "but", "or", "yet", "so"]: #CONJUNCTIONS: FANBOYS
                    conj_idx.append(token.i)

            # print(conj_idx)

            clauses = []
            if conj_idx:
                # First span
                span = text[:conj_idx[0]]
                cl = ' '.join([t.text for t in span])
                clauses.append(cl)

                for j, idx in enumerate(conj_idx):
                    # Other remaining spans
                    start = idx
                    end = conj_idx[j+1] if j+1 < len(conj_idx) else None
                    span = text[start:end] if j+1 < len(conj_idx) else text[start:]
                    has_subj = False
                    has_verb = False
                    for token in span:
                        if token.dep_ == "nsubj" or token.dep_ == "csubj":
                            has_subj = True
                        if token.pos_ == "AUX" or token.pos_ == "VERB":
                            has_verb = True
                        if has_subj and has_verb:
                            break
                    if has_subj and has_verb:
                        cl = ' '.join([t.text for t in span[:]])
                        clauses.append(cl)
                    else:
                        phrase = ' ' + ' '.join([t.text for t in span])
                        clauses[-1] += phrase

                clauses = [cl.strip() for cl in clauses if cl.strip()]
                # print(clauses)
                sentences.update(clauses)

        return list(sentences)

class NLIEngine:
    
    def __init__(self, paths: Dict[str, str], helper: SpacyNLIEngine = SpacyNLIEngine()):
        self._helper = helper
        
        premises = {}
        for trigger_name, path in paths.items():
            with open(path) as f:
                premise = f.readlines()
                premise = [p.strip() for p in premise if p.strip()]
                # check if this is not an empty file
                if premise:
                    premises[trigger_name] = premise

        self._trigger_names: List[str] = list(premises.keys())
        self._premises: Dict[str, List[str]] = premises
         
    @property
    def trigger_names(self) -> List[str]:
        """Returns the trigger names that this engine detects."""
        return self._trigger_names

    def detect(self, msg: str) -> List[Tuple[str, float, str]]:
        """Detect trigger from the given message.
        Args:
            msg: Message to detect trigger from.
        Returns:
            A list of detections. Each detection is a tuple consisting of:
            - The name of the trigger.
            - The probability of the detection. The exact interpretation of
              this probability is a bit unclear, but it can at least be
              viewed as a reasonable confidence measure.
            - The sentence in which the trigger was detected.
        """

        sentences = [msg]
        sub_sentences = self._helper.get_sentences(msg)
        # sentences: a list of text: original message plus its split(s) and simple sentence(s)
        sentences += sub_sentences

        scores = []
        for trigger_name, premise in self._premises.items():
            # use the highest max entailment score to represent a confidence score
            detected_sent = "" # sentence corresponding to the max entailment score
            max_score = 0 # max entailment score
            for sent in sentences:
                score = 0
                for pm in premise:
                    x = tokenizer.encode(pm, sent, return_tensors='pt')
                    logits = model(x)[0] #(contradiction, neutral, entailment)
                    probs = logits.softmax(dim=1)
                    # retrieve an entailment score (idx=2)
                    entailment_score = probs[0][2].item()
                    if entailment_score > score:
                        score = entailment_score

                if score > max_score:
                    detected_sent = sent
                    max_score = score
             
            scores.append((trigger_name, max_score, detected_sent))

        return sorted(scores, key=lambda x: x[1], reverse=True)
