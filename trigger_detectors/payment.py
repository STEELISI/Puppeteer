from typing import List, Mapping, Tuple
from ..observation import Observation, MessageObservation
from ..extractions import Extractions
from ..trigger_detector import TriggerDetector
import re
import nltk
nltk.download('punkt')
from nltk import word_tokenize
from nltk.util import ngrams

payment_methods = ["venmo", "paypal", "zelle", "apple pay", "google pay", "cash app", "samsung pay", "alipay"]

def extract_account(text) -> Extractions:
    #https://stackoverflow.com/questions/17681670/extract-email-sub-strings-from-large-document
    emails = re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', text)
    #https://stackoverflow.com/questions/37393480/python-regex-to-extract-phone-numbers-from-string
    phone_numbers = re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text)
    # print("emails: {}".format(str(emails)))
    # print("phone_numbers: {}".format(str(phone_numbers)))
    account = emails + phone_numbers
    if account:
        extractions = Extractions()
        extractions.add_extraction("account", account)
        return extractions
    else:
        return Extractions()

def extract_other_payment(text, old_extractions) -> Extractions:
    # First of all, make sure text is in lowercase
    text = text.lower()
    unigrams = word_tokenize(text)
    bigrams = ngrams(unigrams, 2)
    bigrams = [' '.join(bg) for bg in bigrams]
    tokens = unigrams + bigrams
    other_payments = set()
    for t in tokens:
        if t in payment_methods:
            other_payments.add(t)
    if old_extractions.has_extraction("payment"):
        # All extracted payment methods except the current one
        current_payment = old_extractions.extraction("payment")
        if current_payment in other_payments:
            other_payments.remove(current_payment)
    if other_payments:
        # Focus only on the first other payments
        new_payment = list(other_payments)[0]
        extractions = Extractions()
        extractions.add_extraction("payment", new_payment)
        return extractions
    else:
        return Extractions()

class AccountPaymentTriggerDetector(TriggerDetector):

    def __init__(self, detector_name="account"):
        self._detector_name = detector_name

    @property
    def trigger_names(self) -> List[str]:
        return ["account"]

    def trigger_probabilities(self, observations: List[Observation], old_extractions: Extractions) -> Tuple[Mapping[str, float], Extractions]:
        messages = []
        for observation in observations:
            if isinstance(observation, MessageObservation):
                messages.append(observation.text.strip())
        message = " ".join(messages)
        extractions = extract_account(message) #extracted account
        return ({"account": 1.0}, extractions) if extractions.has_extraction("account") else ({}, extractions)
            
class OtherPaymentTriggerDetector(TriggerDetector):

    def __init__(self, detector_name="other_payment"):
        self._detector_name = detector_name

    @property
    def trigger_names(self) -> List[str]:
        return ["no_but_other_payment"]

    def trigger_probabilities(self, observations: List[Observation], old_extractions: Extractions) -> Tuple[Mapping[str, float], Extractions]:
        messages = []
        for observation in observations:
            if isinstance(observation, MessageObservation):
                messages.append(observation.text.strip())
        message = " ".join(messages)
        extractions = extract_other_payment(message, old_extractions) #extracted other payment
        return ({"no_but_other_payment": 1.0}, extractions) if extractions.has_extraction("payment") else ({}, extractions)

'''
class TransitionPaymentTriggerDetector(TriggerDetector):

	def __init__(self, detector_name="transition_account"):
		self._detector_name = detector_name
      
	@property
	def trigger_names(self) -> List[str]:
		return ["yes_payment", "no_payment", "no_but_try_payment", "no_but_other_payment", "signup_success", "signup_fail", "account"]

	def trigger_probabilities(self, observations: List[Observation], old_extractions: Extractions) -> Tuple[Mapping[str, float], Extractions]:
		preprocess_observation(observations, old_extractions)
		trigger_map_out = {}
		extractions = Extractions()

		# For each trigger
		other_payment = None
		account = None
		for trigger in self.trigger_names:
			if trigger == "account":
				# Only check unigram + bigram of the whole message which is the first observation
				trigger_map_out[trigger], account = get_regex_score(observations[0].text) #binary score: {0, 1}
			elif trigger == "no_but_other_payment":
				# Only check unigram + bigram of the whole message which is the first observation
				trigger_map_out[trigger], other_payment = get_ngram_score(observations[0].text) #binary score: {0, 1}
			else:
				trigger_map_out[trigger] = get_entailment_score(premises[trigger], observations)

        # Since multiple triggers may overlap in term of semantic and yield high confidence scores
        # resulting in low probability scores after normalization
        # To avoid this, we only choose the highest confidence score from those triggers whose confidence scores
        # are considered high (> high_score_threshold) and zero out the rest 
        # image a screnario where we have two large scores and then we normalize them to (0, 1) scale then
        # suddenly both normalized scores (which previously are large) become less by large fraction

		# Since multiple triggers may overlap in term of semantic and yield high trigger scores (> high_score_threshold), 
		# we only choose the max score from either one of them and zero out the rest to avoid low trigger probability scores after normalization.
		# image we have two large scores and then we normalize them to (0, 1) scale then
		# suddenly both normalized scores (which previously are large) become less by large fraction
		candidates = []
		max_score = 0
		winner = None
		for trigger in self.trigger_names:
			if trigger_map_out[trigger] > high_score_threshold:
				candidates.append(trigger)
				if trigger_map_out[trigger] > max_score:
					max_score = trigger_map_out[trigger]
					winner = trigger

		# If winner is no_payment, we need PAYMENT extraction
		if winner == "no_payment":
			next_payment = get_payment_method()
			extractions.add_extraction("PAYMENT", next_payment) #If None: then we will disregard its action in irc_pydle.py/mydemo.py
		# If winner is no_but_try_payment, we need SIGNUP_INFO extraction
		elif winner == "no_but_try_payment":
			signup_link = get_signup_link()
			extractions.add_extraction("SIGNUP_INFO", signup_link)
		# If winner is no_but_other_payment, we need PAYMENT extraction
		elif winner == "no_but_other_payment":
			extractions.add_extraction("OTHER_PAYMENT", other_payment)
		# If winner is account, we need account extraction
		elif winner == "account":
			extractions.add_extraction("account", account)
		# If winner is signup_fail, we need PAYMENT extraction
		elif trigger == "signup_fail":
			next_payment = get_payment_method()
			extractions.add_extraction("PAYMENT", next_payment) #If None: then we will disregard its action in irc_pydle.py/mydemo.py

		# Now that we have a winner, we zero out other candidate's scores
		for trigger in candidates:
			if trigger != winner:
				trigger_map_out[trigger] = 0.01

		return (trigger_map_out, extractions)
'''
