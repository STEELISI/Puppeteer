import re
from urlextract import URLExtract
url_extractor = URLExtract()
import pyap # extract address
import spacy
nlp = spacy.load("en_core_web_lg")
from typing import List, Mapping, Tuple
from ..observation import Observation, MessageObservation, IntentObservation
from ..extractions import Extractions
from ..trigger_detector import TriggerDetector

keywords = {
        "name": ["name"],
        "website": ["website"],
        "phone": ["your number", "phone number", "call back number"],
        "address": ["location", "located"],
        "payment": ["payment"],
        "account": ["account"]
}

payment_methods = [
    "venmo",
    "paypal",
    "zelle",
    "apple pay",
    "google pay",
    "cash app",
    "samsung pay",
    "alipay",
    "wechat pay",
    "paytm",
    "phonepe",
    "amazon pay"
]

class Extractor():
    def __init__(self):
        self._keys = ["email", "phone", "payment", "website", "broad_location", "specific_location"]

    def extract_names(self, text) -> List[str]:
        names = []
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                names.append(ent.text)
        return names

    def extract_emails(self, text) -> List[str]:
        #https://stackoverflow.com/questions/17681670/extract-email-sub-strings-from-large-document
        emails = re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', text)
        return emails

    def extract_phone_numbers(self, text) -> List[str]:
        #https://stackoverflow.com/questions/37393480/python-regex-to-extract-phone-numbers-from-string
        phone_numbers = re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text)
        return phone_numbers

    def extract_payments(self, text) -> List[str]:
        # don't forget to check negation
        methods = []
        for method in payment_methods:
            if method in text:
                methods.append(method)
        return methods

    def extract_websites(self, text) -> List[str]:
        websites = url_extractor.find_urls(text)
        return websites

    def extract_broad_locations(self, text) -> List[str]:
        broad_locations = []
        text_embedding = nlp(text)
        for word in text_embedding.ents: #name entity recognition
            if word.label_ == "GPE":
                broad_locations.append(word.text)
        return broad_locations

    def extract_specific_locations(self, text) -> List[str]:
        specific_locations = pyap.parse(text, country='US')
        return [str(loc) for loc in specific_locations]

    def get_extractions(self, text):
        extractions = Extractions()

        names = self.extract_names(text)
        if names:
            extractions.add_extraction("name", names)

        emails = self.extract_emails(text)
        if emails:
            extractions.add_extraction("email", emails)

        phone_numbers = self.extract_phone_numbers(text)
        if phone_numbers:
            extractions.add_extraction("phone", phone_numbers)

        payments = self.extract_payments(text)
        if payments:
            extractions.add_extraction("payment", payments)

        websites = self.extract_websites(text)
        if websites:
            extractions.add_extraction("website", websites)

        broad_locations = self.extract_broad_locations(text)
        if broad_locations:
            extractions.add_extraction("address", broad_locations)

        specific_locations = self.extract_specific_locations(text)
        if specific_locations:
            extractions.add_extraction("address", specific_locations)

        return extractions

extractor = Extractor()

def check_kickoff(topic, text):
    for word in keywords[topic]:
        if word in text:
            return True
    return False

def check_get(topic, extractions):
    if topic == "website":
        if extractions.has_extraction("website"):
            return "full_info"
        else:
            return "push_back"

    if topic == "payment":
        if extractions.has_extraction("payment"):
            return "full_info"
        else:
            return "push_back"

    if topic == "account":
        if extractions.has_extraction("email") or extractions.has_extraction("phone"):
            return "full_info"
        else:
            return "push_back"

    if topic == "phone":
        if extractions.has_extraction("phone"):
            return "full_info"
        else:
            return "push_back"

    if topic == "address":
        if extractions.has_extraction("specific_location"):
            return "full_info"
        elif extractions.has_extraction("broad_location"):
            return "partial_info"
        else:
            return "push_back"

class KickoffTriggerDetector(TriggerDetector):

    def __init__(self, topic):
        self._topic = topic
        self._detector_name = "kickoff_" + topic

    @property
    def trigger_names(self) -> List[str]:
        return ["kickoff"]

    def trigger_probabilities(self, observations: List[Observation], new_extractions: Extractions) -> Tuple[Mapping[str, float], Extractions]:
        # Assume we have only one observation
        if isinstance(observations[0], IntentObservation):
            return ({"kickoff": 1.0}, Extractions()) if observations[0].has_intent(self._topic) else ({"kickoff": 0.0}, Extractions())
        elif isinstance(observations[0], MessageObservation):
            text = observations[0].text
            if check_kickoff(self._topic, text):
                return ({"kickoff": 1.0}, Extractions())

        return ({"kickoff": 0.0}, Extractions())

class BasicTriggerDetector(TriggerDetector):

    def __init__(self, topic):
        self._topic = topic
        self._detector_name = "get_" + topic

    @property
    def trigger_names(self) -> List[str]:
        return ["full_info", "partial_info", "push_back"]

    def trigger_probabilities(self, observations: List[Observation], new_extractions: Extractions) -> Tuple[Mapping[str, float], Extractions]:
        # Assume we have only one observation
        if isinstance(observations[0], MessageObservation):
            text = observations[0].text
            extractions = extractor.get_extractions(text)
            t = check_get(self._topic, extractions)
            if t == "full_info":
                return ({"full_info": 1.0, "partial_info": 0.0, "push_back": 0.0}, extractions)
            elif t == "partial_info":
                return ({"full_info": 0.0, "partial_info": 1.0, "push_back": 0.0}, extractions)
            elif t == "push_back":
                return ({"full_info": 0.0, "partial_info": 0.0, "push_back": 1.0}, Extractions())

        return ({"full_info": 0.0, "partial_info": 0.0, "push_back": 1.0}, Extractions())

class PushbackStarterTriggerDetector(TriggerDetector):

    def __init__(self, topic):
        self._topic = topic
        self._detector_name = "push_back_" + topic

    @property
    def trigger_names(self) -> List[str]:
        return ["kickoff"]

    def trigger_probabilities(self, observations: List[Observation], new_extractions: Extractions) -> Tuple[Mapping[str, float], Extractions]:
        # Assume we have only one observation
        if isinstance(observations[0], IntentObservation):
            if observations[0].has_intent("push_back"):
                return ({"kickoff": 1.0}, Extractions())

        return ({"kickoff": 0.0}, Extractions())
