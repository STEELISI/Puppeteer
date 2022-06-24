import string
from typing import Any, List, Mapping, Tuple

from puppeteer import (
    Extractions,
    Observation,
    MessageObservation,
    SnipsTriggerDetector,
    TriggerDetector
)


class LocationInMessageTriggerDetector(SnipsTriggerDetector):

    def __init__(self, engine_path: str, cities_path: str, nlp):
        super(LocationInMessageTriggerDetector, self).__init__(engine_path, nlp, False)
        self._cities_path = cities_path
        self._nlp = nlp

    @property
    def trigger_names(self) -> List[str]:
        return ["broad_loc", "specific_loc"]
    
    def trigger_probabilities(self, observations: List[Observation], old_extractions: Extractions) -> Tuple[Mapping[str, float], float, Extractions]:
        """
        Function to determine if we have a location, and if it's specific.
        
        Should, on input, return a dictionary of
        trigger_name : confidence_value_that_we_saw_trigger
        non_event_prob: probability that there was not a event we care about.
        """
        #print("XXXX Hej!")
        texts = []
        for observation in observations:
            if isinstance(observation, MessageObservation):
                texts.append(observation.text)
        text = "\n".join(texts)

        dict_of_ents = self._nlp.nent_extraction(text)
        locs_list = dict_of_ents['locs']
 
        trigger_map = {}
        extractions = Extractions()

        # See if we got a statement about being _from_ somewhere or living
        # somewhere using our custom snips engine.
        not_intent_locs = []
        max_confidence_we_saw_a_loc_statement = 0.0
        for engine in self._engines:
            snips_results = engine.detect(text)
            for intent, p, sen in snips_results:
                # TODO Check if this is the correct intent? Can the engine contain
                # more than intended?
                if 'NOT' not in intent:
                    if p > max_confidence_we_saw_a_loc_statement:
                        max_confidence_we_saw_a_loc_statement = p
                # For NOT intents, we want to ignore these locations.
                elif 'NOT' in intent and p > .65:
                    not_intent_ents = self._nlp.nent_extraction(sen)
                    not_intent_locs = not_intent_ents['locs']

        # Remove locations mentioned in cases where it's not a 'I live ____'
        # statement.
        locs_list = [x for x in locs_list if x not in not_intent_locs]

        # Add any sentences where the single sentence is actually a location.
        for sen in self._nlp.get_sentences(text):
            sen = sen.translate(str.maketrans('', '', string.punctuation))
            for line in open(self._cities_path, 'r'):
                if " ".join(line.split()).strip().lower() == " ".join(sen.split()).strip().lower():
                    locs_list.append(sen)
                    max_confidence_we_saw_a_loc_statement = 1.0

        non_event_prob = 1.0 - max_confidence_we_saw_a_loc_statement

        #_LOGGER.info("Evaluating these locations: %s" % (','.join(locs_list)))

        # We might have a location. Let's see if we got specifics or not.
        is_specific = False
        for loc in locs_list:
            for line in open(self._cities_path, 'r'):
                #if re.search(loc, line, re.IGNORECASE):
                if " ".join(line.split()).strip().lower() == " ".join(loc.split()).strip().lower():
                    is_specific = True
                    break
            if is_specific:
                extractions.add_extraction("city", loc)
                break
            else:
                #_LOGGER.info("Got broad location of %s", loc)
                #print("Got broad location of %s", loc)
                pass
        if len(locs_list) == 0 and non_event_prob > .9:
            #_LOGGER.debug("Got 0 locations to evaluate.")
            #print("Got 0 locations to evaluate.")
            return ({}, 1.0, extractions)
        elif is_specific:
            # If we have a location, but did not register this as a "I live..." 
            # but we only have _1_ location... 
            #_LOGGER.debug("Got specific location of %s", loc)
            if len(locs_list) < 3:
                non_event_prob = .3
                max_confidence_we_saw_a_loc_statement = .7
            trigger_map['specific_loc'] = max_confidence_we_saw_a_loc_statement
            trigger_map['broad_loc'] = 0.0
        else:
            if len(locs_list) < 3:
                non_event_prob = .3
                max_confidence_we_saw_a_loc_statement = .7
            trigger_map['broad_loc'] = max_confidence_we_saw_a_loc_statement
            trigger_map['specific_loc'] = 0.0
        return (trigger_map, non_event_prob, extractions)


class CityInExtractionsTriggerDetector(TriggerDetector):

  def __init__(self, trigger_name="city_in_extractions"):
      self._trigger_name = trigger_name
      
  @property
  def trigger_names(self) -> List[str]:
      return [self._trigger_name]
    
  def trigger_probabilities(self, observations: List[Observation], old_extractions: Extractions) -> Tuple[Mapping[str, float], float, Extractions]:
    # Kickoff if we have a name but not the city of the person.
    
    # If we already have a location, skip starting this.
    # TODO Assuming that this covers all attribution sources.
    if old_extractions.has_extraction("city"):
      return ({}, 1.0, Extractions())
      
    # If we don't have a name to go with this convo, skip.
    if not (old_extractions.has_extraction("first_name") and old_extractions.has_extraction("last_name")):
      return ({}, 1.0, Extractions())

    # Kickoff condition seen
    return ({self._trigger_name: 1.0}, 0.0, Extractions())
