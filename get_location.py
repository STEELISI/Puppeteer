import string
from typing import Any, List, Mapping, Tuple

from agenda import (
    Agenda, Action, DefaultAgendaPolicy, MessageObservation, Observation,
    Puppeteer, SnipsTriggerDetector, State, Trigger, TriggerDetector
)
from ents import nent_extraction
from nlu import SpacyManager
from spacy_helpers import spacy_get_sentences


class FromLocationTransitionTriggerDetector(SnipsTriggerDetector):

    def __init__(self, engine_path: str, cities_path: str, nlp):
        super(FromLocationTransitionTriggerDetector, self).__init__(engine_path, nlp, False)
        self._cities_path = cities_path
        self._nlp = nlp

    def trigger_probabilities(self, observations: List[Observation], old_extractions: Mapping[str, Any]) -> Tuple[Mapping[str, float], float, Mapping[str, Any]]:
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

        dict_of_ents = nent_extraction(text, self._nlp)
        locs_list = dict_of_ents['locs']
 
        trigger_map = {}
        extractions = {}

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
                    not_intent_ents = nent_extraction(sen, self._nlp)
                    not_intent_locs = not_intent_ents['locs']

        # Remove locations mentioned in cases where it's not a 'I live ____'
        # statement.
        locs_list = [x for x in locs_list if x not in not_intent_locs]

        # Add any sentences where the single sentence is actually a location.
        for sen in spacy_get_sentences(text, nlp=self._nlp):
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
                extractions["city"] = loc
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


class FromLocationKickoffTriggerDetector(TriggerDetector):

  def trigger_probabilities(self, observations: List[Observation], old_extractions: Mapping[str, Any]) -> Tuple[Mapping[str, float], float, Mapping[str, Any]]:
    # Kickoff if we have a name but not the city of the person.
    
    # If we already have a location, skip starting this.
    # TODO Assuming that this covers all attribution sources.
    if "city" in old_extractions:
      return ({}, 1.0, {})
      
    # If we don't have a name to go with this convo, skip.
    if (not "first_name" in old_extractions) or (not "last_name" in old_extractions):
      return ({}, 1.0, {})

    # Kickoff condition seen
    return ({"kickoff": 1.0}, 0.0, {})



def make_puppeteer():
    policy = DefaultAgendaPolicy(reuse=False,
                     max_transitions=5,
                     absolute_accept_thresh=0.6,
                     min_accept_thresh_w_differential=0.2,
                     accept_thresh_differential=0.1,
                     # TODO Convention right now: Have to be sure of kickoff.
                     kickoff_thresh=1.0)
    
    agenda = Agenda("get_location", policy=policy)
    agenda.add_state(State('start_state', 'The state we start in.'))
    agenda.add_state(State('got_specific_loc', 'They gave us a specific enough location we can end.'))
    agenda.add_state(State('need_more_specifics', 'They gave us a location, but we want them to be more specific'))
    agenda.add_state(State('push_back', 'They are objecting to telling us.'))
    agenda.add_state(State('give_up', 'We asked enough - we are going to give up.'))
    agenda.set_start_state('start_state')
    agenda.add_terminus('got_specific_loc')
    agenda.add_terminus('give_up')
    agenda.add_transition_trigger(Trigger('push_back_intent'))
    agenda.add_transition_trigger(Trigger('broad_loc'))
    agenda.add_transition_trigger(Trigger('specific_loc'))
    agenda.add_transition_trigger(Trigger('why_intent'))
    agenda.add_kickoff_trigger(Trigger('kickoff'))
    agenda.add_transition('start_state', 'why_intent', 'push_back')
    agenda.add_transition('start_state', 'push_back_intent', 'push_back')
    agenda.add_transition('start_state', 'specific_loc', 'got_specific_loc')
    agenda.add_transition('start_state', 'broad_loc', 'need_more_specifics')
    agenda.add_transition('push_back', 'push_back_intent', 'give_up')
    agenda.add_transition('push_back', 'why_intent', 'push_back')
    agenda.add_transition('push_back', 'specific_loc', 'got_specific_loc')
    agenda.add_transition('push_back', 'broad_loc', 'need_more_specifics')
    agenda.add_transition('need_more_specifics', 'push_back_intent', 'push_back')
    agenda.add_transition('need_more_specifics', 'why_intent', 'push_back')
    agenda.add_transition('need_more_specifics', 'specific_loc', 'got_specific_loc')
    agenda.add_transition('need_more_specifics', 'broad_loc', 'give_up')
    agenda.add_action_for_state(Action('question', 'What location are you based out of?', True, 3), 'start_state')
    agenda.add_action_for_state(Action('ask_specifics', 'Oh, ok - but specifically which city?', True, 3), 'need_more_specifics')
    agenda.add_action_for_state(Action('push_1', 'Just trying to relate on some things. Where are you based out of?',True, 1), 'push_back')
    agenda.add_action_for_state(Action('push_2', 'I\'m not asking for super personal info. Just asking what city you\'re out of...', True, 1), 'push_back')
    agenda.add_action_for_state(Action('done', 'I hope you like it there!', True, 1), 'got_specific_loc')
    agenda.add_stall_action_for_state(Action('push_3', 'You haven\'t told me where you are located yet?', True, 3), 'start_state')
    agenda.add_stall_action_for_state(Action('ask_specifics', 'Where specifically are you?', True, 2), 'need_more_specifics')
    agenda.add_stall_action_for_state(Action('push_4', 'Okay, but it would be nice to note in my file.', True, 1), 'push_back')
    agenda.add_stall_action_for_state(Action('push_5', 'So I forgot, where are you based out of?', True, 2), 'push_back')
    
    nlp = SpacyManager.nlp()
    
    d = SnipsTriggerDetector("../turducken/data/training/puppeteer/get_location", nlp, multi_engine=False)
    agenda.add_transition_trigger_detector(d)
    d = FromLocationTransitionTriggerDetector("../turducken/data/training/puppeteer/get_location/i_live",
                                              '../turducken/data/dictionaries/cities.txt',
                                                  nlp)
    agenda.add_transition_trigger_detector(d)
    
    d = FromLocationKickoffTriggerDetector()
    agenda.add_kickoff_trigger_detector(d)
    
    return Puppeteer([agenda])


class TestConversation:
    def __init__(self):
        self._puppeteer = make_puppeteer()
        self._extractions = {"first_name": "Mr", "last_name": "X"}
    
    def say(self, text):

        print("-"*40)
        print("You said: %s" % text)

        msg = MessageObservation(text)
        (actions, extractions) = self._puppeteer.react([msg], self._extractions)

        print("-"*40)

        if extractions:
            print("Extractions:")
            for (key, value) in extractions.items():
                print("    %s: %s" % (key, value))
        else:
            print("No extractions")

        print("Agenda state probabilities")
        for (agenda_name, belief) in self._puppeteer._beliefs.items():
            # TODO Hacky access to state probabilities.
            tpm = belief._transition_probability_map
            print("    %s:" % agenda_name)
            for (state_name, p) in tpm.items():
                print("        %s: %.3f" % (state_name, p))
        
        if actions:
            print("Actions:")
            for a in actions:
                print("    %s" % a)
        else:
            print("No actions")

        return (actions, extractions)

if __name__ == "__main__":
    tc = TestConversation()
    tc.say("Hello")
    #tc.say("None of your business")
    tc.say("Why?")
    tc.say("Nice weather today")
    #tc.say("I live in Chicago")


    # tc.say("I live in Chicago")
    # tc.say("Very nice, thank you!")



"Hello"
"None of your business"
"No way"
"I live in Chicago"

