from typing import Any, List, Mapping, Tuple

from agenda import (
    Agenda, Action, DefaultAgendaPolicy,
    Puppeteer, State, Trigger
)
from observation import MessageObservation, Observation
from trigger_detector import TriggerDetector, SnipsTriggerDetector
from nlu import SpacyLoader





def create_agenda():
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
    
    nlp = SpacyLoader.nlp()
    paths = ["../turducken/data/training/puppeteer/get_location/push_back",
             "../turducken/data/training/puppeteer/get_location/why"]
    d = SnipsTriggerDetector(paths, nlp, multi_engine=False)
    agenda.add_transition_trigger_detector(d)
    d = FromLocationTransitionTriggerDetector(["../turducken/data/training/puppeteer/get_location/i_live"],
                                              '../turducken/data/dictionaries/cities.txt',
                                               nlp)
    agenda.add_transition_trigger_detector(d)
    
    d = FromLocationKickoffTriggerDetector()
    agenda.add_kickoff_trigger_detector(d)
    
    return agenda

