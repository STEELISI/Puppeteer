import string
from typing import Any, List, Mapping, Tuple

from agenda import (
    Agenda, Action, DefaultAgendaPolicy, MessageObservation, Observation,
    Puppeteer, SnipsTriggerDetector, State, Trigger, TriggerDetector
)
from ents import nent_extraction
from nlu import SpacyManager
from spacy_helpers import spacy_get_sentences


class MakePaymentKickoffTriggerDetector(TriggerDetector):

    def trigger_probabilities(self, observations: List[Observation], old_extractions: Mapping[str, Any]) -> Tuple[Mapping[str, float], float, Mapping[str, Any]]:
        # Kickoff if we have payment intent in the observations
        for observation in observations:
            if isinstance(observation, MessageObservation):            
                if observation.has_intent("payment"):
                    # Kickoff condition seen
                    return ({"kickoff": 1.0}, 0.0, {})
                else:
                    # No kickoff
                    return ({}, 1.0, {})


def make_puppeteer():
    policy = DefaultAgendaPolicy(reuse=False,
                     max_transitions=5,
                     absolute_accept_thresh=0.6,
                     min_accept_thresh_w_differential=0.2,
                     accept_thresh_differential=0.1,
                     # TODO Convention right now: Have to be sure of kickoff.
                     kickoff_thresh=1.0)
    
    agenda = Agenda("make_payment", policy=policy)

    agenda.add_state(State('start_state', 'The state we start in.'))
    agenda.add_state(State('requested_payment_service', 'They requested a payment service such as paypal or venmo'))
    agenda.add_state(State('requested_credit_debit', 'They requested our credit/debit card information'))
    agenda.add_state(State('requested_acct_info', 'They requested our own bank account information'))
    agenda.add_state(State('gave_acct_info', 'They provided the bank account info that we asked for'))

    agenda.set_start_state('start_state')

    agenda.add_terminus('gave_acct_info')

    agenda.add_transition_trigger(Trigger('req_online_intent'))
    agenda.add_transition_trigger(Trigger('req_card_intent'))
    agenda.add_transition_trigger(Trigger('req_bank_intent'))
    agenda.add_transition_trigger(Trigger('provide_acct_intent'))

    agenda.add_kickoff_trigger(Trigger('kickoff'))

    agenda.add_transition('start_state', 'req_online_intent', 'requested_payment_service')
    agenda.add_transition('start_state', 'req_card_intent', 'requested_credit_debit')
    agenda.add_transition('start_state', 'req_bank_intent', 'requested_acct_info')
    agenda.add_transition('start_state', 'provide_acct_intent', 'gave_acct_info')
    
    agenda.add_transition('requested_payment_service', 'req_online_intent', 'requested_payment_service')
    agenda.add_transition('requested_payment_service', 'req_card_intent', 'requested_credit_debit')
    agenda.add_transition('requested_payment_service', 'req_bank_intent', 'requested_acct_info')
    agenda.add_transition('requested_payment_service', 'provide_acct_intent', 'gave_acct_info')
    
    agenda.add_transition('requested_credit_debit', 'req_card_intent', 'requested_credit_debit')
    agenda.add_transition('requested_credit_debit', 'req_online_intent', 'requested_payment_service')
    agenda.add_transition('requested_credit_debit', 'req_bank_intent', 'requested_acct_info')
    agenda.add_transition('requested_credit_debit', 'provide_acct_intent', 'gave_acct_info')
    
    agenda.add_transition('requested_acct_info', 'req_bank_intent', 'requested_acct_info')
    agenda.add_transition('requested_acct_info', 'req_online_intent', 'requested_payment_service')
    agenda.add_transition('requested_acct_info', 'req_card_intent', 'requested_credit_debit')
    agenda.add_transition('requested_acct_info', 'provide_acct_intent', 'gave_acct_info')
    
    agenda.add_action_for_state(Action('ask_for_bank_acct_a', 'I can send you the money.  What is your routing and bank account number?', True, 2), 'start_state')
    agenda.add_action_for_state(Action('ask_for_bank_acct_b', 'I always prefer to pay by electronic check, so I need your account numbers for that', True, 1), 'start_state')
    agenda.add_action_for_state(Action('ask_for_bank_acct_c', 'If you give me your routing and account numbers would make it really easy for me to send the payment', True, 1), 'start_state')
    agenda.add_action_for_state(Action('deflect_payment_service_a', 'I don\'t know how to use that.', True, 1), 'requested_payment_service')
    agenda.add_action_for_state(Action('deflect_credit_debit_a', 'OK, this might sound weird weird, but I don\'t use credit.  It\'s just a ploy by the lenders to trap you in debt', True, 1), 'requested_credit_debit')
    agenda.add_action_for_state(Action('deflect_credit_debit_b', 'When I pay a bill, I either use cash, check, or e-check', True, 1), 'requested_credit_debit')
    agenda.add_action_for_state(Action('deflect_credit_debit_c', 'I don\'t actually use any cards anymore.  It makes spending too easy.', True, 1), 'requested_credit_debit')
    agenda.add_action_for_state(Action('deflect_acct_info_a', 'You don\'t need my bank info.  I\'ll just send you an electronic check.', True, 1), 'requested_acct_info')
    agenda.add_action_for_state(Action('promise_payment', 'Thanks!  I\'ll send along payment shortly', True, 1), 'gave_acct_info')

    agenda.add_stall_action_for_state(Action('ask_for_bank_acct_d', 'A lot of people don\'t use electronic checks, but I promise it\'s easier', True, 1), 'start_state')
    agenda.add_stall_action_for_state(Action('ask_for_bank_acct_e', 'I pretty much only use e-check on the internet. Sorry', True, 1), 'start_state')

    agenda.add_stall_action_for_state(Action('deflect_payment_service_b', 'Like, I read that those services get hacked all the time.  I can\'t give them my bank info', True, 1), 'requested_payment_service')
    agenda.add_stall_action_for_state(Action('deflect_payment_service_c', 'I just don\'t trust those guys.  Sorry.  Can\'t we just do e-check?', True, 1), 'requested_payment_service')

    agenda.add_stall_action_for_state(Action('deflect_credit_debit_d', 'So I actually got into credit card trouble a few years ago, and Im working through it now.  Part of that was shredding the cards and using the envelope system', True, 1), 'requested_credit_debit')
    agenda.add_stall_action_for_state(Action('deflect_credit_debit_e', 'Credit cards (and debit cards) are just a trap to make spending easier.  Some people do ok with that, but most people don\'t, so I got them out of my life.  It has made saving up money SO much easier', True, 1), 'requested_credit_debit')

    agenda.add_stall_action_for_state(Action('deflect_acct_info_b', 'I only need to know your number to send the money.  It\'s actually pretty straightforward', True, 1), 'requested_acct_info')

    agenda.add_stall_action_for_state(Action('promise_payment', 'Thanks!  I\'ll send along payment shortly', True, 1), 'gave_acct_info')
    
    nlp = SpacyManager.nlp()
    
    d = SnipsTriggerDetector("../turducken/data/training/puppeteer/make_payment", nlp, multi_engine=False)
    agenda.add_transition_trigger_detector(d)
    
    d = MakePaymentKickoffTriggerDetector()
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
        msg.add_intent("payment")
        (actions, extractions) = self._puppeteer.react([msg], self._extractions)

        print("-"*40)

        if extractions:
            print("Extractions:")
            for (key, value) in extractions.items():
                print("    %s: %s" % (key, value))
        else:
            print("No extractions")

        if self._puppeteer._policy_state._current_agenda is None:
            print("No current agenda")
        else:
            print("Current agenda: %s" % self._puppeteer._policy_state._current_agenda.name)

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
    tc.say("Why?")
    tc.say("routing number: 8998 account number: 12321312321")



# "Hello"
# "None of your business"
# "No way"
# "I live in Chicago"

# "routing number: 8998 account number: 12321312321"