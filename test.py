from typing import List

import numpy as np

from agenda import Agenda, Puppeteer
from extractions import Extractions
from observation import MessageObservation
from trigger_detectors.loader import MyTriggerDetectorLoader

# import get_location
# import make_payment


class TestConversation:
    def __init__(self, agendas: List[Agenda]):
        
        self._puppeteer = Puppeteer(agendas, plot_state=True)
        self._extractions = Extractions()
        self._extractions.add_extraction("first_name", "Mr")
        self._extractions.add_extraction("last_name", "X")
        np.random.seed(0)

    
    def say(self, text):

        print("-"*40)
        print("You said: %s" % text)

        msg = MessageObservation(text)
        msg.add_intent("payment")
        (actions, extractions) = self._puppeteer.react([msg], self._extractions)

        print("-"*40)

        if extractions._extractions:
            print("Extractions:")
            for (key, value) in extractions._extractions.items():
                print("    %s: %s" % (key, value))
        else:
            print("No extractions")

        if self._puppeteer._policy._current_agenda is None:
            print("No current agenda")
        else:
            print("Current agenda: %s" % self._puppeteer._policy._current_agenda.name)

        print("Agenda state probabilities")
        for (agenda_name, agenda_states) in self._puppeteer._agenda_states.items():
            # TODO Hacky access to state probabilities.
            tpm = agenda_states._state_probabilities._probabilities
            print("    %s:" % agenda_name)
            for (state_name, p) in tpm.items():
                print("        %s: %.3f" % (state_name, p))
        
        if actions:
            print("Actions:")
            for a in actions:
                print("    %s" % a)
        else:
            print("No actions")

        #print("Puppeteer policy")
        #print("tmc", self._puppeteer._policy._times_made_current)
        #print("twop", self._puppeteer._policy._turns_without_progress)
        return (actions, extractions)


if __name__ == "__main__":
    #agendas = [get_location.create_agenda(), make_payment.create_agenda()]
    #agendas[0].store("agendas/get_location.yaml")
    #agendas[1].store("agendas/make_payment.yaml")
    # Set up trigger detector loader.
    trigger_detector_loader = MyTriggerDetectorLoader(default_snips_path="../turducken/data/training/puppeteer")
    
    # Load agendas
    get_location = Agenda.load("agendas/get_location.yaml", trigger_detector_loader)
    make_payment = Agenda.load("agendas/make_payment.yaml", trigger_detector_loader)
    agendas = [get_location, make_payment]
 
    tc = TestConversation(agendas)
    tc.say("Hello")
    tc.say("Why?")
    tc.say("routing number: 8998 account number: 12321312321")
    tc.say("None of your business")
    tc.say("No way")
    tc.say("routing number: 8998 account number: 12321312321")
    tc.say("No way")
    #tc.say("I live in Chicago")


# get_location.store("get_location.yaml")
# make_payment.store("make_payment.yaml")

# "Hello"
# "None of your business"
# "No way"
# "I live in Chicago"

# "routing number: 8998 account number: 12321312321"