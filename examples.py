from typing import List

from puppeteer import 
from agenda import Action, Agenda, Observation, Puppeteer, State, Trigger


# These two functions are placeholders for logic that produces input to the
# puppeteer and acts based on its output. Especially note that the puppeteer
# only selects a set of actions -- it is an external responsibility to know
# how to actually act
def make_observations() -> List[Observation]:
    pass

def do_actions(actions: List[Action]):
    pass



# Load agendas from files
agendas = [Agenda.load("get_location.yaml"),
           Agenda.load("get_full_name.yaml")]

# Make puppeteer based on agendas
puppeteer = Puppeteer(agendas)

# Run puppeteer
while True:
    observations = make_observations()
    actions = puppeteer.react(observations)
    do_actions(actions)



# Agendas can also be created programmatically
# This example is not showing all setup necessary. A typical use case will be
# loading an agenda from file, but good to have this option as well.
agenda = Agenda('get_location')
# Add states
agenda.add_state(State('start_state', 'The state we start in.'))
agenda.add_state(State('got_specific_loc', 'They gave us a specific enough location we can end.'))
agenda.add_state(State('need_more_specifics', 'They gave us a location, but we want them to be more specific'))
agenda.add_state(State('push_back', 'They are objecting to telling us.'))
agenda.add_state(State('give_up', 'We asked enough - we are going to give up.'))
# Add triggers
snips_folder = "/home/snips"
agenda.add_trigger(Trigger('push_back_intent', snips_folder))
agenda.add_trigger(Trigger('broad_loc', snips_folder))
agenda.add_trigger(Trigger('specific_loc', snips_folder))
agenda.add_trigger(Trigger('why_intent', snips_folder))
# Add transitions
agenda.add_transition('start_state', 'why_intent', 'push_back')
# ...
# Add actions
# ...


