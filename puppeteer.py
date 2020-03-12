import abc
from typing import List, Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np

from agenda import Action, Agenda, AgendaState
from observation import Observation
from extractions import Extractions


class PuppeteerPolicyManager(abc.ABC):
    # A puppeteer policy is responsible for selecting the agenda to run.
    # Corresponds to ConversationOrchestrator from the v0.1 description.

    @abc.abstractmethod
    def act(self, agenda_states: Mapping[str, AgendaState]) -> List[Action]:
        raise NotImplementedError()

    @abc.abstractmethod
    def _plot_state(self, fig):
        raise NotImplementedError()

class DefaultPuppeteerPolicyManager(PuppeteerPolicyManager):
    # Essentially the same policy as run_puppeteer().
    
    def __init__(self, agendas: List[Agenda]):
        self._agendas = agendas
        # State
        self._current_agenda = None
        self._turns_without_progress = {a._name: 0 for a in agendas}
        self._times_made_current = {a._name: 0 for a in agendas}
        self._action_history = {a._name: [] for a in agendas}
        
    def _deactivate_agenda(self, agenda_name: str):
        # TODO Turducken currently keeps the history when an agenda is
        # deactivated. Can lead to avoiding states with few actions when an
        # agenda is re-run.
        #self._turns_without_progress[agenda_name] = 0
        #self._action_history[agenda_name] = []
        pass

    def act(self, agenda_states: Mapping[str, AgendaState]) -> List[Action]:
        agenda = self._current_agenda
        last_agenda = None
        actions = []

        if agenda is not None:
            agenda_state = agenda_states[agenda.name]
            
            # Update agenda state based on message.
            # What to handle in output?
            progress_flag = agenda.policy.made_progress(agenda_state)
            done_flag = progress_flag and agenda.policy.is_done(agenda_state)
            if progress_flag:
                self._turns_without_progress[agenda.name] = 0
            else:
                # At this point, the current agenda (if there is
                # one) was the one responsible for our previous
                # reply in this convo. Only this agenda has its
                # turns_without_progress counter incremented.
                self._turns_without_progress[agenda.name] += 1

            #print("Made progress for", agenda.name, progress_flag, self._turns_without_progress[agenda.name])
                
            turns_without_progress = self._turns_without_progress[agenda.name]
            
            if turns_without_progress >= 2:
                agenda_state.reset()
                self._deactivate_agenda(agenda.name)
                self._current_agenda = None
                last_agenda = agenda
            else:
                # Run and see if we get some actions.
                action_history = self._action_history[agenda.name]
                actions = agenda.policy.pick_actions(agenda_state, action_history, turns_without_progress)
                self._action_history[agenda.name].extend(actions)
                
                if not done_flag:
                    # Keep going with this agenda.
                    return actions
                else:
                    # We inactivate this agenda. Will choose a new agenda
                    # in the main while-loop below.
                    # We're either done with the agenda, or had too many turns
                    # without progress.
                    # Do last action if there is one.
                    agenda_state.reset()
                    self._deactivate_agenda(agenda._name)
                    self._current_agenda = None
                    last_agenda = agenda
                    agenda = None
                    #print("Done with agenda!")
        
        # Try to pick a new agenda.
        #print("Looking for new agenda")
        for agenda in np.random.permutation(self._agendas):
            #print("Trying to kick off agenda %s" % agenda.name)
            #print(agenda, last_agenda)
            agenda_state = agenda_states[agenda.name]
            
            if agenda.policy.can_kickoff(agenda_state):
                # TODO When can the agenda be done already here?
                done_flag = agenda.policy.is_done(agenda_state)
                agenda_state.reset()
                kick_off_condition = True
            else:
                kick_off_condition = False

            if agenda == last_agenda or self._times_made_current[agenda._name] > 1:
                # TODO Better to do this before checking for kickoff?
                self._deactivate_agenda(agenda._name)
                agenda_state.reset()
                continue
    
            # IF we kicked off, make this our active agenda, do actions and return.
            if kick_off_condition:
                #print("Kicking off agenda!")
                # Successfully kicked this off.
                # Make this our current agenda.           
                self._current_agenda = agenda
                
                # Do first action.
                # TODO run_puppeteer() uses [] for the action list, not self._action_history
                new_actions = agenda.policy.pick_actions(agenda_state, [], 0)
                actions.extend(new_actions)
                self._action_history[agenda._name].extend(new_actions)

                # TODO This is the done_flag from kickoff. Should check again now?
                if done_flag:
                    self._deactivate_agenda(agenda._name)
                    self._current_agenda = None
                return actions
            else:
                self._deactivate_agenda(agenda._name)
            
        # We failed to take action with an old agenda
        # and failed to kick off a new agenda. We have nothing.
        return actions

    def _plot_state(self, fig, agenda_states):
        plt.figure(fig.number)
        plt.clf()
        if self._current_agenda is None:
            plt.title("No current agenda")
        else:
            agenda_name = self._current_agenda.name
            turns_without_progress = self._turns_without_progress[agenda_name]
            times_made_current = self._times_made_current[agenda_name]
            action_history = self._action_history[agenda_name]
            title = "Current agenda: %s\n" % agenda_name
            title += "    %d turns without progress\n" % turns_without_progress
            title += "    made current %d times\n" % times_made_current
            title += "    action history: %s" % [a.name for a in action_history]
            plt.title(title)
            agenda_state = agenda_states[self._current_agenda.name]
            agenda_state._plot_state(fig)
        plt.show()


class Puppeteer:
    # Main class for an agendas-based conversation.
    # Corresponds to MachineEngine from the v0.1 description.

    def __init__(self, agendas: List[Agenda], policy_cls=DefaultPuppeteerPolicyManager, plot_state=False):
        self._agendas = agendas
        self._agenda_states = {a.name: AgendaState(a) for a in agendas}
        self._last_actions = []
        self._policy = policy_cls(agendas)
        if plot_state:
            self._fig = plt.figure()
            self._policy._plot_state(self._fig, self._agenda_states)
        else:
            self._fig = None
        
    def react(self, observations: List[Observation], old_extractions: Extractions) -> Tuple[List[Action], Extractions]:
        new_extractions = Extractions()
        for agenda_state in self._agenda_states.values():
            extractions = agenda_state._update(self._last_actions, observations, old_extractions)
            new_extractions.update(extractions)
        self._last_actions = self._policy.act(self._agenda_states)
        if self._fig is not None:
            self._policy._plot_state(self._fig, self._agenda_states)
        return (self._last_actions, new_extractions)

    def get_conversation_state(self):
        # Used for storing of conversation state
        # TODO store _beliefs and _last_actions.
        pass