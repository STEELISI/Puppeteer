import abc
from typing import List, Dict, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np

from agenda import Action, Agenda, AgendaState
from observation import Observation
from extractions import Extractions


class PuppeteerPolicy(abc.ABC):
    """Handles inter-agenda decisions about behavior.

    An PuppeteerPolicy is responsible for making decisions about what agenda to run and when to restart agendas, based
    on agenda states. Agenda-level decisions, e.g., which of the agenda's actions to choose in a given situation, are
    delegated to the AgendaPolicy instance associated with each Agenda.

    This class is an abstract class defining all methods that a PuppeteerPolicy must implement, most notably the act()
    method.

    A concrete PuppeteerPolicy subclass instance (object) is tied to a Puppeteer instance and is responsible for all
    action decisions for the conversation handled by the Puppeteer instance. This means that a PuppeteerPolicy instance
    is tied to a specific conversation, and may hold conversation-specific state.
    """

    def __init__(self, agendas: List[Agenda]) -> None:
        self._agendas = agendas

    @abc.abstractmethod
    def act(self, agenda_states: Dict[str, AgendaState]) -> List[Action]:
        """"Picks zero or more appropriate actions to take, given the current state of the conversation.

        Args:
            agenda_states: For each agenda (indexed by name), the AgendaState object holding the current belief about
                the state of the agenda, based on the latest observations -- the observations that this method is
                reacting to.

        Returns:
            A list of Action objects representing actions to take, in given order.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def plot_state(self, fig: plt.Figure, agenda_states: Dict[str, AgendaState]) -> None:
        raise NotImplementedError()


class DefaultPuppeteerPolicy(PuppeteerPolicy):
    """Handles inter-agenda decisions about behavior.

    This is the default PuppeteerPolicy implementation. See PuppeteerPolicy documentation for further details.
    """
    
    def __init__(self, agendas: List[Agenda]) -> None:
        super(DefaultPuppeteerPolicy, self).__init__(agendas)
        # State
        self._current_agenda = None
        self._turns_without_progress = {a.name: 0 for a in agendas}
        self._times_made_current = {a.name: 0 for a in agendas}
        self._action_history: Dict[str, List[Action]] = {a.name: [] for a in agendas}
        
    def _deactivate_agenda(self, agenda_name: str) -> None:
        # TODO Turducken currently keeps the history when an agenda is
        # deactivated. Can lead to avoiding states with few actions when an
        # agenda is re-run.
        # self._turns_without_progress[agenda_name] = 0
        # self._action_history[agenda_name] = []
        pass

    def act(self, agenda_states: Dict[str, AgendaState]) -> List[Action]:
        """"Picks zero or more appropriate actions to take, given the current state of the conversation.

        See documentation of this method in PuppeteerPolicy.
        """
        agenda = self._current_agenda
        last_agenda = None
        actions: List[Action] = []

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
                    self._deactivate_agenda(agenda.name)
                    self._current_agenda = None
                    last_agenda = agenda

        # Try to pick a new agenda.
        for agenda in np.random.permutation(self._agendas):
            agenda_state = agenda_states[agenda.name]
            
            if agenda.policy.can_kickoff(agenda_state):
                # TODO When can the agenda be done already here?
                done_flag = agenda.policy.is_done(agenda_state)
                agenda_state.reset()
                kick_off_condition = True
            else:
                kick_off_condition = False

            if agenda == last_agenda or self._times_made_current[agenda.name] > 1:
                # TODO Better to do this before checking for kickoff?
                self._deactivate_agenda(agenda.name)
                agenda_state.reset()
                continue
    
            # IF we kicked off, make this our active agenda, do actions and return.
            if kick_off_condition:
                # Successfully kicked this off.
                # Make this our current agenda.           
                self._current_agenda = agenda
                #self._times_made_current[agenda.name] += 1

                # Do first action.
                # TODO run_puppeteer() uses [] for the action list, not self._action_history
                new_actions = agenda.policy.pick_actions(agenda_state, [], 0)
                actions.extend(new_actions)
                self._action_history[agenda.name].extend(new_actions)

                # TODO This is the done_flag from kickoff. Should check again now?
                if done_flag:
                    self._deactivate_agenda(agenda.name)
                    self._current_agenda = None
                return actions
            else:
                self._deactivate_agenda(agenda.name)
            
        # We failed to take action with an old agenda
        # and failed to kick off a new agenda. We have nothing.
        return actions

    def plot_state(self, fig: plt.Figure, agenda_states: Dict[str, AgendaState]) -> None:
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
            agenda_state.plot_state(fig)
        plt.show()


class Puppeteer:
    """Agendas-based dialog bot.

    A Puppeteer instance is responsible for handling all aspects of the computer's side of the conversation, making
    decisions on conversational (or other) actions to take based on the conversational state and, possibly, other
    information about the world. There is a one-to-one relationship between Puppeteers and conversations, i.e., a
    Puppeteer handles a single conversation, and a conversation is typically handled by a single Puppeteer.

    A Puppeteer delegates most of its responsibilities to other classes, most notably:
    - Agenda: The Puppeteer's behavior is largely defined by a set of agendas associated with the Puppeteer. An
        Agenda can be described as a dialog mini-bot handling a specific topic or domain of conversation with a very
        specific and limited goal, e.g., getting to know the name of the other party. A Puppeteer's conversational
        abilities are thus defined by the collective abilities of the Agendas it can use.
    - PuppeteerPolicy: A PuppeteerPolicy is responsible for picking an agenda to use based on the conversational state,
        switching between agendas when appropriate. The choice of policy can be made through the Puppeteer's
        constructor, with the DefaultPuppeteerPolicy class as the default choice if no other class is specified.

    Architecturally, the Puppeteer is implemented much as a general agent, getting information about the world through
    observations and reacting by selecting actions appropriate for some goal. Its main purpose is to be used as a dialog
    bot, but could probably be used for other purposes as well.

    A Puppeteer session consists of first creating the Puppeteer, defining its set of agendas and its policy. Then the
    conversation is simply a series of turns, each turn having the following sequence of events:
        1. The other party acts, typically some kind of conversational action.
        2. The implementation surrounding the Puppeteer registers the actions of the other party, and possibly other
           useful information about the world. This
        3. The information gathered is fed to the Puppeteer through its react() method. The Puppeteer chooses a sequence
           of actions to take based on the information.
        4. The surrounding implementation takes the Puppeteer's action, and realizes them, typically providing some kind
           of reply to the other party.
    """
    def __init__(self, agendas: List[Agenda],
                 policy_cls: Type[PuppeteerPolicy] = DefaultPuppeteerPolicy,
                 plot_state: bool = False) -> None:
        self._agendas = agendas
        self._agenda_states = {a.name: AgendaState(a) for a in agendas}
        self._last_actions: List[Action] = []
        self._policy = policy_cls(agendas)
        if plot_state:
            self._fig = plt.figure()
            self._policy.plot_state(self._fig, self._agenda_states)
        else:
            self._fig = None
        
    def react(self, observations: List[Observation], old_extractions: Extractions) -> Tuple[List[Action], Extractions]:
        """"Picks zero or more appropriate actions to take, given the input and current state of the conversation.

        Note that the actions are only selected by the Puppeteer, but no actions are actually performed. It is the
        responsibility of the surrounding implementation to take concrete action, based on what is returned.

        Args:
            observations: A list of Observations made since the last turn.
            old_extractions: Extractions made during the whole conversation. This may also include extractions made by
                other modules based on the current turn.

        Returns:
            A pair consisting of:
            - A list of Action objects representing actions to take, in given order.
            - An updated Extractions object, combining the input extractions with any extractions made by the Puppeteer
              in this method call.
        """
        new_extractions = Extractions()
        for agenda_state in self._agenda_states.values():
            extractions = agenda_state.update(self._last_actions, observations, old_extractions)
            new_extractions.update(extractions)
        self._last_actions = self._policy.act(self._agenda_states)
        if self._fig is not None:
            self._policy.plot_state(self._fig, self._agenda_states)
        return self._last_actions, new_extractions
