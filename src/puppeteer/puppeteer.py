import abc
from typing import List, Dict, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np

from .agenda import Action, Agenda, AgendaState
from .logger import Logger
from .observation import Observation
from .extractions import Extractions


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
        """Initialize a new policy

        Args:
            agendas: The agendas used by the policy.
        """
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
        """Initialize a new policy

        Args:
            agendas: The agendas used by the policy.
        """
        super(DefaultPuppeteerPolicy, self).__init__(agendas)
        # State
        self._current_agenda = None
        self._turns_without_progress = {a.name: 0 for a in agendas}
        self._times_made_current = {a.name: 0 for a in agendas}
        self._action_history: Dict[str, List[Action]] = {a.name: [] for a in agendas}
        self._log = Logger()

    def act(self, agenda_states: Dict[str, AgendaState]) -> List[Action]:
        """"Picks zero or more appropriate actions to take, given the current state of the conversation.

        See documentation of this method in PuppeteerPolicy.

        Args:
            agenda_states: For each agenda (indexed by name), the AgendaState object holding the current belief about
                the state of the agenda, based on the latest observations -- the observations that this method is
                reacting to.

        Returns:
            A list of Action objects representing actions to take, in given order.
        """
        agenda = self._current_agenda
        last_agenda = None
        actions: List[Action] = []

        if agenda is not None:
            self._log.begin(f"Current agenda is {agenda.name}.")
            agenda_state = agenda_states[agenda.name]

            self._log.begin(f"Puppeteer policy state for {agenda.name}:")
            self._log.add(f"Turns without progress: {self._turns_without_progress[agenda.name]}")
            self._log.add(f"Times used: {self._times_made_current[agenda.name]}")
            self._log.add(f"Action history: {[a.name for a in self._action_history[agenda.name]]}")
            self._log.end()

            # Update agenda state based on message.
            # What to handle in output?
            progress_flag = agenda.policy.made_progress(agenda_state)
            # Note that even if the agenda is considered done at this point, having reached a
            # terminal state as the result of the incoming observations, it still gets to do
            # a final action.
            done_flag = progress_flag and agenda.policy.is_done(agenda_state)
            if progress_flag:
                self._log.add("We have made progress with the agenda.")
                self._turns_without_progress[agenda.name] = 0
            else:
                self._log.add("We have not made progress with the agenda.")
                # At this point, the current agenda (if there is
                # one) was the one responsible for our previous
                # reply in this convo. Only this agenda has its
                # turns_without_progress counter incremented.
                self._turns_without_progress[agenda.name] += 1

            turns_without_progress = self._turns_without_progress[agenda.name]

            if turns_without_progress >= 2:
                self._log.add("The agenda has been going on for too long without progress and will be stopped.")
                agenda_state.reset()
                self._current_agenda = None
                last_agenda = agenda
            else:
                # Run and see if we get some actions.
                action_history = self._action_history[agenda.name]
                self._log.begin("Picking actions for the agenda.")
                actions = agenda.policy.pick_actions(agenda_state, action_history, turns_without_progress)
                self._log.end()
                self._action_history[agenda.name].extend(actions)

                if not done_flag:
                    self._log.add("The agenda is not in a terminal state, so keeping it as current.")
                    # Keep going with this agenda.
                    self._log.end()
                    return actions
                else:
                    self._log.add("The agenda is in a terminal state, so will be stopped.")
                    # We inactivate this agenda. Will choose a new agenda
                    # in the main while-loop below.
                    # We're either done with the agenda, or had too many turns
                    # without progress.
                    # Do last action if there is one.
                    agenda_state.reset()
                    self._current_agenda = None
                    last_agenda = agenda
            self._log.end()
        # Try to pick a new agenda.
        self._log.begin("Trying to find a new agenda to start.")
        for agenda in np.random.permutation(self._agendas):
            agenda_state = agenda_states[agenda.name]
            self._log.begin(f"Considering agenda {agenda.name}.")

            if agenda == last_agenda:
                self._log.add("Just stopped this agenda, will not start it immediately again.")
                self._log.end()
                continue
            elif self._times_made_current[agenda.name] > 1:
                self._log.add(f"This agenda has already been used {self._times_made_current[agenda.name]} times, " +
                              "will not start it again.")
                self._log.end()
                continue

            if agenda.policy.can_kick_off(agenda_state):
                # If we can kick off, make this our active agenda, do actions and return.
                self._log.add("The agenda can kick off. This is our new agenda!")
                self._log.begin(f"Puppeteer policy state for {agenda.name}:")
                self._log.add(f"Turns without progress: {self._turns_without_progress[agenda.name]}")
                self._log.add(f"Times used: {self._times_made_current[agenda.name]}")
                self._log.add(f"Action history: {[a.name for a in self._action_history[agenda.name]]}")
                self._log.end()

                # TODO When can the agenda be done already here?
                done_flag = agenda.policy.is_done(agenda_state)
                agenda_state.reset()

                # Make this our current agenda.
                self._current_agenda = agenda
                self._times_made_current[agenda.name] += 1

                # Do first action.
                # TODO run_puppeteer() uses [] for the action list, not self._action_history
                self._log.begin("Picking actions for the agenda.")
                new_actions = agenda.policy.pick_actions(agenda_state, [], 0)
                self._log.end()
                actions.extend(new_actions)
                self._action_history[agenda.name].extend(new_actions)

                # TODO This is the done_flag from kickoff. Should check again now? Probably better to enforce in Agenda
                # that start states are never terminal.
                if done_flag:
                    self._log.add("We started the agenda, but its start state is a terminal state, so stopping it.")
                    self._log.add("Finishing act phase without a current agenda.")
                    self._current_agenda = None
                self._log.end()
                self._log.end()
                return actions
            self._log.end()
        self._log.end()

        # We failed to take action with an old agenda
        # and failed to kick off a new agenda. We have nothing.
        self._log.add("Finishing act phase without a current agenda.")

        return actions

    def plot_state(self, fig: plt.Figure, agenda_states: Dict[str, AgendaState]) -> None:
        """Plot the state of the current agenda, if any.

        Args:
            fig: Figure to plot to.
            agenda_states: For each agenda (indexed by name), the AgendaState object holding the current agenda state.
        """
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
           useful information about the world.
        3. The information gathered is fed to the Puppeteer through its react() method. The Puppeteer chooses a sequence
           of actions to take based on the information.
        4. The surrounding implementation takes the Puppeteer's action, and realizes them, typically providing some kind
           of reply to the other party.
    """

    def __init__(self, agendas: List[Agenda],
                 policy_cls: Type[PuppeteerPolicy] = DefaultPuppeteerPolicy,
                 plot_state: bool = False) -> None:
        """Initialize a new Puppeteer.

        Args:
            agendas: List of agendas to be used by the Puppeteer.
            policy_cls: The policy delegate class to use.
            plot_state: If true, the updated state of the current agenda is plotted after each turn.
        """
        self._agendas = agendas
        self._agenda_states = {a.name: AgendaState(a) for a in agendas}
        self._last_actions: List[Action] = []
        self._policy = policy_cls(agendas)
        if plot_state:
            self._fig = plt.figure()
            self._policy.plot_state(self._fig, self._agenda_states)
        else:
            self._fig = None
        self._log = Logger()

    @property
    def log(self) -> str:
        """Returns a log string from the latest call to react().

        The log string contains information that is helpful in understanding the inner workings of the puppeteer -- why
        it acts the way it does based on the inputs, and what its internal state is.
        """
        return self._log.log

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
        self._log.clear()
        self._log.begin("Inputs")
        self._log.begin("Observations")
        for o in observations:
            self._log.add(str(o))
        self._log.end()
        self._log.begin("Extractions")
        for name in old_extractions.names:
            self._log.add(f"{name}: '{old_extractions.extraction(name)}'")
        self._log.end()
        self._log.end()
        new_extractions = Extractions()
        self._log.begin("Update phase")
        for agenda_state in self._agenda_states.values():
            extractions = agenda_state.update(self._last_actions, observations, old_extractions)
            new_extractions.update(extractions)
        self._log.end()
        self._log.begin("Act phase")
        self._last_actions = self._policy.act(self._agenda_states)
        self._log.end()
        self._log.begin("Outputs")
        self._log.begin("Actions")
        for a in self._last_actions:
            self._log.add(str(a))
        self._log.end()
        self._log.begin("Extractions")
        for name in new_extractions.names:
            self._log.add(f"{name}: '{new_extractions.extraction(name)}'")
        self._log.end()
        self._log.end()
        if self._fig is not None:
            self._policy.plot_state(self._fig, self._agenda_states)
        return self._last_actions, new_extractions
