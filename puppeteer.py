import abc
from typing import List, Dict, Tuple, Type

import numpy as np
import matplotlib.pyplot as plt

from .agenda import Action, Agenda, AgendaState
from .logging import Logger
from .observation import Observation
from .extractions import Extractions
from .knowledge.knowledge_center import KNOWLEDGE

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
    def act(self, agenda_states: Dict[str, AgendaState], extractions: Extractions) -> List[Action]:
        """"Picks zero or more appropriate actions to take, given the current state of the conversation.

        Args:
            agenda_states: For each agenda (indexed by name), the AgendaState object holding the current belief about
                the state of the agenda, based on the latest observations -- the observations that this method is
                reacting to.

        Returns:
            A list of Action objects representing actions to take, in given order.
        """
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
        # Policy metadata
        self._active_agendas: Dict[str, Agenda] = {} # agendas that are able to start
        self._turns_without_progress = {a.name: -1 for a in agendas} #number of consecutive turns without progress
        self._times_made_current = {a.name: 0 for a in agendas}
        self._action_history: Dict[str, List[Action]] = {a.name: [] for a in agendas}
        self._log = Logger()

    def act(self, agenda_states: Dict[str, AgendaState], extractions: Extractions) -> Tuple[List[Action], Extractions]:
        """"Picks zero or more appropriate actions to take, given the current state of the conversation.

        See documentation of this method in PuppeteerPolicy.

        Args:
            agenda_states: For each agenda (indexed by name), the AgendaState object holding the current belief about
                the state of the agenda, based on the latest observations -- the observations that this method is
                reacting to.

        Returns:
            A list of Action objects representing actions to take, in given order.
        """

        if len(self._active_agendas) == 0:
            # if no active agendas ==> then do nothing
            self._log.add("No active agenda, will do nothing.")
            return ([], Extractions())

        finished_agenda_names: List[str] = []
        actions: List[Action] = []

        self._log.begin("Active agendas")
        # Main loop: go through each active agenda to pick actions
        for agenda_name, agenda in self._active_agendas.items():
            self._log.begin("{}".format(agenda_name))
            # Report active agendas and their metadata 
            # such as turns_without_progress, times_made_current, action_history
            agenda_state = agenda_states[agenda_name]

            self._log.begin("Metadata")
            self._log.add("Previous turns without progress: {}".format(self._turns_without_progress[agenda_name]))
            self._log.add("Times used: {}".format(self._times_made_current[agenda_name]))
            self._log.add("Action history: {}".format([a.name for a in self._action_history[agenda_name]]))
            self._log.end()

            # Check if this agenda makes progress in this turn.
            # There are two types of progress: progress after kick-off (transition triggers) and kick-off.
            # In other words, we consider kicking off an agenda to be a progress too.
            # Note that agenda.policy.made_progress() only checks progress after a agenda has been kicked off (transition triggers).
            # We use turns_without_progress[agenda] to determine whether or not an agenda has just been kicked off
            # if turns_without_progress[agenda] == -1 and agenda is in active_agendas ==> we just kicked off this agenda
            progress_flag = agenda.policy.made_progress(agenda_state) or self._turns_without_progress[agenda_name] == -1
            if progress_flag:
                self._log.add("We have made progress with {}.".format(agenda_name))
                # if making progress, reset turns_without_progress to 0
                self._turns_without_progress[agenda_name] = 0
            else:
                self._log.add("We have not made progress with {}.".format(agenda_name))
                # if no progress, penalize by increasing turns_without_progress by 1.
                self._turns_without_progress[agenda_name] += 1

            # if this agenda does make progress, check if it reached its terminus state
            # Note that even if the agenda is considered done at this point, having reached a
            # terminal state as the result of the incoming observations, it still gets to do
            # a final action.
            done_flag = progress_flag and agenda.policy.is_done(agenda_state)

            # Now let see, what pre-defined actions we have for both progress and non-progress agendas
            # For progress agendas, we will look at their action maps for actions.
            # Whereas, for non-progress agendas, we use their stall action maps.
            turns_without_progress = self._turns_without_progress[agenda_name]
            action_history = self._action_history[agenda_name]

            self._log.begin("Picking actions for {}.".format(agenda_name))
            action = agenda.policy.pick_actions(agenda_state, action_history, turns_without_progress)
            # for act in action:
            #     print(agenda_name, act.text)
            self._action_history[agenda_name].extend(action)
            actions += action
            self._log.end()

            if done_flag:
                # We reach the terminus state of this agenda.
                # Will continue on other active agendas.
                self._log.add("{} is in a terminal state, so it will be stopped.".format(agenda_name))
                agenda_state.reset()
                finished_agenda_names.append(agenda_name)

            self._log.end()

        # Remove finished agendas from a list of active agendas.
        for agenda_name in finished_agenda_names:
            del self._active_agendas[agenda_name]
            self._turns_without_progress[agenda_name] = -1

        # Check if any chosen action posts some knowledge
        extractions = Extractions()
        for action in actions:
            if action.text in KNOWLEDGE:
                knowledge = KNOWLEDGE[action.text]
                for k, v in knowledge.items():
                    extractions.add_extraction(k, v)

        return (actions, extractions)

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
                 policy_cls: Type[DefaultPuppeteerPolicy] = DefaultPuppeteerPolicy,
                 plot_state: bool = False) -> None:
        """Initialize a new Puppeteer.

        Args:
            agendas: List of agendas to be used by the Puppeteer.
            policy_cls: The policy delegate class to use.
            plot_state: If true, the updated state of the current agenda is plotted after each turn.
        """
        self._agendas = agendas
        self._last_actions: List[Action] = []
        self._policy = policy_cls(agendas)
        self._plot_state = plot_state

        if self._plot_state:
            plt.ion()
            agenda_states = {}
            for a in agendas:
                fig, ax = plt.subplots()
                agenda_states[a.name] = AgendaState(a, fig, ax)
            self._agenda_states = agenda_states
        else:
            self._agenda_states = {a.name: AgendaState(a, None, None) for a in agendas}

        self._log = Logger()

    @property
    def log(self):
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
        active_agendas = self._policy._active_agendas
        self._log.begin("Update phase")
        for agenda_state in self._agenda_states.values():
            extractions = agenda_state.update(self._last_actions, observations, old_extractions, active_agendas)
            new_extractions.update(extractions)

        self._log.begin("Updating active agendas")
        for agenda in self._agendas:
            # Update the list of active agendas (if any agendas are kicked off)
            # and increase times_made_current counter: number of times we use this agenda
            agenda_state = self._agenda_states[agenda.name]
            if agenda.name not in active_agendas and agenda.policy.can_kick_off(agenda_state):
                self._log.add("{} is added to the list of active agendas".format(agenda.name))
                active_agendas[agenda.name] = agenda
                self._policy._times_made_current[agenda.name] += 1
        self._log.end()

        if self._plot_state:
            # If plot_state is enabled
            for agenda_state in self._agenda_states.values():
                agenda_state.plot()

        self._log.end()
        self._log.begin("Act phase")
        self._last_actions, action_extractions = self._policy.act(self._agenda_states, new_extractions)
        new_extractions.update(action_extractions)
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

        return self._last_actions, new_extractions

    def get_active_agenda_names(self) -> List[str]:
        """ Returns the active agenda names (the ones already kicked off).
        """
        return list(self._policy._active_agendas.keys())

    def get_active_states(self, active_agenda_names) -> Dict[str, tuple[str, float, str]]:
        """ Given the active agenda names, returns the most likely current state with it probability 
            and turns_without_progress (number of consecutive turns the agenda has been idle) for each agenda name.
        """
        active_states = {}
        for agenda_name in active_agenda_names:
            current_state_probability_map = self._agenda_states[agenda_name]._state_probabilities._probabilities
            current_state_name = max(current_state_probability_map, key=lambda x: current_state_probability_map[x])
            turns_without_progress = self._policy._turns_without_progress[agenda_name]
            turns_without_progress = str(turns_without_progress) if turns_without_progress != -1 else "NA"
            active_states[agenda_name] = (current_state_name, current_state_probability_map[current_state_name], turns_without_progress)

        return active_states
