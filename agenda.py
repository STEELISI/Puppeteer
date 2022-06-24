import abc
from typing import Any, Dict, List, Optional, Tuple, Type

import networkx as nx
import yaml

from .extractions import Extractions
from .logging import Logger
from .observation import Observation
from .trigger_detector import TriggerDetector, TriggerDetectorLoader


def _check_dict_fields(cls: Type, d: Dict[str, Any], fields: List[Tuple[str, Type]]) -> None:
    for (name, typ) in fields:
        if name not in d:
            raise ValueError("Missing field for %s: %s" % (cls.__name__, name))
        elif not isinstance(d[name], typ):
            raise TypeError("Field %s for %s should be of type %s, but got %s" % (name, cls.__name__, typ.__name__, type(d[name]).__name__))
    unexpected = frozenset(d).difference(list(zip(*fields))[0])
    if unexpected:
        raise ValueError("Unexpected field(s) for %s: %s" % (cls.__name__, unexpected))

def _check_policy_dict_fields(cls: Type, d: Dict[str, Any], fields: List[Tuple[str, Type]]) -> Dict[str, Any]:
    for (name, typ) in fields:
        if name not in d:
            raise ValueError("Missing field for %s: %s" % (cls.__name__, name))
        elif not isinstance(d[name], typ):
            if (name == "absolute_accept_thresh" or name == "min_accept_thresh_w_differential" or \
                name == "accept_thresh_differential" or name == "kickoff_thresh"):
                if isinstance(d[name], int):
                    d[name] = d[name] / 1.0
            else:
                raise TypeError("Field %s for %s should be of type %s, but got %s" % (name, cls.__name__, typ.__name__, type(d[name]).__name__))
    unexpected = frozenset(d).difference(list(zip(*fields))[0])
    if unexpected:
        raise ValueError("Unexpected field(s) for %s: %s" % (cls.__name__, unexpected))

    return d

class AgendaAttribute(abc.ABC):
    """Abstract class for attributes of an agenda, such as states, triggers and actions."""

    def __init__(self, name: str) -> None:
        """Initialize a new attribute.

        Args:
            name: The name of the attribute.
        """
        if not name:
            raise ValueError("Name must be non-empty")
        self._name = name

    @property
    def name(self) -> str:
        """Returns the name of this attribute."""
        return self._name

    @abc.abstractmethod
    def to_dict(self) -> Dict[str, str]:
        """Returns a dictionary representation of this attribute object.

        Returns:
            A dictionary representation of this object.
        """
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgendaAttribute":
        """Returns a new attribute object, based on its dictionary representation.

        Args:
            d: Dictionary representation of the object.

        Returns:
            The object.
        """
        raise NotImplementedError()


class State(AgendaAttribute):
    """Class naming and describing a state in an agenda."""
    
    def __init__(self, name: str, description: str = "") -> None:
        """Initialize a new State.

        Args:
            name: The name of the state.
            description: A description of the state.
        """
        super(State, self).__init__(name)
        self._description = description

    @property
    def description(self) -> str:
        """Returns the description of the state."""
        return self._description

    def to_dict(self) -> Dict[str, str]:
        """Returns a dictionary representation of this state.

        Returns:
            A dictionary representation of this state.
        """
        return {"name": self._name, "description": self._description}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "State":
        """Returns a new State object, based on its dictionary representation.

        Args:
            d: Dictionary representation of the state.

        Returns:
            The new object.
        """
        _check_dict_fields(cls, d, [("name", str), ("description", str)])
        return cls(d["name"], d["description"])


class Trigger(AgendaAttribute):
    """Class naming and describing a trigger in an agenda.
    
    A trigger is an event that can be detected, and that triggers some effect in the agenda, either a state transition
    or making agenda kickoff possible. This class is used by Agenda to declare the trigger, i.e., it represents the
    fact that the agenda has a trigger with a certain name, and provides a textual description of how the trigger is
    interpreted.
    
    The detection of when a trigger occurs is delegated to the TriggerDetector class. TriggerDetectors are registered
    in the Agenda, specifying, by trigger name, which of the agenda's triggers they are detecting.
    """

    def __init__(self, name: str, description: str = "") -> None:
        """Initialize a new Trigger.

        Args:
            name: The name of the trigger.
            description: A description of the trigger.
        """
        super(Trigger, self).__init__(name)
        self._description = description

    @property
    def description(self) -> str:
        """Returns the description of the trigger."""
        return self._description

    def to_dict(self) -> Dict[str, str]:
        """Returns a dictionary representation of this trigger.

        Returns:
            A dictionary representation of this trigger.
        """
        return {"name": self._name, "description": self._description}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Trigger":
        """Returns a new Trigger object, based on its dictionary representation.

        Args:
            d: Dictionary representation of the trigger.

        Returns:
            The new object.
        """
        _check_dict_fields(cls, d, [("name", str), ("description", str)])
        return cls(d["name"], d["description"])


class Action(AgendaAttribute):
    """Class naming and describing an action used in an agenda.

    An action defines an exclusive_flag, indicating whether the action can be combined with other actions in the same
    time step, or if it is exclusive. It is the responsibility of the AgendaPolicy to make sure that this flag is
    actually complied with.

    The allowed_repeats limit is the maximum number of times the action may be performed in a conversation.
    """

    def __init__(self, name: str, text: str = "", exclusive_flag: bool = True, allowed_repeats: int = 2) -> None:
        """Initialize a new Action.

        Args:
            name: The name of the action.
            text: The text associated with the action, typically what is "said" through the action.
            exclusive_flag: True if this action cannot be combined with other actions.
            allowed_repeats: The number of times the action may be used in a conversation.
        """
        super(Action, self).__init__(name)
        self._text = text
        self._exclusive_flag = exclusive_flag
        if allowed_repeats < 1:
            raise ValueError("Allowed number of repeats must be positive, got %d" % allowed_repeats)
        self._allowed_repeats = allowed_repeats
    
    def __repr__(self) -> str:
        """Return a string representation of the action."""
        return "%s: '%s'" % (self._name, self._text)
    
    def __str__(self) -> str:
        """Return a string representation of the action."""
        return repr(self)

    @property
    def text(self) -> str:
        """Return the text associated with the action."""
        return self._text

    @property
    def exclusive_flag(self) -> bool:
        """Returns the exclusivity flag."""
        return self._exclusive_flag
    
    @property
    def allowed_repeats(self) -> int:
        """Returns the number of allowed repeats."""
        return self._allowed_repeats

    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation of this action.

        Returns:
            A dictionary representation of this action.
        """
        return {"name": self._name, "text": self._text, "exclusive_flag": self._exclusive_flag,
                "allowed_repeats": self._allowed_repeats}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Action":
        """Returns a new Action object, based on its dictionary representation.

        Args:
            d: Dictionary representation of the action.

        Returns:
            The new object.
        """
        _check_dict_fields(cls, d, [("name", str), ("text", str), ("exclusive_flag", bool), ("allowed_repeats", int)])
        return cls(d["name"], d["text"], d["exclusive_flag"], d["allowed_repeats"])


class AgendaState:
    """Holds trigger and state probabilities for an agenda, for an ongoing conversation.

    This is the main class holding agenda-level conversation state. Some state is also held in the PuppeteerPolicy.

    The update() method is the main method for updating the agenda level state, based on extractions and observations
    made since the last time step.
    """
    def __init__(self, agenda: "Agenda", 
                fig = None,
                ax = None) -> None:
        """Initializes a new AgendaState.

        Args:
            agenda: The agenda that this object holds probabilities for.
        """
        self._agenda = agenda
        self._transition_trigger_probabilities = agenda.trigger_probabilities_cls(agenda, kickoff=False)
        self._kickoff_trigger_probabilities = agenda.trigger_probabilities_cls(agenda, kickoff=True)
        self._state_probabilities = agenda.state_probabilities_cls(agenda)
        self._pos = None
        self._log = Logger()
        if fig != None and ax != None:
            self._fig = fig
            self._ax = ax
            self._g = nx.MultiDiGraph()
            # Add states
            for s in self._agenda.state_names:
                self._g.add_node(s)
            self._g.add_node('ERROR_STATE')
            # Add transitions
            for s0 in self._agenda.state_names:
                for s1 in self._agenda.transition_connected_state_names(s0):
                    self._g.add_edge(s0, s1)

    @property
    def transition_trigger_probabilities(self) -> "TriggerProbabilities":
        """Returns the transition trigger probabilities part of the agenda state."""
        return self._transition_trigger_probabilities

    @property
    def kickoff_trigger_probabilities(self) -> "TriggerProbabilities":
        """Returns the kickoff trigger probabilities part of the agenda state."""
        return self._kickoff_trigger_probabilities

    @property
    def state_probabilities(self) -> "StateProbabilities":
        """Returns the state probabilities part of the agenda state."""
        return self._state_probabilities

    def update(self,
               actions: List[Action],
               observations: List[Observation],
               old_extractions: Extractions,
               active_agendas: Dict[str, "Agenda"],
               ) -> Extractions:
        """Updates the agenda-level state.

        Updating the agenda level state, based on extractions and observations made since the last time step.

        Args:
            actions: Actions performed in the last turn.
            observations: Observations made since the last turn.
            old_extractions: Extractions made in the conversation.

        Returns:
            New extractions made based on the input observations.
        """
        self._log.begin(f"Updating agenda {self._agenda.name}")

        if self._agenda.name not in active_agendas: #check for kickoff trigger if this agenda is not in the active list
            self._log.begin("Kickoff trigger probabilities")
            new_extractions = self._kickoff_trigger_probabilities.update(observations, old_extractions)
            self._log.end()

        else: #if this agenda has been active, check for transition triggers
            self._log.begin("Transition trigger probabilities")
            new_extractions = self._transition_trigger_probabilities.update(observations, old_extractions)
            self._log.end()

        self._log.begin("State probabilities")
        self._state_probabilities.update(self._transition_trigger_probabilities, actions)
        self._log.end()

        self._log.end()

        return new_extractions

    def reset(self) -> None:
        """Reset probabilities to the initial values for a newly started agenda."""
        self._state_probabilities.reset()

    def plot(self) -> None:
        """ Plot the state graph."""
        self._ax.clear()
        
        # Color nodes according to probability map.
        color_transition = ['#fff0e6', '#ffe0cc', '#ffd1b3', '#ffc299', '#ffb380', '#ffa366', '#ff944d', '#ff8533',
                            '#ff751a', '#ff6600']

        labels = {}
        color_map = []
        for node in self._g:
            prob = self._state_probabilities.probability(node)
            labels[node] = '{}\np={:.2f}'.format(node, prob)

            lvl = int(round(prob * 10)) # level or index in color_transition
            if lvl > 0:
                lvl = lvl - 1

            if lvl < len(color_transition):
                color_map.append(color_transition[lvl])
            else:
                color_map.append('grey')
        self._ax.set_title(self._agenda.name)
        nx.draw(G=self._g, pos=nx.circular_layout(self._g), ax=self._ax, node_color=color_map, labels=labels, node_size=2000)


class TriggerProbabilities(abc.ABC):
    """Handles trigger probabilities for an ongoing conversation.

    Trigger probabilities represent all information in observations that is relevant for state transition.

    This is an abstract class. A concrete TriggerProbabilities subclass is also responsible for updating trigger
    probabilities, by implementing the update() method.
    """

    def __init__(self, agenda: "Agenda", kickoff: bool = False) -> None:
        """Initializes a new TriggerProbabilities object.

        Args:
            agenda: The agenda for which this object holds trigger probabilities.
            kickoff: True if the probabilities are kickoff probabilities. Otherwise
                they are transition probabilities.
        """
        if kickoff:
            self._trigger_detectors = agenda.kickoff_trigger_detectors
            self._probabilities = {tr.name: 0.0 for tr in agenda.kickoff_triggers}
        else:
            self._trigger_detectors = agenda.transition_trigger_detectors
            self._probabilities = {tr.name: 0.0 for tr in agenda.transition_triggers}
        self._non_trigger_prob = 1.0

    @property
    def probabilities(self) -> Dict[str, float]:
        """Returns the trigger probabilities."""
        return self._probabilities

    @property
    def non_trigger_prob(self) -> float:
        """Returns the probability of no trigger."""
        return self._non_trigger_prob

    @property
    def trigger_detectors(self) -> List[TriggerDetector]:
        """Return a list of trigger detectors used to compute probabilities."""
        return self._trigger_detectors

    @abc.abstractmethod
    def update(self, observations: List[Observation], old_extractions: Extractions) -> Extractions:
        """Updates trigger probabilities based on extractions and observations since the last time step.

        Trigger probabilities represent all information in observations that is relevant for state transition between
        time steps t and t+1. The input observations were made between time steps t and t+1, while the extractions
        include all extractions made since the start of the conversation, up to time step t.

        Post-update, this TriggerProbabilities represent the trigger probabilities relevant for state transition between
        time steps t and t+1.

        In the default implementation (see DefaultTriggerProbabilities), the update of the trigger probabilities is
        based only on the observations and extractions, and dies not take the previous step's trigger probabilities into
        account. This seems reasonable for most definitions of trigger probability update, but is not required.

        Args:
            observations: Observations made since the last time step.
            old_extractions: Extractions made in the conversation.

        Returns:
            New extractions made based on the observations.
        """
        raise NotImplementedError()
        

class DefaultTriggerProbabilities(TriggerProbabilities):
    """Handles trigger probabilities for an ongoing conversation.

    This is the default TriggerProbabilities implementation. See class TriggerProbabilities for more details.
    """
    def __init__(self, agenda: "Agenda", kickoff: bool = False):
        """Initializes a new DefaultTriggerProbabilities object.

        Args:
            agenda: The agenda for which this object holds trigger probabilities.
            kickoff: True if the probabilities are kickoff probabilities. Otherwise
                they are transition probabilities.
        """
        super(DefaultTriggerProbabilities, self).__init__(agenda, kickoff)
        self._log = Logger()

    def update(self, observations: List[Observation], old_extractions: Extractions) -> Extractions:
        """Updates trigger probabilities based on extractions and observations since the last time step.

        See method documentation in superclass for more details.

        Args:
            observations: Observations made since the last time step.
            old_extractions: Extractions made in the conversation.

        Returns:
            New extractions made based on the observations.
        """
        trigger_map: Dict[str, float] = {}
        new_extractions = Extractions()
       
        for trigger_detector in self.trigger_detectors:
            #print(trigger_detector.trigger_names)
            self._log.begin(f"Trigger detector with trigger names {trigger_detector.trigger_names}")
            trigger_map_out, extractions = trigger_detector.trigger_probabilities(observations, old_extractions)

            if extractions.names:
                self._log.begin("Extractions")
                for name in extractions.names:
                    self._log.add(f"{name}: {extractions.extraction(name)}")
                self._log.end()
            new_extractions.update(extractions)

            self._log.begin("Triggers")
            # if not trigger_map_out:
            #     self._log.add("no trigger(s) detected")

            for (trigger_name, p) in trigger_map_out.items():
                self._log.add(f"{trigger_name}: {p:.3f}")
                if trigger_name in self._probabilities:
                    if trigger_name not in trigger_map:
                        trigger_map[trigger_name] = p
                    elif trigger_map[trigger_name] < p:
                        trigger_map[trigger_name] = p
            self._log.end()
            self._log.end()

        if trigger_map:
            non_trigger_prob = 1.0 - max(trigger_map.values())
        else:
            non_trigger_prob = 1.0

        # Normalization
        sum_total = sum(trigger_map.values()) + non_trigger_prob        
        non_trigger_prob = non_trigger_prob / sum_total
        for intent in trigger_map:
            trigger_map[intent] = trigger_map[intent] / sum_total

        for t in self._probabilities.keys():
            if t in trigger_map:
                self._probabilities[t] = trigger_map[t]
            else:
                self._probabilities[t] = 0.0
        
        self._non_trigger_prob = non_trigger_prob

        if trigger_map:
            self._log.begin("Final trigger probabilities")
            for (name, p) in self._probabilities.items():
                self._log.add(f"{name}: {p:.3f}")
            self._log.add(f"no trigger: {non_trigger_prob:.3f}")
            self._log.end()

        return new_extractions


class StateProbabilities(abc.ABC):
    """Handles state probabilities for an ongoing conversation.

    The state probabilities extend the state space of the agenda with a special "error" state that reflects the
    situation where the conversation has failed or become confused in some way.

    The Puppeteer currently uses a concept of state probabilities where the sum of all state probabilities may not sum
    to exactly one. Some individual state probabilities may even be greater than one. To interpret these probability
    measures as "standard" probabilities, simply normalize them by dividing by the sum.

    This is an abstract class. A concrete StateProbabilities subclass is also responsible for updating state
    probabilities, implementing the update() method. This includes defining the exact interpretation of the error state
    and how its probability is set.
    """

    def __init__(self, agenda: "Agenda") -> None:
        """Initializes a new StateProbabilities object.

        Args:
            agenda: The agenda for which this object holds state probabilities.
        """
        self._agenda = agenda
        self._probabilities = {s.name: 0.0 for s in agenda.states}
        self._probabilities["ERROR_STATE"] = 0.0
        self._probabilities[agenda.start_state.name] = 1.0

    @property
    def probabilities(self) -> Dict[str, float]:
        """Returns the state probabilities."""
        return self._probabilities

    def probability(self, state_name: str) -> float:
        """Returns the probability for the state with the given name.

        Args:
            state_name: The name of the state.

        Returns:
            The state probability.
        """
        return self._probabilities[state_name]

    def reset(self) -> None:
        """Reset state probabilities to the initial values for a newly started agenda."""
        for state_name in self._probabilities:
            if state_name == self._agenda.start_state.name:
                self._probabilities[state_name] = 1.0
            else:
                self._probabilities[state_name] = 0.0

    @abc.abstractmethod
    def update(self, trigger_probabilities: TriggerProbabilities, actions: List[Action]) -> None:
        """Updates state probabilities based on trigger probabilities.

        Trigger probabilities represent all information in observations that is relevant for state transition. This
        method is responsible for updating state probabilities, taking this information into account. If the pre-update
        probabilities are the probabilities for time step t, post-update probabilities are the probabilities for time
        step t+1, where the trigger probabilities reflect observations made between steps t and t+1.

        Args:
            trigger_probabilities: The trigger probabilities.
            actions: Actions performed in the last turn.
        """
        raise NotImplementedError()


class DefaultStateProbabilities(StateProbabilities):
    """Handles state probabilities for an ongoing conversation.

    This is the default StateProbabilities implementation. See class StateProbabilities for more details.
    """
    def __init__(self, agenda: "Agenda"):
        """Initializes a new DefaultStateProbabilities object.

        Args:
            agenda: The agenda for which this object holds state probabilities.
        """
        super(DefaultStateProbabilities, self).__init__(agenda)
        self._log = Logger()

    def update(self, trigger_probabilities: TriggerProbabilities, actions: List[Action]) -> None:
        """Updates state probabilities based on trigger probabilities.

        Trigger probabilities represent all information in observations that is relevant for state transition. This
        method is responsible for updating state probabilities, taking this information into account. If the pre-update
        probabilities are the probabilities for time step t, post-update probabilities are the probabilities for time
        step t+1, where the trigger probabilities reflect observations made between steps t and t+1.

        Args:
            trigger_probabilities: The trigger probabilities.
            actions: Actions performed in the last turn.
        """

        """ We don't need this for multiple agendas

        # Check if the last of the actions taken "belongs" to this agenda. Earlier
        # actions may be the finishing actions of a deactivated agenda.
        #print('actions: {}'.format(actions))
        if actions and not actions[-1] in self._agenda.actions:
            return

        """
        
        current_probability_map = self._probabilities
        trigger_map = trigger_probabilities.probabilities
        non_event_prob = trigger_probabilities.non_trigger_prob

        # Set up our new prob map.
        new_probability_map = {st: 0.0 for st in self._agenda.state_names}
        new_probability_map['ERROR_STATE'] = 0.0

        # Chance we actually have an event:
        p_event = 1.0 - non_event_prob
        #print('p_event: {}'.format(p_event))
        #print('current_prob_map: {}'.format(current_probability_map))
        #print('trigger_map: {}'.format(trigger_map))

        # # 1) Update state probability according to to_move
        for st in self._agenda.state_names:
            # Only if the state is not terminus
            if st in self._agenda.terminus_names:
                continue
            to_move = current_probability_map[st] * p_event
            new_probability_map[st] = max(0.0, current_probability_map[st] - to_move)

        # 2) Update state probability according to trigger_map
        for st in self._agenda.state_names:
            # Only if the state is not terminus
            if st in self._agenda.terminus_names:
                continue

            to_move = current_probability_map[st] * p_event
            if round(to_move, 1) > 0.0:
                for event in trigger_map:
                    trans_prob = to_move * trigger_map[event]
                    if event in self._agenda.transition_trigger_names(st):
                        st2 = self._agenda.transition_end_state_name(st, event)
                        new_probability_map[st2] += min(1.0, trans_prob)
                        # Decrease our confidence that we've had some problems following the script, previously.
                        # Not part of paper.
                        new_probability_map['ERROR_STATE'] = max(0.0, new_probability_map['ERROR_STATE'] - trans_prob)
                    else:
                        # XXX Downgrade our probabilities if we don't have an event that matches a transition?
                        # for this particular state.
                        # Not part of paper.
                        # new_probability_map[st] = max(0.05, current_probability_map[st]-trigger_map[event])

                        # Up our confidence that we've had some problems following the script.
                        new_probability_map['ERROR_STATE'] = new_probability_map['ERROR_STATE'] + trans_prob
                        # if st not in self._agenda.terminus_names: 
                        #     new_probability_map[st] = max(0.00, current_probability_map[st] - trigger_map[event])

        self._probabilities = new_probability_map

        self._log.begin("Updated state probabilities")
        for (name, p) in self._probabilities.items():
            self._log.add(f"{name}: {p:.3f}")
        self._log.end()


class AgendaPolicy(abc.ABC):
    """Handles agenda-level decisions about behavior.

    An AgendaPolicy is responsible for making decisions about how to execute an agenda, most notably by choosing next
    action(s). This class is an abstract class defining all queries that an AgendaPolicy must handle.

    An AgendaPolicy does not hold any conversation-level state. It is considered an integral part of an Agenda, and its
    defining parameters are stored together with other agenda-defining information in agenda files.

    The conversation-level information that the AgendaPolicy uses to make its decisions is stored in AgendaState, and
    the PuppeteerPolicy object controlling the entire conversation. This information is provided to AgendaPolicy's query
    methods and has been updated by the Puppeteer to take new observations and/or extractions into account, before the
    method call.
    """
    def __init__(self, agenda: "Agenda") -> None:
        """Initializes a new AgendaPolicy.

        Args:
            agenda: The agenda that this policy handles.
        """
        self._agenda = agenda

    @abc.abstractmethod
    def made_progress(self, state: AgendaState) -> bool:
        """Returns true if the agenda made progress in the last turn.

        Args:
            state: The current state of the agenda.

        Returns:
            True if the agenda made progress in the last turn.
        """
        raise NotImplementedError()

    @abc.abstractmethod    
    def is_done(self, state: AgendaState) -> bool:
        """Returns true if the agenda is likely in a terminus state.

        Args:
            state: The current state of the agenda.

        Returns:
            True if the agenda is likely in a terminus state.
        """
        raise NotImplementedError()

    @abc.abstractmethod    
    def can_kick_off(self, state: AgendaState) -> bool:
        """Returns true if the agenda is likely in a state where it can kick off.

        Args:
            state: The current state of the agenda.

        Returns:
            True if the agenda is likely in a state where it can kick off.
        """
        raise NotImplementedError()

    @abc.abstractmethod    
    def pick_actions(self, state: AgendaState, action_history: List[Action],
                     turns_without_progress: int) -> List[Action]:
        """Picks zero or more appropriate actions to take, given the current state of the agenda.

        Args:
            state: The current state of the agenda.
            action_history: List of previous actions taken in the conversation.
            turns_without_progress: The number of turns passed without the agenda making progress.

        Returns:
            A list of actions to take.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation of the state of this policy.

        Returns:
            A dictionary representation of the state of this policy.
        """
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, d: Dict[str, Any], agenda: "Agenda") -> "AgendaPolicy":
        """Returns a new policy object, based on the dictionary representation of its state.

        Args:
            d: Dictionary representation of the state of the policy.
            agenda: The agenda that the policy handles.

        Returns:
            The new policy object.
        """
        raise NotImplementedError()
        
        
class DefaultAgendaPolicy(AgendaPolicy):
    """Handles agenda-level decisions about behavior.

    This is the default AgendaPolicy implementation.

    The policy is controlled by the following parameters:
    - absolute_accept_thresh:
        If we reach an acceptance state with more than this confidence, we are done.
    - min_accept_thresh_w_differential:
        The min threshold for considering confidence we've reached a terminus. If we update the state machine often, our
        confidence prob in each state can get diffused across multiple states over long input sequences. So if we are
        accept_thresh_differential more likely than other states, and above min_accept_thresh_w_differential, we
        consider ourselves done.
    - accept_thresh_differential:
        The relative threshold between a candidate terminus and the next likely state. Probabilities "bleed out" over
        multiple transitions (e.g., the longer the input sequences, the more likely we'll get confidence/probability
        spread across multiple states), so relative probabilities are important.
    - kickoff_thresh:
        If we have an kickoff trigger with at least this confidence, we consider the kickoff condition triggered.

    The following two parameters are included in the class, but not currently in use by the policy:
    - reuse
        True if we can keep using this agenda in the same conversation.
    - max_transitions
        Max number of times we have triggering conditions before giving up on reaching a terminus.

    See AgendaPolicy documentation for further details.
    """

    def __init__(self,
                 agenda: "Agenda",
                 reuse: bool = False,
                 max_transitions: int = 5,
                 absolute_accept_thresh: float = 0.6,
                 min_accept_thresh_w_differential: float = 0.2,
                 accept_thresh_differential: float = 0.1,
                 # TODO Convention right now: Have to be sure of kickoff.
                 kickoff_thresh: float = 1.0) -> None:
        """Initializes a new DefaultAgendaPolicy.

        Args:
            agenda: The agenda that this policy handles.
            reuse: True if the agenda can be reused. Currently not enforced.
            max_transitions: Maximum number of state transitions. Currently not enforced.
            absolute_accept_thresh: Absolute termination probability threshold.
            min_accept_thresh_w_differential: Absolute termination probability threshold for considering
                relative threshold.
            accept_thresh_differential: Relative (by difference) termination probability threshold.
            kickoff_thresh: Kickoff trigger probability threshold.
        """
        super(DefaultAgendaPolicy, self).__init__(agenda)
        # TODO The reuse field in currently unused. Not used by turducken
        self._reuse = reuse
        if max_transitions < 1:
            raise ValueError("max_transitions must be positive, got %d" % max_transitions)
        # TODO The max_transitions field in currently unused. Not used by turducken
        self._max_transitions = max_transitions
        if absolute_accept_thresh <= 0.0:
            raise ValueError("absolute_accept_thresh must be positive, got %f" % absolute_accept_thresh)
        self._absolute_accept_thresh = absolute_accept_thresh
        if min_accept_thresh_w_differential <= 0.0:
            raise ValueError("min_accept_thresh_w_differential must be positive, got %f" %
                             min_accept_thresh_w_differential)
        self._min_accept_thresh_w_differential = min_accept_thresh_w_differential
        if accept_thresh_differential <= 0.0:
            raise ValueError("accept_thresh_differential must be positive, got %f" % accept_thresh_differential)
        self._accept_thresh_differential = accept_thresh_differential
        if kickoff_thresh <= 0.0:
            raise ValueError("kickoff_thresh must be positive, got %f" % kickoff_thresh)
        self._kickoff_thresh = kickoff_thresh
        self._log = Logger()

    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation of the state of this policy.

        Returns:
            A dictionary representation of the state of this policy.
        """
        field_names = ["_reuse", "_max_transitions", "_absolute_accept_thresh",
                       "_min_accept_thresh_w_differential",
                       "_accept_thresh_differential", "_kickoff_thresh"]
        d = {f[1:]: getattr(self, f) for f in field_names}
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any], agenda: "Agenda") -> "DefaultAgendaPolicy":
        """Returns a new policy object, based on the dictionary representation of its state.

        Args:
            d: Dictionary representation of the state of the policy.
            agenda: The agenda that the policy handles.

        Returns:
            The new policy object.
        """
        d = _check_policy_dict_fields(cls, d, [("reuse", bool), ("max_transitions", int), ("absolute_accept_thresh", float),
                                    ("min_accept_thresh_w_differential", float), ("accept_thresh_differential", float),
                                    ("kickoff_thresh", float)])
        return cls(agenda, d["reuse"], d["max_transitions"],
                   d["absolute_accept_thresh"],
                   d["min_accept_thresh_w_differential"],
                   d["accept_thresh_differential"],
                   d["kickoff_thresh"])

    def __str__(self):
        return("reuse: {}, max_transitions: {}, absolute_accept_thresh: {}, min_accept_thresh_w_differential: {}, "\
                "accept_thresh_differential: {}, kickoff_thresh: {}".format(self._reuse, self._max_transitions, \
                self._absolute_accept_thresh, self._min_accept_thresh_w_differential, self._accept_thresh_differential, \
                self._kickoff_thresh))

    def made_progress(self, state: AgendaState) -> bool:
        """Returns true if the agenda made progress in the last turn.

        Args:
            state: The current state of the agenda.

        Returns:
            True if the agenda made progress in the last turn.
        """
        non_event_probability = state.transition_trigger_probabilities.non_trigger_prob
        error_state_probability = state.state_probabilities.probabilities["ERROR_STATE"]
        return non_event_probability <= 0.4 and error_state_probability <= .8

    def is_done(self, state: AgendaState) -> bool:
        """Returns true if the agenda is likely in a terminus state.

        Args:
            state: The current state of the agenda.

        Returns:
            True if the agenda is likely in a terminus state.
        """
        best = None
            
        # For state by decresing probabilities that we're in that state. 
        # TODO Probably simpler: just look at best and second-best state
        # TODO Don't access probability map directly
        probability_map = state.state_probabilities.probabilities
        sorted_states = {k: v for k, v in sorted(probability_map.items(), key=lambda item: item[1], reverse=True)}
        for (rank, st) in enumerate(sorted_states):
            if st in self._agenda.terminus_names:
                # If this is an accept state, we can set our best exit candidate.
                if rank == 0 and probability_map[st] >= self._absolute_accept_thresh:
                    return True
                elif rank == 0 and probability_map[st] >= self._min_accept_thresh_w_differential:
                    best = probability_map[st]
            # If we have an exit candidate, 
            if best is not None and rank == 1:
                if best - probability_map[st] >= self._accept_thresh_differential:
                    return True
        return False

    def can_kick_off(self, state: AgendaState) -> bool:
        """Returns true if the agenda is likely in a state where it can kick off.

        Args:
            state: The current state of the agenda.

        Returns:
            True if the agenda is likely in a state where it can kick off.
        """
        non_kickoff_probability = state.kickoff_trigger_probabilities.non_trigger_prob
        return 1.0 - non_kickoff_probability >= self._kickoff_thresh

    def pick_actions(self, state: AgendaState, action_history: List[Action],
                     turns_without_progress: int) -> List[Action]:
        """Picks zero or more appropriate actions to take, given the current state of the agenda.

        Args:
            state: The current state of the agenda.
            action_history: List of previous actions taken in the conversation.
            turns_without_progress: The number of turns passed without the agenda making progress.

        Returns:
            A list of actions to take.
        """
        actions_taken: List[Action] = []
        
        # Action map - maps states to a list of tuples of:
        # (action_name, function, arguments, 
        #  boolean to indicate if this an exclusive action that cannot be used
        #  with other actions, number of allowed repeats for this action)
        if turns_without_progress == 0:
            self._log.add("Using normal action map.")
            action_map = self._agenda.action_map
        else:
            self._log.add("Using stall action map.")
            action_map = self._agenda.stall_action_map
            
        # Work over the most likely state, to least likely, taking the first
        # actions we are allowed to given repeat allowance & exclusivity.
        # for state by decreasing probabilities that we're in that state:
        for st in {k: v for k, v in sorted(state.state_probabilities.probabilities.items(), key=lambda item: item[1],
                                           reverse=True)}:
            # XXX Maybe need to check likelihood.
            if st in action_map:
                self._log.add(f"State {st} is the most likely state that has actions defined.")
                for action_name in action_map[st]:
                    action = self._agenda.action(action_name)
                    exclusive_flag = action.exclusive_flag
                    allowed_repeats = action.allowed_repeats
                    
                    num_times_action_was_used = action_history.count(action)
                    
                    if num_times_action_was_used < allowed_repeats:
                        if exclusive_flag and actions_taken:
                            # Can't do an exclusive action if a non-exclusive
                            # action is already taken.
                            continue

                        actions_taken.append(action)
                        if exclusive_flag:
                            # No more actions to add
                            break
                if actions_taken:
                    self._log.add(f"Doing actions: {[a.name for a in actions_taken]}")
                    return actions_taken
                elif action_map == self._agenda.action_map:
                    self._log.add("No normal actions left to take.")
                    # All normal actions were used the maximum number of times.
                    # See if there are stall actions left.
                    if st in self._agenda.stall_action_map:
                        for action_name in self._agenda.stall_action_map[st]:
                            action = self._agenda.action(action_name)
                            allowed_repeats = action.allowed_repeats
                            
                            num_times_action_was_used = action_history.count(action)
                            
                            if num_times_action_was_used < allowed_repeats:
                                self._log.add(f"Using stall action {action.name} instead.")
                                return [action]
                    self._log.add("No stall actions to take either.")
                else:
                    self._log.add("No stall actions left to take.")
                # Couldn't find any action for most likely state.
                break
        return []


class Agenda:
    """Class defining agenda behavior.

    An Agenda object completely defines the behavior of an agenda. This includes:
    - The name of the agenda.
    - The agenda's state graph with:
      - States, also defining start and terminus states
      - Triggers for state transitions and kickoff
      - State transitions.
    - Trigger detectors for both kickoff and state transitions. These should implement trigger detection for:
      - All state transition triggers defined by the state graph.
      - All of the agenda's kickoff triggers.
    - All actions used by the agenda.
    - Action mappings defining which actions are applicable in which states. This mapping are divided into a normal
      action map that is used in situations where the conversation is going well, and a stall action map that is used
      when the conversation has stalled.
    - A policy class used by the agenda to make decisions regarding things like action selection. This is specified by
      providing a subclass of AgendaPolicy when creating an Agenda, either to the constructor, or the load() method.
    - A trigger probability handling class used by the agenda to keep track of trigger probabilities. This is specified
      by providing a subclass of TriggerProbabilities when creating an Agenda, either to the constructor, or the load()
      method.
    - A state probability handling class used by the agenda to keep track of state probabilities. This is specified by
      providing a subclass of StateProbabilities when creating an Agenda, either to the constructor, or the load()
      method.

    Note that an Agenda object only defines agenda behavior and does not store any conversation-level state. The same
    Agenda object can be used by different concurrently running conversations.

    An Agenda object can be created in two different ways:
    - Creating an object using the Agenda() constructor. The resulting object is a mostly empty agenda that must be
      further defined by calling its various setter methods. In this mode of agenda creation, the user must make sure
      that all required agenda information is set in the object, before it is actually used in a conversation.
    - Loading an agenda from file using the load() method. In this case, the method returns a ready-to-use agenda.
    """

    def __init__(self, name: str,
                 policy_cls: Type[AgendaPolicy] = DefaultAgendaPolicy,
                 state_probabilities_cls: Type[StateProbabilities] = DefaultStateProbabilities,
                 trigger_probabilities_cls: Type[TriggerProbabilities] = DefaultTriggerProbabilities) -> None:
        """ Initialize a new Agenda.

        Args:
            name: The name of the agenda.
            policy_cls: The policy class to use to control the agenda behavior.
            state_probabilities_cls: The class to use to compute state probabilities.
            trigger_probabilities_cls: The class to use to compute trigger probabilities.
        """
        self._name = name
        self._policy = policy_cls(self)
        self._trigger_probabilities_cls = trigger_probabilities_cls
        self._state_probabilities_cls = state_probabilities_cls
        # Setting everything else empty to begin with
        self._states: Dict[str, State] = {}
        self._transition_triggers: Dict[str, Trigger] = {}
        self._kickoff_triggers: Dict[str, Trigger] = {}
        self._transitions: Dict[str, Dict[str, str]] = {}
        self._start_state_name: Optional[str] = None
        self._terminus_names: List[str] = []
        self._actions: Dict[str, Action] = {}
        self._action_map: Dict[str, List[str]] = {}
        self._stall_action_map: Dict[str, List[str]] = {}
        self._kickoff_trigger_detectors: List[TriggerDetector] = []
        self._transition_trigger_detectors: List[TriggerDetector] = []

    @property
    def name(self) -> str:
        """Returns the name of the agenda."""
        return self._name

    @property
    def policy(self) -> AgendaPolicy:
        """Returns the policy of the agenda."""
        return self._policy

    @property
    def states(self) -> List[State]:
        """Returns the list of agenda states."""
        return list(self._states.values())

    @property
    def state_names(self) -> List[str]:
        """Returns the state names of the agenda."""
        return list(self._states.keys())

    @property
    def kickoff_triggers(self) -> List[Trigger]:
        """Returns the kickoff triggers of the agenda."""
        return list(self._kickoff_triggers.values())

    @property
    def transition_triggers(self) -> List[Trigger]:
        """Returns the transition triggers of the agenda."""
        return list(self._transition_triggers.values())

    @property
    def actions(self) -> List[Action]:
        """Returns the actions of the agenda."""
        return list(self._actions.values())

    def action(self, action_name: str) -> Action:
        """Returns the agenda action with the given name.

        Args:
            name: The name of the action:

        Returns:
            The action with the given name.
        """
        if action_name not in self._actions:
            raise ValueError("No action with name '%s'" % action_name)
        return self._actions[action_name]

    @property
    def action_map(self) -> Dict[str, List[str]]:
        """Returns the action map of the agenda."""
        return dict(self._action_map)

    @property
    def stall_action_map(self) -> Dict[str, List[str]]:
        """Returns the stall action map of the agenda."""
        return dict(self._stall_action_map)

    @property
    def start_state(self) -> State:
        """Returns start state of the agenda."""
        if self._start_state_name is not None:
            return self._states[self._start_state_name]
        else:
            raise Exception("Start state name undefined.")

    @property
    def terminus_states(self) -> List[State]:
        """Returns the terminus states of the agenda."""
        return [self._states[s] for s in self._terminus_names]

    @property
    def terminus_names(self) -> List[str]:
        """Returns the names of the terminus states of the agenda."""
        return list(self._terminus_names)

    def transition_trigger_names(self, state_name: str) -> List[str]:
        """Returns the names of the transition triggers used in the given state.

        Args:
            state_name: The name of the state.

        Returns:
            List of trigger names.
        """
        if state_name not in self._transitions:
            raise ValueError("No state with name '%s'" % state_name)
        return list(self._transitions[state_name].keys())

    def transition_end_state_name(self, state_name: str, trigger_name: str) -> str:
        """Returns the names of destination state for given trigger in given state.

        Args:
            state_name: The name of the original state.
            trigger_name: The name of the trigger.

        Returns:
            Name of the destination state.
        """
        if state_name not in self._transitions:
            raise ValueError("No state with name '%s'" % state_name)
        elif trigger_name not in self._transitions[state_name]:
            raise ValueError("No trigger with name '%s' for state '%s'" % (trigger_name, state_name))
        return self._transitions[state_name][trigger_name]

    def transition_connected_state_names(self, state_name: str) -> List[str]:
        """Returns the names of possible next states for given state.

        Args:
            state_name: The name of the original state.

        Returns:
            Names of the possible next states.
        """
        if state_name not in self._transitions:
            raise ValueError("No state with name '%s'" % state_name)
        return list(set(self._transitions[state_name].values()))

    @property
    def state_probabilities_cls(self) -> Type[StateProbabilities]:
        """Returns the class used by the agenda for computing state probabilities."""
        return self._state_probabilities_cls

    @property
    def trigger_probabilities_cls(self) -> Type[TriggerProbabilities]:
        """Returns the class used by the agenda for computing trigger probabilities."""
        return self._trigger_probabilities_cls

    @property
    def kickoff_trigger_detectors(self) -> List[TriggerDetector]:
        """Returns the kickoff trigger detectors used by this agenda."""
        return list(self._kickoff_trigger_detectors)

    @property
    def transition_trigger_detectors(self) -> List[TriggerDetector]:
        """Returns the transition trigger detectors used by this agenda."""
        return list(self._transition_trigger_detectors)

    def _to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation of this agenda.

        Returns:
            A dictionary representation of this agenda.
        """
        def to_dict(x: Any) -> Any:
            if isinstance(x, str):
                return x
            elif isinstance(x, int):
                return x
            elif isinstance(x, float):
                return x
            elif isinstance(x, list):
                return [to_dict(v) for v in x]
            elif isinstance(x, dict):
                return {k: to_dict(v) for (k, v) in x.items()}
            else:
                return x.to_dict()
        d = {"name": self._name}
        # Handle named fields separately
        field_names = ["_states", "_actions", "_transition_triggers", "_kickoff_triggers"]
        d.update({f[1:]: to_dict(list(getattr(self, f).values())) for f in field_names})
        # Other fields stored as-is
        field_names = ["_start_state_name", "_terminus_names", 
                       "_transitions", "_action_map",
                       "_stall_action_map", "_policy"]
        d.update({f[1:]: to_dict(getattr(self, f)) for f in field_names})
        return d

    @classmethod
    def _from_dict(cls, d: Dict[str, Any],
                   policy_cls: Type[AgendaPolicy],
                   state_probabilities_cls: Type[StateProbabilities],
                   trigger_probabilities_cls: Type[TriggerProbabilities]) -> "Agenda":
        """Returns a new Agenda object, based on its dictionary representation.

        Args:
            d: Dictionary representation of the agenda.
            policy_cls: The policy class to use to control the agenda behavior.
            state_probabilities_cls: The class to use to compute state probabilities.
            trigger_probabilities_cls: The class to use to compute trigger probabilities.

        Returns:
            The new agenda.
        """
        # Replace with objects in d, where appropriate.
        def from_dict_list(dict_list: List[Dict[str, Any]], new_cls: Type[AgendaAttribute]) -> Dict[str, Any]:
            if not isinstance(dict_list, list):
                raise TypeError("Expected list of dicts for class %s" % new_cls.__name__)
            obj_dict = {}
            for dd in dict_list:
                if not isinstance(dd, dict):
                    raise TypeError("Expected list of dicts for class %s" % new_cls.__name__)
                new_obj = new_cls.from_dict(dd)
                obj_dict[new_obj.name] = new_obj
            return obj_dict
        _check_dict_fields(Agenda, d, [("states", object), ("kickoff_triggers", object),
                                       ("transition_triggers", object), ("actions", object), ("policy", object),
                                       ("name", str), ("start_state_name", str), ("terminus_names", list),
                                       ("transitions", dict), ("action_map", dict), ("stall_action_map", dict)])
        states = from_dict_list(d["states"], State)
        kickoff_triggers = from_dict_list(d["kickoff_triggers"], Trigger)
        transition_triggers = from_dict_list(d["transition_triggers"], Trigger)
        actions = from_dict_list(d["actions"], Action)

        agenda = cls(d["name"], policy_cls=policy_cls, state_probabilities_cls=state_probabilities_cls,
                     trigger_probabilities_cls=trigger_probabilities_cls)
        # Special handling of policy
        agenda._policy = policy_cls.from_dict(d["policy"], agenda)
        #print(str(agenda._policy))
        # Restore all other fields, as stored in dict
        for state in states.values():
            agenda.add_state(state)
        agenda.set_start_state(d["start_state_name"])
        for trigger in kickoff_triggers.values():
            agenda.add_kickoff_trigger(trigger)
        for trigger in transition_triggers.values():
            agenda.add_transition_trigger(trigger)
        for action in actions.values():
            agenda.add_action(action)
        for name in d["terminus_names"]:
            if not isinstance(name, str):
                raise TypeError("Expected string for terminus name, got: %s" % type(name))
            agenda.add_terminus(name)
        for start_state_name in d["transitions"]:
            if not isinstance(start_state_name, str):
                raise TypeError("Expected string for start state name for transition, got: %s" % type(start_state_name))
            for trigger_name in d["transitions"][start_state_name]:
                if not isinstance(trigger_name, str):
                    raise TypeError("Expected string for trigger name for transition, got: %s" % type(trigger_name))
                end_state_name = d["transitions"][start_state_name][trigger_name]
                if not isinstance(end_state_name, str):
                    raise TypeError("Expected string for end state name for transition, got: %s" % type(end_state_name))
                agenda.add_transition(start_state_name, trigger_name, end_state_name)
        for state_name in d["action_map"]:
            if not isinstance(state_name, str):
                raise TypeError("Expected string for state name for action map, got: %s" % type(state_name))
            for action_name in d["action_map"][state_name]:
                if not isinstance(action_name, str):
                    raise TypeError("Expected string for action name for action map, got: %s" % type(action_name))
                agenda.add_action_for_state(action_name, state_name)
        for state_name in d["stall_action_map"]:
            if not isinstance(state_name, str):
                raise TypeError("Expected string for state name for stall action map, got: %s" % type(state_name))
            for action_name in d["stall_action_map"][state_name]:
                if not isinstance(action_name, str):
                    raise TypeError("Expected string for stall action name for action map, got: %s" % type(action_name))
                agenda.add_stall_action_for_state(action_name, state_name)
        return agenda

    def add_state(self, state: State) -> None:
        """Add a state to the agenda.

        Args:
            state: The state to add.
        """
        if state.name in self._states:
            raise ValueError("Agenda already has a state with name '%s'" % state.name)
        self._states[state.name] = state
        self._action_map[state.name] = []
        self._stall_action_map[state.name] = []
        self._transitions[state.name] = {}

    def set_start_state(self, state_name: str) -> None:
        """Set the name of the start state of the agenda.

        Args:
            state_name: The name of the start state.
        """
        if state_name not in self._states:
            raise ValueError("Invalid start state, no state with name '%s'" % state_name)
        self._start_state_name = state_name

    def add_terminus(self, state_name: str) -> None:
        """Add a state name to the list of terminus names for the agenda.

        Args:
            state_name: The terminus state name to add.
        """
        if state_name not in self._states:
            raise ValueError("Invalid terminus state, no state with name '%s'" % state_name)
        elif state_name in self._terminus_names:
            raise ValueError("Terminus state already set: '%s'" % state_name)
        self._terminus_names.append(state_name)

    def add_transition_trigger(self, trigger: Trigger) -> None:
        """Add a transition trigger to the agenda.

        Args:
            trigger: The trigger to add.
        """
        if trigger.name in self._transition_triggers:
            raise ValueError("Agenda already has a transition trigger with name '%s'" % trigger.name)
        self._transition_triggers[trigger.name] = trigger

    def add_kickoff_trigger(self, trigger: Trigger) -> None:
        """Add a kickoff trigger to the agenda.

        Args:
            trigger: The kickoff trigger to add.
        """
        if trigger.name in self._kickoff_triggers:
            raise ValueError("Agenda already has a kickoff trigger with name '%s'" % trigger.name)
        self._kickoff_triggers[trigger.name] = trigger

    def add_transition(self, start_state_name: str, trigger_name: str, end_state_name: str) -> None:
        """Add a transition to the agenda.

        Args:
            start_state_name: The name of the start (origin) state of the transition.
            trigger_name: The name of the trigger for the transition.
            end_state_name: The name of the end (destination) state of the transition.
        """
        if start_state_name not in self._transitions:
            raise ValueError("Invalid start state for transition, no state with name '%s'" % start_state_name)
        elif trigger_name not in self._transition_triggers:
            raise ValueError("Invalid trigger for transition, no transition trigger with name '%s'" % trigger_name)
        elif end_state_name not in self._states:
            raise ValueError("Invalid end state for transition, no state with name '%s'" % end_state_name)
        elif trigger_name in self._transitions[start_state_name]:
            raise ValueError("End state for transition, from state '%s' already set for trigger '%s'" %
                             (start_state_name, trigger_name))
        self._transitions[start_state_name][trigger_name] = end_state_name
    
    def add_action(self, action: Action) -> None:
        """Add an action to the agenda.

        Args:
            action: The action to add.
        """
        if action.name in self._actions:
            raise ValueError("Agenda already has an action with name '%s'" % action.name)
        self._actions[action.name] = action

    def add_action_for_state(self, action_name: str, state_name: str) -> None:
        """Add an action to a state of the agenda.

        Args:
            action_name: The name of the action.
            state_name: The name of the state.
        """
        if action_name not in self._actions:
            raise ValueError("Agenda has no action with name '%s'" % action_name)
        elif state_name not in self._states:
            raise ValueError("Invalid state for action, no state with name '%s'" % state_name)
        elif action_name in self._action_map[state_name]:
            raise ValueError("State '%s' already has action with name '%s'" % (state_name, action_name))
        self._action_map[state_name].append(action_name)
    
    def add_stall_action_for_state(self, action_name: str, state_name: str) -> None:
        """Add a stall action to a state of the agenda.

        Args:
            action_name: The name of the stall action.
            state_name: The name of the state.
        """
        if action_name not in self._actions:
            raise ValueError("Agenda has no action with name '%s'" % action_name)
        elif state_name not in self._states:
            raise ValueError("Invalid state for stall action, no state with name '%s'" % state_name)
        elif action_name in self._stall_action_map[state_name]:
            raise ValueError("State '%s' already has stall action with name '%s'" % (state_name, action_name))
        self._stall_action_map[state_name].append(action_name)

    def add_transition_trigger_detector(self, trigger_detector: TriggerDetector) -> None:
        """Add a transition trigger detector to the agenda.

        Args:
            trigger_detector: The trigger detector to add.
        """
        self._transition_trigger_detectors.append(trigger_detector)

    def add_kickoff_trigger_detector(self, trigger_detector: TriggerDetector) -> None:
        """Add a transition trigger detector to the agenda.

        Args:
            trigger_detector: The trigger detector to add.
        """
        self._kickoff_trigger_detectors.append(trigger_detector)

    def store(self, filename: str) -> None:
        """Store the agenda to file.

        Args:
            filename: Name of file to store the agenda in.
        """
        with open(filename, "w") as file:
            yaml.dump(self._to_dict(), file, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, filename: str, trigger_detector_loader: TriggerDetectorLoader,
             snips_multi_engine: bool = False,
             policy_cls: Type[AgendaPolicy] = DefaultAgendaPolicy,
             state_probabilities_cls: Type[StateProbabilities] = DefaultStateProbabilities,
             trigger_probabilities_cls: Type[TriggerProbabilities] = DefaultTriggerProbabilities) -> "Agenda":
        """Load an agenda from file.

        The policy class provided must be consistent with the policy parameters given in the file, i.e., it is the
        user's responsibility to know what policies are compatible with the file. This is of course only an issue when
        non-default policy classes are used.

        See class TriggerDetectorLoader for information on the trigger_detector_loader and snips_multi_engine
        parameters.

        Args:
            filename: The name of the file to load.
            trigger_detector_loader: The trigger detector loader to use to get trigger detectors for the agenda.
            snips_multi_engine: If True, load Snips trigger detectors in multi-engine mode.
            policy_cls: The policy class to use to control the agenda behavior.
            state_probabilities_cls: The class to use to compute state probabilities.
            trigger_probabilities_cls: The class to use to compute trigger probabilities.

        Returns:
            The loaded agenda.
        """
        with open(filename, "r") as file:
            d = yaml.load(file, Loader=yaml.FullLoader)
        agenda = cls._from_dict(d, policy_cls, state_probabilities_cls, trigger_probabilities_cls)

        # Load trigger detectors
        # Transition triggers
        trigger_names = list(agenda._transition_triggers.keys())
        detectors = trigger_detector_loader.load(agenda.name, trigger_names, snips_multi_engine=snips_multi_engine)

        for detector in detectors:
            #print('Transition TGD: {}'.format(detector))
            agenda.add_transition_trigger_detector(detector)

        # Kickoff triggers
        trigger_names = list(agenda._kickoff_triggers.keys())
        detectors = trigger_detector_loader.load(agenda.name, trigger_names, snips_multi_engine=snips_multi_engine)

        for detector in detectors:
            #print('Kickoff TGD: {}'.format(detector))
            agenda.add_kickoff_trigger_detector(detector)
        return agenda

    def __str__(self):
        states = str([st for st in self._states.keys()])
        kickoff_triggers = str([kt for kt in self._kickoff_triggers.keys()])
        transition_triggers = str([tt for tt in self._transition_triggers.keys()])
        actions = str([act for act in self._actions.keys()])
        policy = str(self._policy)

        return  '=== AGENDA ===\n'\
                '{}\n' \
                '=== STATES ===\n' \
                '{}\n' \
                '=== KICKOFF TRIGGERS ===\n'\
                '{}\n' \
                '=== TRANSITIONS TRIGGERS ===\n'\
                '{}\n' \
                '=== ACTIONS ===\n'\
                '{}\n' \
                '=== POLICY ===\n'\
                '{}\n' \
                .format(self._name, states, kickoff_triggers, transition_triggers, actions, policy)

