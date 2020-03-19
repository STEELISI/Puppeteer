import abc
from typing import Any, Dict, List, Optional, Type

import matplotlib.pyplot as plt
import networkx as nx
import yaml

from extractions import Extractions
from observation import Observation
from trigger_detector import TriggerDetector, TriggerDetectorLoader


class State:
    """Class naming and describing a state in an agenda."""
    
    def __init__(self, name: str, description: str = "") -> None:
        self._name = name
        self._description = description
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description

    def to_dict(self) -> Dict[str, str]:
        return {"name": self._name, "description": self._description}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "State":
        return cls(d["name"], d["description"])


class Trigger:
    """Class naming and describing a trigger in an agenda."""

    def __init__(self, name: str, description: str = "") -> None:
        self._name = name
        self._description = description
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description

    def to_dict(self) -> Dict[str, str]:
        return {"name": self._name, "description": self._description}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Trigger":
        return cls(d["name"], d["description"])


class Action:
    """Class naming and describing an action used in an agenda.

    An action defines an exclusive_flag, indicating whether the action can be combined with other actions in the same
    time step, or if it is exclusive. It is the responsibility of the AgendaPolicy to make sure that this flag is
    actually complied with.

    The allowed_repeats limit is the maximum number of times the action may be performed in a conversation.
    """

    def __init__(self, name: str, text: str = "", exclusive_flag: bool = True, allowed_repeats: int = 2) -> None:
        self._name = name
        self._text = text
        self._exclusive_flag = exclusive_flag
        self._allowed_repeats = allowed_repeats
    
    def __repr__(self) -> str:
        return "%s: %s" % (self._name, self._text)
    
    def __str__(self) -> str:
        return repr(self)
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def text(self) -> str:
        return self._text

    @property
    def exclusive_flag(self) -> bool:
        return self._exclusive_flag
    
    @property
    def allowed_repeats(self) -> int:
        return self._allowed_repeats

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self._name, "text": self._text, "exclusive_flag": self._exclusive_flag,
                "allowed_repeats": self._allowed_repeats}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Action":
        return cls(d["name"], d["text"], d["exclusive_flag"], d["allowed_repeats"])


class AgendaState:
    """Holds trigger and state probabilities for an agenda, for an ongoing conversation.

    This is the main class holding agenda-level conversation state. Some state is also held in the PuppeteerPolicy.

    The update() method is the main method for updating the agenda level state, based on extractions and observations
    made since the last time step.
    """
    def __init__(self, agenda: "Agenda") -> None:
        self._agenda = agenda
        self._transition_trigger_probabilities = agenda.trigger_probabilities_cls(agenda, kickoff=False)
        self._kickoff_trigger_probabilities = agenda.trigger_probabilities_cls(agenda, kickoff=True)
        self._state_probabilities = agenda.state_probabilities_cls(agenda)
        self._pos = None

    @property
    def transition_trigger_probabilities(self) -> "TriggerProbabilities":
        return self._transition_trigger_probabilities

    @property
    def kickoff_trigger_probabilities(self) -> "TriggerProbabilities":
        return self._kickoff_trigger_probabilities

    @property
    def state_probabilities(self) -> "StateProbabilities":
        return self._state_probabilities

    def update(self,
               actions: List[Action],
               observations: List[Observation],
               old_extractions: Extractions
               ) -> Extractions:
        """Updates the agenda-level state.

        Updating the agenda level state, based on extractions and observations made since the last time step.
        """
        
        new_extractions = Extractions()

        extractions = self._kickoff_trigger_probabilities.update(observations, old_extractions)
        new_extractions.update(extractions)
        
        extractions = self._transition_trigger_probabilities.update(observations, old_extractions)
        new_extractions.update(extractions)

        self._state_probabilities.update(self._transition_trigger_probabilities, actions)
        
        return new_extractions

    def reset(self) -> None:
        self._state_probabilities.reset()

    def plot_state(self, fig: plt.Figure) -> None:
        g = nx.MultiDiGraph()
        
        # Add states
        for s in self._agenda.state_names:
            g.add_node(s)
        g.add_node('ERROR_STATE')
        
        for s0 in self._agenda.state_names:
            # Add transitions
            for s1 in self._agenda.transition_connected_state_names(s0):
                g.add_edge(s0, s1)
        
        # Color nodes according to probability map.
        color_transition = ['#fff0e6', '#ffe0cc', '#ffd1b3', '#ffc299', '#ffb380', '#ffa366', '#ff944d', '#ff8533',
                            '#ff751a', '#ff6600']
        color_map = []
        labels = {}
        for node in g:
            prob = self._state_probabilities.probability(node)
            labels[node] = "%s\np=%.2f" % (node, prob)

            prob = int(round(prob * 10))
            if prob > 0:
                prob = prob - 1

            if prob < len(color_transition):
                color_map.append(color_transition[prob])
            else:
                color_map.append('grey')
        
        # Draw
        plt.figure(fig.number)
        if self._pos is None:
            self._pos = nx.circular_layout(g)
        nx.draw(g, pos=self._pos, node_color=color_map, labels=labels)    


class TriggerProbabilities(abc.ABC):
    """Handles trigger probabilities for an ongoing conversation.

    Trigger probabilities represent all information in observations that is relevant for state transition.

    This is an abstract class. A concrete TriggerProbabilities subclass is also responsible for updating trigger
    probabilities, by implementing the update() method.
    """

    def __init__(self, agenda: "Agenda", kickoff: bool = False) -> None:
        if kickoff:
            self._trigger_detectors = agenda.kickoff_trigger_detectors
            self._probabilities = {tr.name: 0.0 for tr in agenda.kickoff_triggers}
        else:
            self._trigger_detectors = agenda.transition_trigger_detectors
            self._probabilities = {tr.name: 0.0 for tr in agenda.transition_triggers}
        self._non_trigger_prob = 1.0
    
    @property
    def probabilities(self) -> Dict[str, float]:
        # TODO Replace this with a per-trigger lookup method.
        return self._probabilities

    @property
    def non_trigger_prob(self) -> float:
        return self._non_trigger_prob

    @property
    def trigger_detectors(self) -> List[TriggerDetector]:
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
        """
        raise NotImplementedError()
        

class DefaultTriggerProbabilities(TriggerProbabilities):
    """Handles trigger probabilities for an ongoing conversation.

    This is the default TriggerProbabilities implementation. See class TriggerProbabilities for more details.
    """

    def update(self, observations: List[Observation], old_extractions: Extractions) -> Extractions:
        trigger_map: Dict[str, float] = {}
        non_trigger_probs: List[float] = []
        new_extractions = Extractions()
        
        for trigger_detector in self.trigger_detectors:
            (trigger_map_out, non_trigger_prob, extractions) = trigger_detector.trigger_probabilities(observations,
                                                                                                      old_extractions)

            new_extractions.update(extractions)
            non_trigger_probs.append(non_trigger_prob)
            for (trigger_name, p) in trigger_map_out.items():
                if trigger_name in self._probabilities:
                    if trigger_name not in trigger_map:
                        trigger_map[trigger_name] = p
                    elif trigger_map[trigger_name] < p:
                        trigger_map[trigger_name] = p

        if trigger_map:
            non_trigger_prob = 1.0 - max(trigger_map.values())
        else:
            non_trigger_prob = 1.0
        
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
        self._agenda = agenda
        self._probabilities = {s.name: 0.0 for s in agenda.states}
        self._probabilities["ERROR_STATE"] = 0.0
        self._probabilities[agenda.start_state.name] = 1.0

    @property
    def probabilities(self) -> Dict[str, float]:
        # TODO Replace use of this with per-trigger lookup.
        return self._probabilities

    def probability(self, state_name: str) -> float:
        return self._probabilities[state_name]

    def reset(self) -> None:
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
        """
        raise NotImplementedError()


class DefaultStateProbabilities(StateProbabilities):
    """Handles state probabilities for an ongoing conversation.

    This is the default StateProbabilities implementation. See class StateProbabilities for more details.
    """

    def update(self, trigger_probabilities: TriggerProbabilities, actions: List[Action]) -> None:
        # Note: This is essentially copied from puppeteer_base, with updated
        #       accesses to the agenda definition through self._agenda.

        # Check if the last of the actions taken "belongs" to this agenda. Earlier
        # actions may be the finishing actions of a deactivated agenda.
        if actions and not actions[-1] in self._agenda.actions:
            return
        
        current_probability_map = self._probabilities
        trigger_map = trigger_probabilities.probabilities
        non_event_prob = trigger_probabilities.non_trigger_prob

        # Set up our new prob map.
        new_probability_map = {}
        for st in self._agenda.state_names:
            new_probability_map[st] = 0.0
        new_probability_map['ERROR_STATE'] = 0.0

        # Chance we actually have an event:
        p_event = 1.0 - non_event_prob
        
        # For each state in the machine, do:
        for st in self._agenda.state_names:
            to_move = current_probability_map[st] * p_event
            new_probability_map[st] = max(0.05, current_probability_map[st] - to_move, new_probability_map[st])
                      
        # For each state in the machine, do:
        for st in self._agenda.state_names:
            to_move = current_probability_map[st] * p_event
            
            if round(to_move, 1) > 0.0:
                for event in trigger_map:
                    trans_prob = to_move * trigger_map[event]
                    if event in self._agenda.transition_trigger_names(st):
                        st2 = self._agenda.transition_end_state_name(st, event)
                        new_probability_map[st2] = new_probability_map[st2] + trans_prob   
                        # Decrease our confidence that we've had some problems following the script, previously.
                        # Not part of paper.
                        new_probability_map['ERROR_STATE'] = max(0.05, new_probability_map['ERROR_STATE'] - trans_prob)
                    else:
                        # XXX Downgrade our probabilities if we don't have an event that matches a transition?
                        # for this particular state.
                        # Not part of paper.
                        new_probability_map[st] = max(0.05, current_probability_map[st]-trigger_map[event])

                        # Up our confidence that we've had some problems following the script.
                        new_probability_map['ERROR_STATE'] = new_probability_map['ERROR_STATE'] + trans_prob

        self._probabilities = new_probability_map


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
        self._agenda = agenda

    @abc.abstractmethod
    def made_progress(self, state: AgendaState) -> bool:
        """Returns true if the agenda became reached a terminus state as a result of the latest observations."""
        raise NotImplementedError()

    @abc.abstractmethod    
    def is_done(self, state: AgendaState) -> bool:
        """Returns true if the agenda is likely in a terminus state."""
        raise NotImplementedError()

    @abc.abstractmethod    
    def can_kickoff(self, state: AgendaState) -> bool:
        """Returns true if the agenda is in a state where it can kick off."""
        raise NotImplementedError()

    @abc.abstractmethod    
    def pick_actions(self, state: AgendaState, action_history: List[Action],
                     turns_without_progress: int) -> List[Action]:
        """Picks zero or more appropriate actions to take, given the current state of the agenda."""
        raise NotImplementedError()

    @abc.abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, d: Dict[str, Any], agenda: "Agenda") -> "AgendaPolicy":
        raise NotImplementedError()
        
        
class DefaultAgendaPolicy(AgendaPolicy):
    """Handles agenda-level decisions about behavior.

    This is the default AgendaPolicy implementation. See AgendaPolicy documentation for further details.
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
        super(DefaultAgendaPolicy, self).__init__(agenda)
        self._reuse = reuse                        # TODO When to use?
        self._max_transitions = max_transitions    # TODO When to use?
        self._absolute_accept_thresh = absolute_accept_thresh
        self._min_accept_thresh_w_differential = min_accept_thresh_w_differential
        self._accept_thresh_differential = accept_thresh_differential
        self._kickoff_thresh = kickoff_thresh

    def to_dict(self) -> Dict[str, Any]:
        field_names = ["_reuse", "_max_transitions", "_absolute_accept_thresh",
                       "_min_accept_thresh_w_differential",
                       "_accept_thresh_differential", "_kickoff_thresh"]
        d = {f[1:]: getattr(self, f) for f in field_names}
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any], agenda: "Agenda") -> "DefaultAgendaPolicy":
        return cls(agenda, d["reuse"], d["max_transitions"],
                   d["absolute_accept_thresh"],
                   d["min_accept_thresh_w_differential"],
                   d["accept_thresh_differential"],
                   d["kickoff_thresh"])

    def made_progress(self, state: AgendaState) -> bool:
        non_event_probability = state.transition_trigger_probabilities.non_trigger_prob
        error_state_probability = state.state_probabilities.probabilities["ERROR_STATE"]
        return non_event_probability <= 0.4 and error_state_probability <= .8

    def is_done(self, state: AgendaState) -> bool:
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

    def can_kickoff(self, state: AgendaState) -> bool:
        non_kickoff_probability = state.kickoff_trigger_probabilities.non_trigger_prob
        return 1.0 - non_kickoff_probability >= self._kickoff_thresh

    def pick_actions(self, state: AgendaState, action_history: List[Action],
                     turns_without_progress: int) -> List[Action]:
        actions_taken: List[Action] = []
        
        # Action map - maps states to a list of tuples of:
        # (action_name, function, arguments, 
        #  boolean to indicate if this an exclusive action that cannot be used
        #  with other actions, number of allowed repeats for this action)
        if turns_without_progress == 0:
            action_map = self._agenda.action_map
        else:
            action_map = self._agenda.stall_action_map
            
        # Work over the most likely state, to least likely, taking the first
        # actions we are allowed to given repeat allowance & exclusivity.
        # for state by decreasing probabilities that we're in that state:
        done = False
        for st in {k: v for k, v in sorted(state.state_probabilities.probabilities.items(), key=lambda item: item[1], reverse=True)}:
            # XXX Maybe need to check likelihood.
            if st in action_map:
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
                            # TODO Skip done flag and return here?
                            done = True
                            break
            if done:
                break

        return actions_taken


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
        # Do we want to store trigger detectors somewhere else?
        # Separate domain (graph + actions(?)) from detection and policy logic?
        self._kickoff_trigger_detectors: List[TriggerDetector] = []
        self._transition_trigger_detectors: List[TriggerDetector] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def policy(self) -> AgendaPolicy:
        return self._policy

    @property
    def states(self) -> List[State]:
        return list(self._states.values())

    @property
    def state_names(self) -> List[str]:
        return list(self._states.keys())

    @property
    def kickoff_triggers(self) -> List[Trigger]:
        return list(self._kickoff_triggers.values())

    @property
    def transition_triggers(self) -> List[Trigger]:
        return list(self._transition_triggers.values())

    @property
    def actions(self) -> List[Action]:
        return list(self._actions.values())

    def action(self, action_name: str) -> Action:
        return self._actions[action_name]

    @property
    def action_map(self) -> Dict[str, List[str]]:
        # TODO Replace this method with per-state lookup
        return self._action_map

    @property
    def stall_action_map(self) -> Dict[str, List[str]]:
        # TODO Replace this method with per-state lookup
        return self._stall_action_map

    @property
    def start_state(self) -> State:
        if self._start_state_name is not None:
            return self._states[self._start_state_name]
        else:
            raise Exception("Start state name undefined.")

    @property
    def terminus_states(self) -> List[State]:
        return [self._states[s] for s in self._terminus_names]

    @property
    def terminus_names(self) -> List[str]:
        return list(self._terminus_names)

    def transition_trigger_names(self, state_name: str) -> List[str]:
        return list(self._transitions[state_name].keys())

    def transition_end_state_name(self, state_name: str, trigger_name: str) -> str:
        return self._transitions[state_name][trigger_name]

    def transition_connected_state_names(self, state_name: str) -> List[str]:
        return list(set(self._transitions[state_name].values()))

    @property
    def state_probabilities_cls(self) -> Type[StateProbabilities]:
        return self._state_probabilities_cls

    @property
    def trigger_probabilities_cls(self) -> Type[TriggerProbabilities]:
        return self._trigger_probabilities_cls

    @property
    def kickoff_trigger_detectors(self) -> List[TriggerDetector]:
        return list(self._kickoff_trigger_detectors)

    @property
    def transition_trigger_detectors(self) -> List[TriggerDetector]:
        return list(self._transition_trigger_detectors)

    def to_dict(self) -> Dict[str, Any]:
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
        # TODO Anything from belief manager?
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any],
                  policy_cls: Type[AgendaPolicy],
                  state_probabilities_cls: Type[StateProbabilities],
                  trigger_probabilities_cls: Type[TriggerProbabilities]) -> "Agenda":
        obj = cls(d["name"], policy_cls=policy_cls, state_probabilities_cls=state_probabilities_cls,
                  trigger_probabilities_cls=trigger_probabilities_cls)
        # Special handling of policy
        d_policy = d["policy"]
        del d["policy"]
        obj._policy = policy_cls.from_dict(d_policy, obj)
        # Restore all fields, as stored in dict
        for (name, value) in d.items():
            setattr(obj, "_" + name, value)

        # Replace with objects, where appropriate.
        def from_dict(dict_list: List[Dict[str, Any]], new_cls: Type[object]) -> Dict[str, Any]:
            obj_dict = {}
            for dd in dict_list:
                new_obj = new_cls.from_dict(dd)
                obj_dict[new_obj.name] = new_obj
            return obj_dict
        obj._states = from_dict(obj._states, State)
        obj._kickoff_triggers = from_dict(obj._kickoff_triggers, Trigger)
        obj._transition_triggers = from_dict(obj._transition_triggers, Trigger)
        obj._actions = from_dict(obj._actions, Action)
        return obj

    def add_state(self, state: State) -> None:
        self._states[state.name] = state
        self._action_map[state.name] = []
        self._stall_action_map[state.name] = []
        self._transitions[state.name] = {}

    def set_start_state(self, state_name: str) -> None:
        self._start_state_name = state_name

    def add_terminus(self, state_name: str) -> None:
        self._terminus_names.append(state_name)

    def add_transition_trigger(self, trigger: Trigger) -> None:
        self._transition_triggers[trigger.name] = trigger

    def add_kickoff_trigger(self, trigger: Trigger) -> None:
        self._kickoff_triggers[trigger.name] = trigger

    def add_transition(self, start_state_name: str, trigger_name: str, end_state_name: str) -> None:
        self._transitions[start_state_name][trigger_name] = end_state_name
    
    def add_action_for_state(self, action: Action, state_name: str) -> None:
        self._actions[action.name] = action
        self._action_map[state_name].append(action.name)
    
    def add_stall_action_for_state(self, action: Action, state_name: str) -> None:
        self._actions[action.name] = action
        self._stall_action_map[state_name].append(action.name)

    def add_transition_trigger_detector(self, trigger_detector: TriggerDetector) -> None:
        self._transition_trigger_detectors.append(trigger_detector)

    def add_kickoff_trigger_detector(self, trigger_detector: TriggerDetector) -> None:
        self._kickoff_trigger_detectors.append(trigger_detector)

    def store(self, filename: str) -> None:
        with open(filename, "w") as file:
            yaml.dump(self.to_dict(), file, default_flow_style=False, sort_keys=False)

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
        """
        with open(filename, "r") as file:
            d = yaml.load(file)
        agenda = cls.from_dict(d, policy_cls, state_probabilities_cls, trigger_probabilities_cls)
        # Load trigger detectors
        # Transition triggers
        trigger_names = list(agenda._transition_triggers.keys())
        detectors = trigger_detector_loader.load(agenda.name, trigger_names, snips_multi_engine=snips_multi_engine)

        for detector in detectors:
            agenda.add_transition_trigger_detector(detector)
        # Kickoff triggers
        trigger_names = list(agenda._kickoff_triggers.keys())
        detectors = trigger_detector_loader.load(agenda.name, trigger_names, snips_multi_engine=snips_multi_engine)

        for detector in detectors:
            agenda.add_kickoff_trigger_detector(detector)
        return agenda
