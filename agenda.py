import abc
from typing import List, Map

import numpy as np

from files import dirnames_from_dirname
from nlu import SnipsManager
from puppeteer import Action, Belief, MultiBelief, Observation, Policy, Puppeteer


class AgendaState:
    # Class naming and describing a state in an agenda.
    
    # TODO String is enough? No, probably has description, at least.
    def __init__(self, name: str, description: str=""):
        self._name = name
        self._description = description
    
    @property
    def name(self):
        return self._name
    
    @property
    def description(self):
        return self._description


class AgendaTrigger:
    # Class naming and describing a trigger in an agenda.

    def __init__(self, name: str, description: str=""):
        self._name = name
        self._description = description
    
    @property
    def name(self):
        return self._name
    
    @property
    def description(self):
        return self._description

        

class AgendaAction(Action):
    # Class naming and describing an action used in an agenda.

    # Note: Agenda implementation considers actions to be equal (for purposes
    # of updating the state probabilities and checking how many times an action
    # has been performed).

    def __init__(self, name: str, description: str="", exclusive_flag=True, allowed_repeats=2):
        self._name = name
        self._description = description
        self._exclusive_flag = exclusive_flag
        self._allowed_repeats = allowed_repeats
    
    @property
    def name(self):
        return self._name
    
    @property
    def description(self):
        return self._description

    @property
    def exclusive_flag(self):
        return self._exclusive_flag
    
    @property
    def allowed_repeats(self):
        return self._allowed_repeats
    


class Agenda:
    # Class defining all properties of an agenda, but not keeping track of a
    # conversation state. A single Agenda object can be reused between many
    # conversations (Puppeteers). An Agenda object may use language models (from
    # the nlu module) through its trigger detectors, but language models are
    # shared between agendas using the same model.
    
    # TODO Setters and getters for all members

    def __init__(self, name: str):
        self._name = name
        self._states = {}
        self._transition_triggers = {}
        self._kickoff_triggers = {}
        self._transitions = {}
        self._start_state_name = None
        self._terminus_names = []
        self._actions = {}
        self._action_map = {}
        self._stall_action_map = {}
        # Do we want to store trigger detectors somewhere else?
        # Separate domain (graph + actions(?)) from detection and policy logic?
        self._kickoff_trigger_detectors = []
        self._transition_trigger_detectors = []
        # Parameters, from Turducken
        self._reuse = False               # TODO When to use?
        self._self_max_transitions = 5    # TODO When to use?
        self._absolute_accept_thresh = 0.6
        self._min_accept_thresh_w_differential = 0.2
        self._accept_thresh_differential = 0.1
        # Parameters, new
        self._kickoff_thresh = 0.5

    def add_state(self, state: AgendaState):
        self._states[state.name] = state
        self._actions[state.name] = {}
        self._stall_actions[state.name] = {}
        self._transitions[state.name] = {}

    def set_start_state(self, state_name: str):
        self._start_state_name = state_name

    def add_terminus(self, state_name: str):
        self._terminus_names.append(state_name)

    def add_transition_trigger(self, trigger: AgendaTrigger):
        self._transition_triggers[trigger.name] = trigger

    def add_kickoff_trigger(self, trigger: AgendaTrigger):
        self._kickoff_triggers[trigger.name] = trigger

    def add_transition(self, start_state_name: str, trigger_name: str, end_state_name: str):
        self._transitions[start_state_name][trigger_name] = end_state_name
    
    def add_action_for_state(self, action: AgendaAction, state_name: str):
        self._action_map[state_name][action.name] = action
        self._actions[action.name] = action
    
    def add_stall_action_for_state(self, action: AgendaAction, state_name: str):
        self._stall_action_map[state_name][action.name] = action
        self._actions[action.name] = action

    def add_transition_trigger_detector(self, trigger_detector: AgendaTriggerDetector):
        self._transition_trigger_detectors.append(trigger_detector)

    def add_kickoff_trigger_detector(self, trigger_detector: AgendaTriggerDetector):
        self._kickoff_trigger_detectors.append(trigger_detector)

    def load(self, file):
        raise NotImplementedError()
        
    # Methods used by the policy

    def made_progress(self, belief: AgendaBelief) -> bool:
        return belief.non_event_prob <= 0.4 and belief.error_state_prob <= .8

    def is_done(self, belief: AgendaBelief) -> bool:
        shortcut = False
        best = None
            
        # For state by decresing probabilities that we're in that state. 
        # TODO Probably simpler: just look at best and second-best state
        # TODO Don't access probability map directly
        probability_map = belief._probability_map
        sorted_states = {k: v for k, v in sorted(probability_map.items(), key=lambda item: item[1], reverse=True)}
        for (rank, st) in enumerate(sorted_states):
            if st in self._terminus: 
                # If this is an accept state, we can set our best exit candidate.
                if rank == 0 and probability_map[st] >= self._absolute_accept_thresh:
                    return True
                elif rank == 0 and probability_map[st] >= self._min_accept_thresh_w_differential:
                    best = probability_map[st]
            # If we have an exit candidate, 
            if best != None and rank == 1:
                if best - probability_map[st] >= self._accept_thresh_differential:
                    return True
        return False

    def can_kickoff(self, belief: AgendaBelief) -> bool:
        return 1.0 - belief.non_kickoff_prob(self._name) > self._kickoff_thresh

    def pick_actions(self, belief: AgendaBelief, action_history: List[Action], turns_without_progress: int) -> List[Action]:
        current_probability_map = belief._probability_map
        past_action_list = action_history
        
        actions_taken = []
        
        # Action map - maps states to a list of tuples of:
        # (action_name, function, arguments, 
        #  boolean to indicate if this an exclusive action that cannot be used
        #  with other actions, number of allowed repeats for this action)
        if turns_without_progress == 0:
            action_map = self._action_map
        else:
            action_map = self._stall_action_map
            
        # Work over the most likely state, to least likely, taking the first
        # actions we are allowed to given repeat allowance & exclusivity.
        # for state by decresing probabilities that we're in that state:
        done = False
        for st in {k: v for k, v in sorted(current_probability_map.items(), key=lambda item: item[1], reverse=True)}:
            # XXX Maybe need to check likeyhood.
            if st in action_map:
                for action in action_map[st]:
                    exclusive_flag = action.exclusive_flag
                    allowed_repeats = action.allowed_repeats
                    
                    num_times_action_was_used = past_action_list.count(action)
                    
                    if num_times_action_was_used < allowed_repeats:
                        if exclusive_flag and actions_taken:
                            # Can't do an exclusive action if a non-exclusive
                            # action is already taken.
                            continue

                        function(arguments, analytics_reply_node=analytics_reply_node)
                        actions_taken.append(action_name)
                        if exclusive_flag:
                            # TODO Skip done flag and return here?
                            done = True
                            break
            if done:
                break
        
        return actions_taken


class AgendaTriggerDetector(abc.ABC):
    # A trigger detector can take observations and return probabilities that
    # its triggers are "seen" in the observation. A trigger detector has a set
    # of triggers it is looking for.

    @abc.abstractmethod
    def trigger_probabilities(self, observations: List[Observation]) -> Map[str, float]:
        raise NotImplementedError()


class AgendaSnipsTriggerDetector(AgendaTriggerDetector):
    # A trigger detector using one or more Snips engines to detect triggers in
    # observations.

    def __init__(self, path: str, nlp, multi_engine=False):
        # path: Path to the root of the training data. 
        self._engines = []
        self._trigger_names = None

       # TODO Is there a Snips convention for how to do store its training data?

       # Create our Snips engine or engines.
        if multi_engine:
            dirs = dirnames_from_dirname(path)
            for dir in dirs:
                self._engines.append(SnipsManager.engine(dir, nlp))
        else:
            self._engines.append(SnipsManager.engine(path, nlp))

    def trigger_probabilities(self, observations: List[Observation]) -> Map[str, float]:
        texts = []
        for observation in observations:
            if isinstance(observation, MessageObservation):
                texts.append(observation.text)
        text = "\n".join(texts)

        trigger_map = {}
        for engine in self._engines:
            snips_results = engine.detect(text)
                        
            for intent, p, sen in snips_results:
                if 'NOT' not in intent:
                    trigger_name = intent + '_intent'
                    if intent + '_intent' not in trigger_map:
                        trigger_map[trigger_name] = p
                    elif trigger_map[trigger_name] < p:
                        trigger_map[trigger_name] = p
        return trigger_map



class AgendaBelief(Belief):
    # Global belief, over all agendas. There will be one AgendaBelief object
    # per conversation.
    # This abstract class just defines queries to be handled -- queries used by
    # an AgendaPolicy to make its decisions.
    # Note that AgendaBelief also is responsible for detecting kickoff
    # conditions, based on trigger detectors set in the Agenda.

    @abc.abstractmethod
    def kickoff_probability(self, agenda_name: str, trigger_name: str):
        # Probability that we saw a kickoff trigger in the last observations.
        return self._probability_map[trigger_name]

    @abc.abstractmethod
    def non_kickoff_probability(self, agenda_name: str):
        return self._non_kickoff_prob

    @abc.abstractmethod
    def error_state_prob(self, agenda_name: str):
        return self._probability_map["ERROR_STATE"]

    @abc.abstractmethod
    def non_event_probability(self, agenda_name: str):
        return self._non_event_prob


class AgendaTurduckenBelief(AgendaBelief):
    # A concrete implementation of AgendaBelief, using the same implementation
    # of belief update as used in Turducken. This class needs a new name.

    def __init__(self, agendas: List[Agenda]):
        self._state_beliefs = {a.name: StateBelief(a) for a in agendas}
        self._kickoff_beliefs = {a.name: KickoffBelief(a) for a in agendas}
        self.reset()
    
    def kickoff_probability(self, agenda_name: str, trigger_name: str):
        return self._kickoff_beliefs[agenda_name]._probability_map[trigger_name]

    def non_kickoff_probability(self, agenda_name: str):
        return self._kickoff_beliefs[agenda_name]._non_kickoff_prob

    def error_state_prob(self, agenda_name: str):
        return self._state_beliefs[agenda_name]._probability_map["ERROR_STATE"]

    def non_event_probability(self, agenda_name: str):
        return self._state_beliefs[agenda_name]._non_event_prob

    def reset(self):
        for b in self._state_beliefs.values():
            b.reset()
        for b in self._kickoff_beliefs.values():
            b.reset()
    
    def update(self, actions: List[Action], observations: List[Observation]):
        for b in self._state_beliefs.values():
            b.update(actions, observations)
        for b in self._kickoff_beliefs.values():
            b.update(actions, observations)


class KickoffBelief(Belief):
    # This class implements belief in a kickoff trigger for an agenda being
    # present in the latest observations. Note that the belief is updated
    # regardless of which agenda selected the last set of actions.
    
    # TODO Lots of overlap with StateBelief
    def __init__(self, agenda: Agenda):
        self._agenda = agenda
        self._trigger_detectors = agenda._kickoff_trigger_detectors.copy()
        self._probability_map = None
        self._non_kickoff_prob = None

    def update(self, actions: List[Action], observations: List[Observation]):
        # 1. Compute normalized trigger probabilities, including probability
        #    of no trigger.
        trigger_map = {}
        
        for trigger_detector in self._trigger_detectors:
            trigger_map_out = trigger_detector.trigger_probabilities(observations)
            for (trigger_name, p) in trigger_map_out.items():
                if trigger_name not in trigger_map:
                    trigger_map[trigger_name] = p
                elif trigger_map[trigger_name] < p:
                    trigger_map[trigger_name] = p

        max_probability = max(trigger_map.values())                      
        non_event_prob = 1.0 - max_probability
        
        # We may have multiple snips engines and we may also have multiple
        # intents in the input text.  Ultimately, we want the probabilities
        # that we had no_event + events we did have an event = 1.
        # This is playing fast an loose with the probabilty tracking, but
        # still encorporates weights across multiple engines/events.
        sum_total = sum(trigger_map.values()) + non_event_prob
        
        non_event_prob = non_event_prob / sum_total
        for intent in trigger_map:
            trigger_map[intent] = trigger_map[intent] / sum_total
        
        self._non_kickoff_prob = non_event_prob

    def reset(self):
        self._probability_map = {tr.name: 0.0 for tr in self._agenda._kickoff_triggers.keys()}
        self._non_kickoff_prob = 1.0


class StateBelief(Belief):
    # This class implements our belief in being in different states for an
    # agenda, based on the latest observations. Note that the belief is updated
    # only for the agenda that selected the last set of actions. Intended to be
    # equivalent with belief update used by Turducken, at least in the case
    # where a Turducken agnda only uses Snips triggers. (This belief update does
    # not distinguish between Snips triggers and other types of triggers, so
    # normalization of probabilities will be slightly different from the case
    # where Turducken uses other triggers.) 
    
    def __init__(self, agenda: Agenda):
        self._agenda = agenda
        self._trigger_detectors = agenda._transition_trigger_detectors.copy()
        self._probability_map = None
        self._non_event_prob = None

    def reset(self):
        self._probability_map = {st: 0.0 for st in self._agenda._states.keys()}
        self._probability_map[self._agenda._start_state] = 1.0
        self._probability_map['ERROR_STATE'] = 0.0
        self._non_even_prob = 0.0

    def update(self, actions: List[Action], observations: List[Observation]):
        # 0. Check if any of the actions taken "belong" to this agenda.
        #    TODO What if actions are shared between agendas?
        if not any([a in self._agenda._actions.values() for a in actions]):
            return
        
        # 1. Compute normalized trigger probabilities, including probability
        #    of no trigger.
        trigger_map = {}
        
        for trigger_detector in self._trigger_detectors:
            trigger_map_out = trigger_detector.trigger_probabilities(observations)
            for (trigger_name, p) in trigger_map_out.items():
                if trigger_name not in trigger_map:
                    trigger_map[trigger_name] = p
                elif trigger_map[trigger_name] < p:
                    trigger_map[trigger_name] = p

        max_probability = max(trigger_map.values())                      
        non_event_prob = 1.0 - max_probability
        
        # We may have multiple snips engines and we may also have multiple
        # intents in the input text.  Ultimately, we want the probabilities
        # that we had no_event + events we did have an event = 1.
        # This is playing fast an loose with the probabilty tracking, but
        # still encorporates weights across multiple engines/events.
        sum_total = sum(trigger_map.values()) + non_event_prob
        
        non_event_prob = non_event_prob / sum_total
        for intent in trigger_map:
            trigger_map[intent] = trigger_map[intent] / sum_total
        
        self._non_event_prob = non_event_prob

        # 2. Update state probabilities
        self._probability_map = self._run_transition_probabilities(self._probability_map, trigger_map, non_event_prob)
    
    def _run_transition_probabilities(self, current_probability_map, trigger_map, non_event_prob):
        # Note: This is essentially copied from puppeteer_base, with updated
        #       accesses to the agenda definition through self._agenda.
        """
        Input:
          - current_probability_map: where we left off with the state machine before.
          - trigger_map: events we think were triggered *and* how sure we are that they triggered. Dictionary of dict[event] = prob
          - non_event_prob: probability that we had 0 events.
        Returns:
            - Probability map after new input (e.g. a dicitonary of [state_name] = prob_we_are_in_this_state)
        """

        # Set up our new prob map.
        new_probability_map = {}
        for st in self._agenda._states:
            new_probability_map[st] = 0.0
        new_probability_map['ERROR_STATE'] = 0.0

        # Chance we actually have an event:
        p_event = 1.0 - non_event_prob
        
        # For each state in the machine, do:
        for st in self._agenda._states:
            to_move = current_probability_map[st] * p_event
            new_probability_map[st] = max(0.05, current_probability_map[st] - to_move, new_probability_map[st])
                      
        # For each state in the machine, do:
        for st in self._agenda._states:
            to_move = current_probability_map[st] * p_event
            
            if round(to_move,1) > 0.0:
                for event in trigger_map:
                    trans_prob = to_move * trigger_map[event]
                    if event in self._agenda._transitions[st]:
                        st2 = self._agenda._transitions[event]
                        new_probability_map[st2] = new_probability_map[st2] + trans_prob   
                        #_LOGGER.debug("Updating %s prob to %.3f" % (st2, new_probability_map[st2]))
                        # Decrease our confidence that we've had some problems following the script, previously.
                        # Not part of paper.
                        new_probability_map['ERROR_STATE'] = max(0.05, new_probability_map['ERROR_STATE'] - trans_prob)
                    else:
                        # XXX Downgrade our probabilites if we don't have an event that matches a transition?
                        # for this particular state.
                        # Not part of paper.
                        new_probability_map[st] = max(0.05, current_probability_map[st]-trigger_map[event])
                        #_LOGGER.debug("Updating %s prob with downgrade to %.3f" % (st, new_probability_map[st]))                         
                        
                        # Up our confidence that we've had some problems following the script.
                        new_probability_map['ERROR_STATE'] = new_probability_map['ERROR_STATE'] + trans_prob
        
        #for state in new_probability_map:
        #    _LOGGER.info("Prob at end for %s: %.2f" % (state, new_probability_map[state]))
        return new_probability_map



class AgendaPolicy(Policy):
    # The action-selection policy of a Puppeteer, one object per conversation.
    # Essentially the same as run_puppeteer(), but not fully ported yet.
    
    def __init__(self, agendas: List[Agenda]):
        self._agendas = agendas
        self._current_agenda = None

    def act(self, belief: AgendaBelief) -> List[AgendaAction]:
        # TODO Mostly based on implementation from engine.py

        # - Kickoff temporarily handled by letting the agendas peek at the
        #   observations. Later fix to have states for kickoff. Or, just
        #   introduce these states?

        listen_in = False
        
        actions_taken = []
        action_arguments = []  # TODO Should be part of the Action
        
        agenda = self._current_agenda
        last_agenda = None
        
        if agenda is not None:
            # Update agenda state based on message.
            # What to handle in output?
            progress_flag = agenda.made_progress(belief)
            done_flag = progress_flag and agenda.is_done(belief)
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
                # TODO Reset belief for current agenda?
                self._current_agenda = None
                # TODO Do we really want to exit here?
                return ([], [])
            else:
                # Run and see if we get some actions. 
                actions = agenda.pick_actions(belief, self._action_history, turns_without_progress)
                self._action_history.extend(actions)
                
                if not done_flag:
                    # Keep going with this agenda.
                    return actions
                else:
                    # We inactivate this agenda. Will choose a new agenda
                    # in the main while-loop below.
                    # We're either done with the agenda, or had too many turns
                    # without progress.
                    # Do last action if there is one.
                    # TODO Reset belief for current agenda?
                    self._current_agenda = None
                    last_agenda = agenda
                
        # See if any incative agendas can be kicked off based on the
        # incoming message.
        # we randomize agendas since we might have multiple trigger - this way we get more agendas
        # out there.
        agenda = None
        
        for agenda_name in np.random.permutation(list(machines.keys())):
            machine = machines[agenda_name]
    
            current_probability_map, kick_off_condition, done_flag = machine.check_triggers_for_kickoff(message_text, msg_node=msg_node)
            # Activate the agenda..
            puppeteer_state.activate_agenda(agenda_name, current_probability_map)
            agenda = puppeteer_state.get_agenda(agenda_name)
    
            if agenda_name == last_agenda or puppeteer_state.get_times_made_current(agenda_name) > 1:
                if agenda != None:
                    puppeteer_state.deactivate_agenda(agenda)
                continue
    
    
            # IF we kicked off, make this our active agenda, do actions and return.
            if kick_off_condition:
                # Successfully kicked this off.
                # Make this our current agenda.            
                puppeteer_state.set_current_agenda(agenda)
                
                # Do first action.
                (actions_taken, action_arguments) = machine.do_actions(0, current_probability_map, [], analytics_reply_node=analytics_reply_node, return_arguments=True)
                if actions_taken:
                    if agenda != None:
                        agenda.add_actions(actions_taken)
                    else:
                        # TODO Can we really end up here?
                        pass
                else:
                        # New agenda couldn't do anything.
                        pass
                if done_flag:
                    puppeteer_state.deactivate_agenda(agenda)
                return (actions_taken, action_arguments)
            else:
                if agenda != None:
                    puppeteer_state.deactivate_agenda(agenda)
            
        # We failed to take action with an old agenda
        # and failed to kick off a new agenda. We have nothing.
        return (actions_taken, action_arguments)


class AgendaPuppeteer(Puppeteer):
    # Main Puppeteer class for an agendas-based conversation.
    def __init__(self, agendas: List[Agenda]):
        super(AgendaPuppeteer, self).__init__(AgendaPolicy(agendas),
                                              AgendaTurduckenBelief(agendas))




