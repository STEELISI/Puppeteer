import abc
from typing import List, Dict, Tuple, Type, Set
import matplotlib.pyplot as plt
import random

from .agenda import Action, Agenda, AgendaState
from .logging import Logger
from .observation import IntentObservation, Observation, MessageObservation
from .extractions import Extractions

from .trigger_detectors.helper import extractor, keywords as victim_keywords

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

    def __init__(self, agendas: Dict[str, Dict[str, Agenda]]) -> None:
        """Initialize a new policy

        Args:
            agendas: The agendas used by the policy.
        """
        self._agendas = agendas

    @abc.abstractmethod
    def act(self, agenda_states: Dict[str, AgendaState], omit_keywords: List[str]) -> List[str]:
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

    def __init__(self, agendas: Dict[str, Dict[str, Agenda]]) -> None:
        """Initialize a new policy

        Args:
            agendas: The agendas used by the policy.
        """
        super(DefaultPuppeteerPolicy, self).__init__(agendas)
        # Policy metadata
        self._active_agendas: Dict[str, Dict[str, Agenda]] = {} # agendas that are able to start
        self._finished_agenda_names: Set[str] = set() # complete agenda(s) -- we got the extraction
        self._turns_without_progress = {t: -1 for t in agendas} #number of consecutive turns without progress
        self._times_made_current = {t: 0 for t in agendas}
        self._action_history: List[str] = []
        self._log = Logger()

    def act(self, agenda_states: Dict[str, AgendaState], omit_keywords: List[str]) -> List[str]:
        """"Picks zero or more appropriate actions to take, given the current state of the conversation.

        See documentation of this method in PuppeteerPolicy.

        Args:
            agenda_states: For each agenda (indexed by name), the AgendaState object holding the current belief about
                the state of the agenda, based on the latest observations -- the observations that this method is
                reacting to.

        Returns:
            A list of Action objects representing actions to take, in given order.
        """
        
        print("active", self._active_agendas.keys())
        print("finished", self._finished_agenda_names)

        ### check if there is an agenda that can be kicked off ###
        for topic in self._agendas:
            if topic == "react":
                continue
            if topic in self._finished_agenda_names:
                continue
            # Update the list of active agendas (if any agendas are kicked off)
            agenda_state = agenda_states[topic]
            if topic not in self._active_agendas and self._agendas[topic]["main"].policy.can_kick_off(agenda_state):
                self._log.add("{} is added to the list of active agendas".format(topic))
                print("added", topic)
                self._active_agendas[topic] = self._agendas[topic]

        ### get some actions from active agendas ###
        actions: List[str] = []
        for topic in self._active_agendas:
            print("active", topic)
            agenda_state = agenda_states[topic]
            print("trigger", agenda_state._transition_trigger_probabilities["main"]._probabilities)
            print("state prob", agenda_state._state_probabilities["main"]._probabilities)
            self._log.add("Picking actions for {}.".format(topic))
            if agenda_state._transition_trigger_probabilities["main"]._probabilities["push_back"]:
                action = self._active_agendas[topic]["push_back"].policy.pick_actions(agenda_state, self._action_history, omit_keywords)
            else:
                action = self._active_agendas[topic]["main"].policy.pick_actions(agenda_state, self._action_history, omit_keywords)
            # for act in action:
            #     print(agenda_name, act.text)
            actions += action

            done_flag = self._active_agendas[topic]["main"].policy.is_done(agenda_state)
            if done_flag:
                # We reach the terminus state of this agenda.
                # Will continue on other active agendas.
                self._log.add("{} is in a terminal state, so it will be stopped.".format(topic))
                self._finished_agenda_names.add(topic)

        ### delete finished agendas from active agendas ###
        for topic in self._finished_agenda_names:
            if topic in self._active_agendas.keys():
                del self._active_agendas[topic]
                print('delete', topic)

        ### if there is no action from active agendas ==> check if there is any action we can perform from reactive agendas ###
        if len(actions) == 0:
            max_t, max_prob = "", 0
            for t in agenda_states["react"]._agenda.keys():
                likelihood = agenda_states["react"]._kickoff_trigger_probabilities[t]._probabilities["kickoff"]
                if likelihood > max_prob:
                    max_t = t
                    max_prob = likelihood

            if max_prob >= 0.6:
                # we replay one of the actions
                sub_agenda = agenda_states["react"]._agenda[max_t]
                action_map = sub_agenda.action_map
                action_names = action_map["react"]
                chosen_actions = set()
                while len(chosen_actions) != len(action_names):
                    chosen_action_name = random.choice(action_names)
                    action = sub_agenda.action(chosen_action_name).text
                    if action not in self._action_history and not sub_agenda.policy.contain_keywords(action, omit_keywords):
                        # have not done this action before and did not contain omit keywords
                        actions.append(action)
                        break
                    chosen_actions.add(action)

        # Capitalize the first letter of each action
        actions = [a.capitalize() for a in actions]

        return actions

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

    def __init__(self, agendas: Dict[str, Dict[str, Agenda]],
                 policy_cls: Type[DefaultPuppeteerPolicy] = DefaultPuppeteerPolicy,
                 plot_state: bool = False,
                 enable_neural_dialog: bool = False) -> None:
        """Initialize a new Puppeteer.

        Args:
            agendas: List of agendas to be used by the Puppeteer.
            policy_cls: The policy delegate class to use.
            plot_state: If true, the updated state of the current agenda is plotted after each turn.
        """
        self._agendas = agendas
        self._policy = policy_cls(agendas)
        self._plot_state = plot_state
        self._enable_neural_dialog = enable_neural_dialog

        if self._plot_state:
            plt.ion()
            agenda_states = {}
            for t in agendas:
                fig, ax = plt.subplots()
                agenda_states[t] = AgendaState(t, agendas[t], fig, ax)
            self._agenda_states = agenda_states
        else:
            self._agenda_states = {t: AgendaState(t, agendas[t], None, None) for t in agendas}

        self._log = Logger()
        if self._enable_neural_dialog:
            #from .neural_dialog_engine import NeuralDialogEngine
            from .chatgpt_engine import ChatGPTEngine
            self._neural_dialog_engine = ChatGPTEngine()

        self._kickoff = False
        self._scams = self.get_scams()

    @property
    def log(self):
        """Returns a log string from the latest call to react().

        The log string contains information that is helpful in understanding the inner workings of the puppeteer -- why
        it acts the way it does based on the inputs, and what its internal state is.
        """
        return self._log.log

    def get_scams(self, scam_file="/nas/home/pcharnset/irc-user-study/puppeteer/scams.txt") -> List[str]:
        with open(scam_file) as f:
            scams = f.readlines()
            scams = [s.strip() for s in scams if s.strip()]
        return scams

    def is_kickoff(self, observations: List[Observation]) -> bool:
        if self._kickoff:
            return self._kickoff
        for o in observations:
            if isinstance(o, MessageObservation):
                if o.text.strip() in self._scams:
                    self._kickoff = True
                    break
        return self._kickoff

    def react(self, observations: List[Observation], old_extractions: Extractions) -> Tuple[List[str], Extractions]:
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

        if self.is_kickoff(observations) == False:
            print("Puppeteer: waiting to be kicked off ...")
            return ([], Extractions())

        self._log.begin("Inputs")
        self._log.begin("Observations")
        for o in observations:
            self._log.add(str(o))
        user_text = ' '.join([o.text.strip() for o in observations if isinstance(o, MessageObservation)])

        if self._enable_neural_dialog:
            self._neural_dialog_engine.append_chat_history(user_text, "attacker")
        self._log.end()
        self._log.begin("Old Extractions")
        for name in old_extractions.names:
            self._log.add(f"{name}: '{old_extractions.extraction(name)}'")
        self._log.end()

        new_extractions = extractor.get_extractions(user_text)
        self._log.begin("New Extractions")
        for name in new_extractions.names:
            self._log.add(f"{name}: '{new_extractions.extraction(name)}'")
        self._log.end()
        self._log.end()

        """
        if new extraction include information we are looking for in an agenda
        that we have not kicked off => mark that agenda as finished
        """
        for name in new_extractions.names:
            if name in self._policy._agendas and \
                    name not in self._policy._active_agendas.keys() and \
                        name not in self._policy._finished_agenda_names:
                self._policy._finished_agenda_names.add(name)

        active_agendas = self._policy._active_agendas
        finished_agenda_names = self._policy._finished_agenda_names
        self._log.begin("Update phase")
        for t in self._agenda_states:
            if t in finished_agenda_names:
                continue
            _ = self._agenda_states[t].update(observations, new_extractions, active_agendas)


        """
        we will not ask for the information we already have
        """
        omit_keywords = set()
        seen_extraction_name = list(set(old_extractions.names + new_extractions.names))
        for name in seen_extraction_name:
            if name in victim_keywords:
                for word in victim_keywords[name]:
                    omit_keywords.add(word)
        omit_keywords = list(omit_keywords)
        print("omit_keywords", omit_keywords)

        self._log.begin("Act phase")
        actions = self._policy.act(self._agenda_states, omit_keywords)

        """
        check if this is not a first turn,
        and there is no action to perform,
        and there is some agendas we have not kicked off
        """
        remaining_agenda_names = list(self._agendas.keys() - active_agendas.keys() - finished_agenda_names - {"react"})
        if len(remaining_agenda_names) > 0 and \
                len(self._policy._action_history) > 0 and len(actions) == 0:
            chosen_agenda_name = random.choice(remaining_agenda_names)
            intent_observation = IntentObservation()
            intent_observation.add_intent(chosen_agenda_name)
            _ = self._agenda_states[chosen_agenda_name].update([intent_observation], new_extractions, active_agendas)
            actions = self._policy.act(self._agenda_states, omit_keywords)
            
        self._log.end()

        self._log.begin("Outputs")
        self._log.begin("Actions")

        ### if neural dialog engine is enabled
        if self._enable_neural_dialog:
            bot_text = ' '.join([a.strip() for a in actions])
            ### debug ###
            #print("chat_hist: {}".format(self._neural_dialog_engine._chat_history))
            #print("chat_hist_ids: {}".format(self._neural_dialog_engine._chat_history_ids))
            #print("chat_hist_lens: {}".format(self._neural_dialog_engine._chat_history_lens))
            ### end ###
            if bot_text.strip():
                # we got some action texts from agendas => appending them to neural bot's chat history
                self._neural_dialog_engine.append_chat_history(bot_text.strip(), "victim")
            else:
                # we got nothing => turn to neural bot, generate a response
                neural_text = self._neural_dialog_engine.generate_response()
                actions = [neural_text]

        for a in actions:
            self._log.add(str(a))

        ### record past actions ###
        self._policy._action_history.extend(actions)

        self._log.end()
        self._log.end()

        return actions, new_extractions

    def reset(self, enable_neural_dialog=False):
        """ Restart Puppeteer: reset all agenda states to their initial values.
        """
        for agenda in self._agenda_states:
            # don't forget to reset the policy
            self._policy = DefaultPuppeteerPolicy(self._agendas)
            self._agenda_states[agenda].reset("main")

        if enable_neural_dialog:
            self._neural_dialog_engine.reset()

