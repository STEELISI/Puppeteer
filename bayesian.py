

class MultiBelief(abc.ABC):

    def __init__(self, beliefs: List[Belief]):
        self._beliefs = beliefs
    
    def update(self, actions: List[Action], observations: List[Observation]):
        for belief in self._beliefs:
            belief.update(actions, observations)


class Belief(abc.ABC):
    # TODO Rigth now, this ties three things together in one class:
    # - The belief
    # - The observation probability function
    # - The transition probability function
    # Could turn the PFs into "plugin" classes.
    def __init__(self, belief: np.array):
        self._belief = belief
        
    @abc.abstracmethod
    def update(self, actions: List[Action], observations: List[Observation]):
        raise NotImplementedError()



class BayesianBelief(Belief):
    # TODO Rigth now, this ties three things together in one class:
    # - The belief
    # - The observation probability function
    # - The transition probability function
    # Could turn the PFs into "plugin" classes.
    def __init__(self, belief: np.array):
        self._belief = belief
        
    @abc.abstracmethod
    def observation_probability(self, actions: List[Action], observation: Observation) -> np.array:
        raise NotImplementedError()
    
    @abc.abstracmethod
    def transition_probability(self, actions: List[Action], observation: Observation) -> np.array:
        # Note that this is based on the current state belief.
        raise NotImplementedError()
    
    def update(self, actions: List[Action], observation: Observation):
        prior = self.transition_probability(actions)
        posterior = prior * self.observation_probability(actions, observation)
        self._belief = posterior / np.sum(posterior)


class AgendaBayesianBelief(Belief):
    
    def __init__(self, agenda: Agenda):
        self._agenda = agenda
        # Convention for now: first state is start state.
        # TODO: Do we want to allow several possible start states?
        # Belief is nbr_states-by-(nbr_triggers+1)
        # Last trigger component is "no-trigger".
        nbr_states = len(agenda._states)
        nbr_triggers = len(agenda._triggers)
        belief = np.zeros((nbr_states, nbr_triggers + 1))
        belief[0, -1] = 1.0
        # Compile into a standard representation?
        self._kickoff_belief = 0.0
        super(AgendaBelief, self).__init__(belief)

    def observation_probability(self, actions: List[Action], observation: Observation) -> np.array:
        # Make observation update, posterior probability.
        # TODO Better to split this out into separate class?
        triggers = self._agenda._triggers
        p = np.zeros(len(triggers + 1))
        for (i, trigger) in enumerate(triggers):
            op = trigger.observation_probabilities(observation)
            # TODO Handle op == 1.0
            # Note: This assumes that we have exactly one trigger
            p[i] = op / (1.0 - op)
        p[-1] = 1.0
        # Normalize
        p = p / np.sum(p)
        nbr_states = len(self._agenda._states)
        return np.tile(p, (nbr_states, 1))
    
    def transition_probability(self, actions: List[Action], observation: Observation) -> np.array:
        # TODO Equal probability for each transition, if the action is from
        # this agenda.
        # TODO What identifies an action? Equals by name or by object?
        raise NotImplementedError()

    # Should this live somewhere else? 
    def made_progress(self) -> bool:
        p_progress = 1.0 - np.sum(self._belief[:, -1])
        # TODO compare with threshold parameter.



# Questions:
# - How to handle situation with multiple names that we might have heard for
#   the attacker. Will have mutual exclusivity.
#   - Solve by storing them all in a partial state? Yes! And expand state when
#     a new name occurs.

# class State:
#     # Full state representation
#     # Collection of partial, probabilistically independent states
#     # Parts are named
#     # Can be variable-length
#     # State can consist of
#     # - State of mind of the attacker.
#     # - Information that we have gathered.
#     pass


# class PartialState:
#     # Can have a probability (belief) distributuion over this.
#     pass


 
    
# class StateBeliefs:
#     # Belief (probability) distribution over states.
#     pass


# class PartialStateBeliefs:
#     # Belief (probability) distribution over partial states.
#     # This can model dependencies that can be represented systematically.
#     # What about a set of mutually exclusive names? Better represented by a
#     # single multi-valued discrete var?
#     pass


# class TransitionModel:
#     # Models the T. Will consist of partial models.
#     # Produces a prior belief in the state given the previous state.
#     # A partial model updates a partial state based on the same previous part.
#     # - Can be seen as a matrix from old to new. May be non-quadratic if the
#     #   action introduces state. Does it ever?
#     # This model is strongly connected to a PartialState, so it may be linked
#     # to from / owned by the partial state.
#     # May take action history into account.
#     # May do the same pre-application state representation modify as the
#     # observation model.
#     # Needs to know about actions.
#     pass


# class ObservationModel:
#     # Models the O. Will consist of partial models.
#     # Produces a posterior belief in the state given the previous state.
#     # O(o | s', a). This is a function of s'.
#     # - Can an observation depend on more than one partial state?
#     # - We should have some independencies between observations. Perhaps
#     #   full so?
#     # - What if an observation affects a state, e.g., introduces a new possible
#     #   name for the attacker? Name not in prior state, but in posterior.
#     # May take action history into account.
#     # May redefine / modify state representation (and probabilities) based on
#     # observation, before applying the model. Example is taking a new name into
#     # account, splitting it out from the "other" category.
#     # Produces a probability vector over states.
#     # Assume that an observation connects to only one partial state (?), but
#     # one partial state may be connected to many observations.
#     # observations may or may not be present. Is this necessary? Possible to
#     # allow many-to-many, but with the awareness that we are not treating in
#     # the strict mathematical way by wrongly assuming independence.
#     # The input is simply a set of observations.
#     # An observation model also connects to a state, knowing what kind of
#     # observation it can handle.
#     # Needs to know about actions.
#     pass


# class LanguageModel:
#     # This is used by the observation models.
#     # Produces feature vectors that are then analyzed by the observation models.
#     # Or, does an Observation consist of the feature vectors? Seems reasonable
#     # to define human-readable set of observations as input and have a next-level
#     # representation that makes feature vectors.
#     # At some point we want to produce the SNIPS CV-based probabilities.
#     pass


# - Can we build blocks that we can build a strategy out of?


# Can we easily combine local strategies (agendas) into global strategies
# (puppeteers)?
# - Observation models can look at latest actions and determine whether the
#   observation is relevant, or a skip. Can use some ratio of applicability.
#   Thus ok to let all states "listen" in.
# - How to define a policy? Keep the agenda concept here?
#   - Need to know if an agenda can run.
#   - Need to know when an agenda has reached its conclusion.
#   - Agenda adds (partial) state, but may share with others.
#   - Agenda policy: choose likeliest state and 


# Concepts in policy
# - Kickoff triggers, based on input message. (Observations)
#  - Seems like something that the agenda knows. Only based on state, so
#    needs to get this info from the observation into the state. Or, special
#    permission for agents to look at observations?
# - Set of states
#   - That the agenda is based on. Make sure that they exist. May overlap with
#     states of other agendas. Where are they (used state) defined?
#     - Knows that states are connected to OMs and TMs?
# - Start state
#   - How is start belief set? Global (not agenda-level) concern. Sort of related to kickoff.
# - Terminal states
#   - Objective reached, or failed. Need to implement some criterion, agenda-level.
# - Set of triggers (observations)
#   - Transferred into state. Not part of the policy.
# - Transitions (defined by triggers / observations). Triggers "on edges".
#   - Need to map old transition map on new model with obsernvations in nodes.
# - Action map: Per state, a set of actions, with exclusivity and number of uses.
#   - Can be essentially the same, with same set of features.



# Handle confusion more on global level?
# - Did reply something reasonable to last reply?
# - Strategy implementation detail.





# POMDP stuff

# class IndependentBelief:
    
#     def __init__(self, tpf, opfs, belief):
#         self._tpf = tpf
#         self._opfs = opfs
#         self._belief = belief

#     # Owns state description
#     def update(self, observation: Observation):
#         self._belief *= self._tpf(observation)
#         for opf in self._opfs:
#             self._belief *= opf(observation)
#         self._belief /= np.sum(self._belief)


# class TransitionProbabilityFunction:
#     pass


# class ObservationProbabilityFunction:
#     pass










# Need to take a step back to something like a POMDP model to understand what
# the pieces are and how they fit together.
# What would such a system look like? Perhaps not implement all aspects.
# Does the user want to think about POMDPs?
# Need to implement one of our agendas to understand if this works, and to work
# out details.

# Moving information back to the "graph" is an outside thing.

pass


# Work out some existing agenda. Let this be the driver. Look at active ones.
# - get_location
# - get_name
# - move_to_linkeding
# Work out TM and OM connection to states.
# Work out Observation to language model to OM.

# Need to figure out how to handle domains. What needs to be domain-specific?
# Somehow dependencies between policy and BU -- compatible. Can merge domains
# or have hierarchical domains? An agenda has its own little mini-domain.
# Can a BU implement handlers for the type of state and other stuff in a domain,
# i.e, the BU (and Policy) knows about the domain, but not the other way around?


# Needs discretized description of domain
# - Agenda-based approach of Piranha *
# - Something more adapted to POMDP

# Agenda specification.
# - Goes into BU
#   - Piranha-style
#   - Compile to POMDP *
# - Goes into Policy






# Define agenda
# - As in Piranha
# Define set of agendas.

# Create puppeteer using these agendas
# - Choose policy
# - Choose BU (or, always POMDP?)
    
# Start state is "start" for all agendas.




# How to handle confusion?

# Scheme for translating "old" agendas directly
# - Make a single PartialState with all old states.
# - The triggers are indicative of being (having arrived) in the target state

# States
# Original agenda has three levels. Two levels essentially counting number of refusals.
# Current attacker mood
# - unaware of request
# - agreeing to change
# - refusing to change
# Counter for number of refusal statements.

# Actions
# - Request change if we haven't made any requests before.
# - Plead
# - Thank you

