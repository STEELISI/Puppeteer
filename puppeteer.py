import abc
from typing import List


class Observation(abc.ABC):
    # Abstract base class for all types of observations used by the Puppeteer
    # to update its beliefs.
    pass


class MessageObservation(Observation):
    # An Observation class implementing a message that has been received. Can
    # subclass this for more specific message types with more specific information.
    
    def __init__(self, text: str):
        self._text = text
    
    def text(self) -> str:
        return self._text


class Action(abc.ABC):
    # An Action object defines an action that can be taken by the Puppeteer.
    pass


class Belief(abc.ABC):
    # The full belief state of a Puppeteer, representing what the Puppeteer
    # believes about the world. Note: this replaces the old MultiBelief class.

    @abc.abstracmethod
    def reset(self):
        raise NotImplementedError()
    
    @abc.abstracmethod
    def update(self, actions: List[Action], observations: List[Observation]):
        raise NotImplementedError()


class Policy(abc.ABC):
    # A Policy implements a Puppeteer's policy for selecting actions to take
    # based on its belief about the world.
    
    @abc.abstracmethod
    def act(self, belief: Belief) -> List[Action]:
        raise NotImplementedError()


class Puppeteer:
    # Main class implementing a puppeteer. Uses a Policy and Belief internally.
    # One object per conversation.
    
    def __init__(self, policy: Policy, belief: Belief):
        self._policy = policy
        self._belief = belief
        self._last_actions = []

    def react(self, observations: List[Observation]) -> List[Action]:
        self._belief.update(self._last_actions, observations)
        self._last_actions = self._policy.act(self._belief)
        return self._last_actions


