# Puppeteer

The `puppeteer` library implements a modular dialog bot. The dialog bot itself is
the *puppeteer*, and its behavior is defined by a set of *agenda* modules, each
implementing a conversational strategy in a small domain, e.g., trying to find
out the name of the other party. To put it simply, each agenda can be thought
of as a dialog mini-bot, and the set of agendas used by a puppeteer defines
what the puppeteer can talk about.

The other end of the conversation could be handled by a human being, a computer
or something else. For simplicity, we will refer to it as the *other party*,
regardless of what kind of entity it may be.

## Puppeteer basics

We will start by looking at how a puppeteer is created and used. For a user of
this library who is only going to use the puppeteer with pre-defined agendas,
understanding this section may be sufficient.

### Using a puppeteer

The following code shows a part of an example implementation of a message
exchange between a puppeteer and some other party.

```python
# We create a puppeteer somehow. More on this later.
puppeteer = ...

# We start with an empty set of extractions.
extractions = Extractions()

# Main while-loop, one turn of the conversation per iteration.
while True:
    # Get next message from the other party. 
    text = get_message()
    if text is not None:
        # Get the puppeteer's reaction to the message.
        observations = [MessageObservation(text)]
        (actions, new_extractions) = puppeteer.react(observations, extractions)
        
        # Update extractions with new extractions made by the puppeteer.
        extractions.update(new_extractions)
        
        # Perform actions selected by the puppeteer. 
        do_actions(actions)
    else:
        # No more messages. End this conversation.
        break
```

The main method used to interact with the puppeteer is the `react()` method.
This method takes two arguments, both giving the puppeteer information about
the state of the outside world.
- The first argument is a list of `Observation` objects, in this case a single
`MessageObservation` object containing the latest message text from the other
party. In general, the observations argument to `react()` gives the puppeteer
information about things that have happened since the previous turn of the
conversation, i.e., since the last time `react()` was called.
- The second argument holds *extractions* -- key-value pairs that hold
information about the world. This is typically long-term information that is
kept and updated between turns in the conversation. There are sources of
extractions:
  - Extractions made by the puppeteer, returned by `react()`. These are facts
  that the puppeteer has extracted from the observations argument of `react()`.
  Returning extractions is the typical way the puppeteer provides information
  about things it has learned through the conversation.
  - Extractions made outside the puppeteer. This is a way to provide the
  puppeteer with extra background information about the world. Note that this
  mechanism is not used by the example above.
 
Getting the information for and constructing the list of observations for
`react()` is the responsibility of the surrounding implementation, i.e., not
done by the puppeteer. In this example, the `get_message()` function is
responsible for knowing how to get the message text sent by the other party.

Similarly, the puppeteer does not actually perform any actions. Its `react()`
method returns a list of `Action` objects, but these are only selected by the
puppeteer, not performed. It is the responsibility of the surrounding
implementation to perform the selected actions, in this example through the
call to `do_actions()`.

The actions returned by the puppeteer will typically be of conversational
nature, e.g., adding a bit of text to a reply message, but other types of
actions are certainly possible.

### Setting up a puppeteer

The following code shows an example of setting up a puppeteer.

```python
# Set up trigger detector loading.
loader = TriggerDetectorLoader(default_snips_path="path/to/my/snips/engines")

# Load agendas.
first_agenda = Agenda.load("my_first_agenda.yaml", loader)
second_agenda = Agenda.load("my_second_agenda.yaml", loader)
agendas = [first_agenda, second_agenda]

# Create puppeteer.
puppeteer = Puppeteer(agendas, plot_state=True)
```

Creating a puppeteer, using existing agendas from file, is fairly straight-
forward. In this example we choose to have the puppeteer visualize the state of
the current agenda as the conversation goes along, by setting the `plot_state`
flag of the `Puppeteer` constructor.

The part of the above example that warrants a bit of explanation is the trigger
detector loader used when loading the agendas. To do this, we first need to
discuss the role of *trigger detectors* in a puppeteer. This will also explain
the `default_snips_path` argument to the `TriggerDetectorLoader` constructor.

#### Trigger detectors

The information (observations and extractions) passed to the puppeteer's
`react()` method typically needs some interpretation before it can be used to
guide the actions of the puppeteer. As an example, a received message text
typically needs to undergo some kind of natural language processing (NLP)
before its contents can be used to affect the choice of next action for the
agenda currently being used by the puppeteer. This task is handled by *trigger
detectors*.

A trigger detector implements a boolean feature that can be extracted from
the puppeteer's inputs (observations and extractions). The detector determines
whether the feature is present or not in the inputs. The feature itself is
called a *trigger*, and triggers are the main mechanism controlling the
behavior of agendas.

Linking of agenda triggers to trigger detectors is done by name. Agendas define
names for the triggers they use, and trigger detectors define names for the
triggers they detect. A trigger detector can be used for a trigger in an agenda
if names match.  

A puppeteer trigger detector will typically have some kind of textual input,
but this is not a strict requirement. Any kind of observations and extractions
can be used an input to a trigger detector, and each detector (implemented as a
subclass of `TriggerDetector`) specifies what kinds of inputs it reacts to.

##### Snips triggers

The puppeteer library comes with some pre-defined trigger detectors.
Specifically, the library comes with support for using
[SnipsNLU](https://github.com/snipsco/snips-nlu) as an engine for trigger
detectors, using the class `SnipsTriggerDetector`. Snips is a library for
natural language understanding (NLU), using the *intent* of a user sentence as
a central concept in its text analysis. The intent concept maps nicely to
text-based puppeteer triggers, allowing us to define a Snips intent for each
puppeteer trigger, and train Snips to do the detection. 

To create a Snips-based trigger detector, all a user of the puppeteer library
needs to do is provide training data for the underlying Snips engine. This is
done by providing positive and negative examples of sentences where the trigger
feature is or is not present.

#### Loading agendas and trigger detectors

The loading of an agenda from an agenda file, using the `Agenda.load()` method,
also links agendas to trigger detectors, so that the resulting `Agenda` object
is ready for use in a puppeteer. In this process, a `TriggerDetectorLoader` is
used as a lookup, serving `TriggerDetector` objects to `Agenda.load()`.

For details about how to set up the loader, please refer to the
`TriggerDetectorLoader` class. In the simplest case, as shown below, where we
only use Snips-based trigger detectors, it is sufficient to create the loader
and specify the root path where the training data for the Snips intents is
located.

```python
loader = TriggerDetectorLoader(default_snips_path="path/to/my/snips/engines")
agenda = Agenda.load("my_agenda.yaml", loader)
```

As discussed above, each Snips-based trigger detector, corresponding to a Snips
intent, is learned based on sets of example sentences provided in text files.
More specifically, for each intent, there is a folder with the same name as the
intent and this folder contains two text files, each file with a number of
sentences. Each sentence is on its own line in the file. One of the files
contains positive examples, sentences where the trigger is present, and the
other file contains negative example sentences. If the intent name is "xyz",
the file with positive examples should be called "xyz.txt" and the file with
negative examples should be called "NOTxyz.txt".

The `default_snips_path` argument of the `TriggerDetectorLoader` constructor
points out a root folder where the training data for the trigger detectors is
located, each intent with its own sub-folder. Then, using the `loader` as
lookup, `Agenda.load()` links each agenda trigger to the Snips-based trigger
detector that has the same intent name.

## Making new agendas

Defining and extending puppeteer functionality is mostly done by implementing
agendas to handle different domains of conversation. Before going into how to
actually define an agenda, we need to take a look at the different parts that
make up an agenda.

After discussing agenda definition, we will briefly discuss how to define
trigger detectors, and how to load agenda files.

### Parts of an agenda

This is a brief overview of what an agenda looks like. For a more comprehensive
introduction, please refer to the documentation of the code, or this
[paper](http://???).

#### State graph

An agenda is mainly implemented as a state machine with discrete states. States
represent how the agenda is going -- what has been said, how the other party is
reacting, what information has been gained, etc. The states are represented by
nodes in the state machine's state graph. The agenda also defines a single
start state, and a set of terminating states, where the agendas is either
considered to have reached its goal, or to have failed.

Transitions between states are represented by directed edges in the state
graph, and may take place as the result of receiving input to the puppeteer's
`react()` method. Each directed edge *(u, v)* is labeled with one or more
triggers, meaning that the transition from state *u* to state *v* will happen
if one of the triggers is detected in the inputs.

In reality, it is not always 100% clear when a state transition has happened.
There may be several triggers happening at once, leading to different states,
and triggers themselves may be detected with stronger or weaker confidence. For 
this reason, when it is executed, an agenda internally holds a probability
distribution over states, reflecting its relative beliefs in being in different
states at the given time. Thankfully, this complication can be ignored when the
agenda is defined. Defining states, triggers and transitions can thus be done
using a deterministic mental model where the agenda is always in a well-defined
state, and triggers are either "on" or "off".

#### Actions

Each state defines a set of actions that can be taken by the agenda in that
state. In each turn, the current agenda will choose one or more actions from
one of its states. An action typically defines a reply text from the puppeteer
to the other party. Actions come in two flavors: normal actions, and *stall
actions*. Stall action are to be used when the conversation has stalled in a
certain state (as determined by the agenda's policy, described below), and
normal actions in all other cases.

#### Kickoff triggers

Some agendas are not appropriate to run at any time in a conversation, but
should only be used if some specific condition holds. These conditions are
defined by the agenda as its *kickoff triggers*. Just like the *transition
triggers* used to control transitions in the state graph, these are boolean
features computed by trigger detectors.

#### Policy

Apart from the agenda behavior defined by the concepts described above,
there are some remaining aspects of agenda behavior that need to be defined,
among them the exact mechanism for selecting the set of next actions. These
"remaining aspects" are collected in the agenda's *policy*.

The `Agenda` class comes with a pre-defined default policy that can be
customized by setting its parameters in an agenda's text-file specification.
For a discussion about replacing the default policy with a custom one,
defined by implementing a subclass of `AgendaPolicy`, refer to the final
section of this overview, on more customizing of puppeteer behavior.

### Defining an agenda

There are two main ways to define an agenda. The first, and easiest, option is
to define the agenda in an agenda file. This is the recommended way of defining
agendas, at least for new users, and the next section contains a full example
of how this may look.

The other option is to create an Agenda object programmatically, defining the
state graph, triggers, actions, etc. in code, and connecting trigger detectors
"manually". For more information about how to do this, please refer to the
documentation and code in the `Agenda` class. 

#### Agenda files

We will look at an example agenda file defining a very simple agenda that asks
the other party about the time, going through the file section by section.

We start by defining the name of the agenda. The convention is that the names
of agenda entities don't contain whitespace, using underscore, "_", to separate
words.

```yaml
name: ask_for_the_time
```

Next we define the states. Each state gets a name that is used to reference it,
and a description. The description is a comment used to help make the agenda
definition more understandable -- it is not used as a part of agenda behavior.

```yaml
states:
- name: asking_for_time
  description: "We don't know the time, but will ask."
- name: got_time
  description: "They told us the time."
- name: did_not_get_time
  description: "They did not know the time, or refused to tell us."
```

The agenda starts in the state where we are about to ask for the time.

```yaml
start_state_name: asking_for_time
```

The two other states are terminating states, indicating whether we were
successful in learning the time, or not.

```yaml
terminus_names:
- got_time
- did_not_get_time
```

The state transition triggers define the key things that can "happen" in the
conversation as a result of the other party's input. In this agenda, there are
three things that can happen, as shown below.

Obviously there are many ways of modeling a conversation in terms of a state
graph, even when we restrict ourselves to very small domains of conversation.
Designing this conversation model, choosing the set of states and transition
triggers for the graph, is the main challenge in defining an agenda that works
well in practice.

```yaml
transition_triggers:
- name: they_tell_the_time
  description: "They tell us the time."
- name: they_refuse
  description: "They refuse to tell us the time."
- name: they_do_not_know
  description: "They don't know the time."
```

The kickoff trigger defines the condition for starting the agenda. There can
be more than one kickoff trigger, but this simple agenda just uses one.

Note that the definition of this specific kickoff trigger does not give any
hint about what the actual kickoff condition should be. It merely serves as a
hook for the attachment of a trigger detector that defines the actual kickoff
behavior. This is important to keep in mind about triggers: Even in cases where
the name and description of the trigger in the agenda file say relevant things
about the interpretation of the trigger, it is the trigger detector that makes
the *actual* interpretation, and it is up to the programmer to make sure that
the linking of trigger detectors to triggers is consistent with the intents of
trigger definitions in the agenda.

```yaml
kickoff_triggers:
- name: kickoff
  description: "This is the condition for starting the agenda."
```

In our simple agenda state graph, we only have two directed edges, going from
the start state to one of the terminating states. Two triggers lead to the
failure state and one to the success state.

In general, state graphs may be much more complex. Most will have intermediary
states that are neither starting nor terminating states. There may be cycles in
the graph. Specifically, a transition may lead from a state back to the same
state. 

```yaml
transitions:
  asking_for_time:
    they_tell_the_time: got_time
    they_refuse: did_not_get_time
    they_do_not_know: did_not_get_time
```

Each action defines a text that is its contribution to the conversation. It
also has a flag, `exclusive_flag`, telling if the action can be combined with
other actions in the same turn or not. The `allowed_repeats` field specifies
how many times the same action can be used in the same conversation.

```yaml
actions:
- name: ask_time
  text: "What time is it?"
  exclusive_flag: true
  allowed_repeats: 2
- name: ask_time_politely
  text: "Could you please tell me the time?"
  exclusive_flag: true
  allowed_repeats: 2
- name: ask_time_rudely
  text: "Tell me the time!"
  exclusive_flag: true
  allowed_repeats: 2
- name: say_thanks
  text: "Great, thank you very much!"
  exclusive_flag: true
  allowed_repeats: 1
- name: say_ok
  text: "Aha, ok, no problem."
  exclusive_flag: true
  allowed_repeats: 1
```

The action map specifies what actions are applicable in what states.

```yaml
action_map:
  asking_for_time:
  - ask_time
  - ask_time_politely
  - ask_time_rudely
  got_time:
  - say_thanks
  did_not_get_time:
  - say_ok
```

The stall action map specifies what actions are applicable in what states, in
cases where the agenda considers itself stalled in a state. In this agenda, we
simply let this be identical to the action map, but this does not have to be
case in general.

```yaml
stall_action_map:
  asking_for_time:
  - ask_time
  - ask_time_politely
  - ask_time_rudely
  got_time:
  - say_thanks
  did_not_get_time:
  - say_ok
```

Finally, there are some parameters for the agenda's policy. The values used
here are reasonable defaults, and will not be discussed further. Please refer
to class `DefaultAgendaPolicy` for more details.

```yaml
policy:
  reuse: false
  max_transitions: 5
  absolute_accept_thresh: 0.6
  min_accept_thresh_w_differential: 0.2
  accept_thresh_differential: 0.1
  kickoff_thresh: 1.0
```

The complete agenda file looks like this:

```yaml
name: ask_for_the_time

states:
- name: asking_for_time
  description: "We don't know the time, but will ask."
- name: got_time
  description: "They told us the time."
- name: did_not_get_time
  description: "They did not know the time, or refused to tell us."

start_state_name: asking_for_time

terminus_names:
- got_time
- did_not_get_time

transition_triggers:
- name: they_tell_the_time
  description: "They tell us the time."
- name: they_refuse
  description: "They refuse to tell us the time."
- name: they_do_not_know
  description: "They don't know the time."

kickoff_triggers:
- name: kickoff
  description: "This is the condition for starting the agenda."

transitions:
  asking_for_time:
    they_tell_the_time: got_time
    they_refuse: did_not_get_time
    they_do_not_know: did_not_get_time

actions:
- name: ask_time
  text: "What time is it?"
  exclusive_flag: true
  allowed_repeats: 2
- name: ask_time_politely
  text: "Could you please tell me the time?"
  exclusive_flag: true
  allowed_repeats: 2
- name: ask_time_rudely
  text: "Tell me the time!"
  exclusive_flag: true
  allowed_repeats: 2
- name: say_thanks
  text: "Great, thank you very much!"
  exclusive_flag: true
  allowed_repeats: 1
- name: say_ok
  text: "Aha, ok, no problem."
  exclusive_flag: true
  allowed_repeats: 1

action_map:
  asking_for_time:
  - ask_time
  - ask_time_politely
  - ask_time_rudely
  got_time:
  - say_thanks
  did_not_get_time:
  - say_ok

stall_action_map:
  asking_for_time:
  - ask_time
  - ask_time_politely
  - ask_time_rudely
  got_time:
  - say_thanks
  did_not_get_time:
  - say_ok

policy:
  reuse: false
  max_transitions: 5
  absolute_accept_thresh: 0.6
  min_accept_thresh_w_differential: 0.2
  accept_thresh_differential: 0.1
  kickoff_thresh: 1.0
```

### Defining trigger detectors

As seen above, trigger detectors play a key part in agenda behavior. There are
currently two main ways of implementing trigger detectors. The first is to do
it programmatically, implementing a subclass of `TriggerDetector`. The second
way, not requiring any programming, is to use trigger detectors based on the
Snips library, as described in [this](#snips-triggers) section above.

## More customizing of puppeteer behavior

Apart from the defining puppeteer behavior by writing agendas and trigger
detectors the inner workings of the puppeteer can also be redefined. Three
of the key aspects of the puppeteer's implementation are handled by delegate
objects whose default implementation can be replaced by custom ones.

- The agenda's policy, most notably responsible for action selection can be
replaced. Refer to classes `AgendaPolicy` and `DefaultAgendaPolicy` for
details. Note that parameters of the agenda policy (if any) are stored in the
agenda file, so redefining the agenda policy may affect the format of agenda
files as well.
- The agenda's module for computing state probabilities in handled by a
replaceable delegate subclass of `StateProbabilities`. The class
`DefaultStateProbabilities` is used by default, unsurprisingly.
- The puppeteer itself also has a replaceable policy-handling delegate,
responsible for selecting which agenda to run at any given turn. Refer to
`PuppeteerPolicy` and `DefaultPuppeteerPolicy` for details.
