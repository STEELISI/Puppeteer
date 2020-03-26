# Puppeteer

The puppeteer library implements a modular dialog bot. The dialog bot itself is
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
subclass of TriggerDetector) specifies what kinds of inputs it reacts to.

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

## Making new agendas

Defining and extending puppeteer functionality is mostly done by implementing
agendas to handle different domains of conversation. Before going into how to
actually define an agenda, we need to take a look at the different parts that
make up an agenda.

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

### Writing an agenda

#### Agenda files

#### Programmatically


### Writing a trigger detector

#### Snips files

#### Programmatically


## More customizing of puppeteer behavior

### Agenda policy

### Agenda state belief

### Puppeteer policy