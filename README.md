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
    # Get next message from the othder party. 
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
        # No more messges. End this conversation.
        break
```

The main method used to interact with the puppeteer is the `react()` method.
This method takes two arguments, both giving the puppeteer information about
the state of the outside world.
- The first argument is a list of `Observation` objects, in this case a single
`MessageObservation` containing the latest message text from the other party.
In general, the observations argument to `react()` gives the puppeteer
information about things that have happened since the last turn of the
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
discuss the role of *trigger detectors* in a puppeteer.

#### Trigger detectors

Trigger detectors...

Snips...

## Making new agendas

### Structure of an agenda

### Writing an agenda

### Writing a trigger detector


## Customizing behavior

### Agenda policy

### Agenda state belief

### Puppeteer policy