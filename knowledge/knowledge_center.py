from .payment import payment_knowledge

"""
Knowledge is a dictionary of extraction values (which is also a dictionary). 
Specifically, a key is an action text and its value is pair(s) of key and value, similarly to Extractions.
This knowledge is nothing but an extraction label for the given action. Puppeteer may use this information 
while the conversation progresses as parts of the decision making in detecting triggers or picking actions.
"""

""" To Do: have a function something like knowledge loader that gather
    all knowledge from every agenda (different .py file in this knowledge dir)
    in the KNOWLEDGE variable so Puppeteer can only need to import this KNOWLEDGE variable.
"""

KNOWLEDGE = payment_knowledge 
