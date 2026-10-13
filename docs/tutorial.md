# Tutorial: build a customer-message classifier

By the end you will have run a Llama model on your own computer, made it return data instead of
prose, and used it to sort customer messages for a bike shop into Repair, Order or
Feedback, with a score that shows how often it is right. It takes about 20 minutes after the
install.

Before you start, finish the [setup](setup.md): the install and the 2 GB model download. Run every command from the repository root, with your virtual environment
active.

## 1. Ask a question

```bash
python lessons/01_basics/01_chat.py
```

You see text about why bicycle tires lose air (the wording differs on every run, and it may wander).
The script loaded the model once with `get_llm()` and asked it to continue a piece of text. Plain
completion tends to ramble, which is the next step's problem. [Lesson 1](lessons/01-basics.md)
explains the parts.

## 2. Use chat messages

```bash
python lessons/01_basics/02_conversation.py
```

You see a numbered list of steps for cleaning and oiling a bike chain. This time the question went in
as a chat message, with a role. The model is trained for that format, so the answer is shaped like
an answer.

## 3. Get data back, not prose

```bash
python lessons/02_prompting/03_json_inventory.py
```

You see:

```
{'inner_tubes': 24, 'brake_pads': 60, 'saddles': 7}
```

The input was a supplier's delivery note: "two dozen inner tubes, sixty brake pads and seven
saddles." The model was forced to reply in JSON, and `chat_json` turned the reply into a Python
dictionary. A program can use that. [Lesson 2](lessons/02-prompting.md) shows how to fix the keys
and types with a JSON Schema.

## 4. Classify one message

Save this as `try_it.py` in the repository root:

```python
from llama_course.classify import classify
from llama_course.llm import get_llm

llm = get_llm()
for message in [
    "My chain jumps when I pedal hard. Can you look at it this week?",
    "Is the Cityline helmet available in size L?",
    "Your mechanic was so patient with my questions. Thank you!",
]:
    print(classify(llm, message), "<-", message)
```

```bash
python try_it.py
```

You see:

```
Repair <- My chain jumps when I pedal hard. Can you look at it this week?
Order <- Is the Cityline helmet available in size L?
Feedback <- Your mechanic was so patient with my questions. Thank you!
```

`classify` sends the model a system prompt that defines the three categories, six worked examples,
and a JSON Schema that allows only those three words, so the answer is always one of them. Change the
messages and run it again.

## 5. Measure it

One right answer proves little. The repository holds 33 labelled test messages, which the prompt was
not built on:

```bash
python lessons/03_project/classify_messages.py
```

You see the accuracy, precision and recall per category, accuracy on easy and hard messages, and a
confusion matrix (which categories get mixed up). On the 3B model it scored 100% on this test set.
[Lesson 3](lessons/03-project.md) explains why that number says less than it seems, and how the
worked examples helped on some messages and hurt on others.

## 6. Try a bigger model

```bash
python scripts/download_model.py --profile laptop-8b
LLAMA_PROFILE=laptop-8b python lessons/03_project/classify_messages.py
```

The 8B model is 4.9 GB and generates about 23 tokens per second, against 45 for the 3B model. It also
scores 100% on the test messages. On the 129 training messages it was more accurate than the 3B model
on hard messages: 96.3% against 81.5%. See [Change models](howto/switch-models.md) for the details.

## What you built

A classifier that runs offline, returns one of three labels every time, and reports its own
accuracy. You can now:

- swap the model without changing code,
- check answers against a held-out test set,
- add your own messages and score again ([how-to](howto/add-your-own-messages.md)).

## Where next

You just used pieces of lessons 1 to 3. The lessons explain each piece, in this order:

| Lesson | What it adds to what you just built |
|--------|-------------------------------------|
| [1. Basics](lessons/01-basics.md) | What `get_llm()` loads and what a chat message contains (steps 1 and 2) |
| [2. Prompting](lessons/02-prompting.md) | How the JSON in step 3 is forced, and how to fix its keys and types |
| [3. Project](lessons/03-project.md) | How the classifier in steps 4 to 6 is built and scored, and what the worked examples cost |
| [4. Generation](lessons/04-generation.md) | Why answers vary between runs, and what happens when a chat gets too long |
| [5. RAG](lessons/05-rag.md) | Answer customer questions from the shop's policy documents |
| [6. Tools](lessons/06-tools.md) | Look up live stock and shipping costs by calling Python functions |
| [7. Fine-tuning](lessons/07-finetuning.md) | Train the model on the messages instead of describing the task in the prompt |
