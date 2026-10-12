# Tutorial: build an email classifier

By the end you will have run a Llama model on your own computer, made it return data instead of
prose, and used it to sort emails into **Priority**, **Updates** or **Promotions**, with a score that
shows how often it is right. It takes about 20 minutes after the install.

Before you start, finish the [quick start in the README](../README.md#quick-start): the install and
the 2 GB model download. Run every command from the repository root, with your virtual environment
active.

## 1. Ask a question

```bash
python lessons/01_basics/01_chat.py
```

You see a few paragraphs about databases (the wording differs on every run). The script loaded the
model once with `get_llm()` and asked it to continue a piece of text. Plain completion tends to
ramble, which is the next step's problem. [Lesson 1](lessons/01-basics.md) explains the parts.

## 2. Use chat messages

```bash
python lessons/01_basics/02_conversation.py
```

You see four numbered steps for fixing an internet connection. This time the question went in as a
chat message, with a role. The model is trained for that format, so the answer is shaped like an
answer.

## 3. Get data back, not prose

```bash
python lessons/02_prompting/03_json_inventory.py
```

You see:

```
{'apples': 15, 'oranges': 33, 'potatoes': 5052}
```

The input was the sentence "Fifteen apples, thirty-three oranges, and five thousand fifty-two
potatoes." The model was forced to reply in JSON, and `chat_json` turned the reply into a Python
dictionary. A program can use that. [Lesson 2](lessons/02-prompting.md) shows how to fix the keys
and types with a JSON Schema.

## 4. Classify one email

Save this as `try_it.py` in the repository root:

```python
from llama_course.classify import classify
from llama_course.llm import get_llm

llm = get_llm()
for email in [
    "Your invoice is due Friday. Please pay today.",
    "50% off sneakers this weekend only!",
    "Your package was delivered at 2:14 PM.",
]:
    print(classify(llm, email), "<-", email)
```

```bash
python try_it.py
```

You see:

```
Priority <- Your invoice is due Friday. Please pay today.
Promotions <- 50% off sneakers this weekend only!
Updates <- Your package was delivered at 2:14 PM.
```

`classify` sends the model a system prompt that defines the three categories and a JSON Schema that
allows only those three words, so the answer is always one of them. Change the emails and run it
again.

## 5. Measure it

One right answer proves little. The repository holds 33 labelled test emails, which the prompt was
not built on:

```bash
python lessons/04_project/classify_emails.py
```

You see the accuracy, precision and recall per category, accuracy on easy and hard emails, and a
confusion matrix (which categories get mixed up). On the 3B model the rules-only prompt scored 100%
on this test set. [Lesson 4](lessons/04-project.md) explains why that number is less impressive than
it looks, and why adding worked examples to the prompt made things worse.

## 6. Try a bigger model

```bash
python scripts/download_model.py --profile laptop-8b
LLAMA_PROFILE=laptop-8b python lessons/04_project/classify_emails.py
```

The 8B model is 4.9 GB. It scored 93.9% on the same emails, lower than the 3B model, and generates
about 23 tokens per second against 54. A bigger model is not automatically better on a small task,
and a 33-email test set is too small to settle it. See
[Change models](howto/switch-models.md) for the details.

## What you built

A classifier that runs offline, returns one of three labels every time, and reports its own
accuracy. You can now:

- swap the model without changing code,
- check answers against a held-out test set,
- add your own emails and score again ([how-to](howto/add-your-own-emails.md)).

## Where next

- [Lesson 6](lessons/06-rag.md): answer questions from your own documents.
- [Lesson 7](lessons/07-tools.md): let the model call Python functions.
- [Lesson 3](lessons/03-finetuning.md): fine-tune the model on your emails.
