# Check a small model with a larger one

Use the 8B model to flag the 3B model's labels that need a human look. This is a form of
LLM-as-judge: a second model reviews the first one's output, which helps when you have
no correct labels for new messages.

You need both models downloaded: `laptop-16gb` and `laptop-8b`. Together they use about 8 GB of
memory.

## Ask for a second label, not a verdict

The obvious design is to show the judge the message and the 3B model's label and ask "Is this
right? Reply yes or no." On the 60-message challenge set the 8B model failed at this:

| Judge prompt | Labels it rejected | Real mistakes among them |
|--------------|--------------------|--------------------------|
| "Is "Repair" the right category? Reply with only yes or no." | 60 of 60 | 1 of 1 |
| "Does the proposed category fit? Answer correct or wrong." | 39 of 60 | 1 of 1 |
| Label the message itself, then compare (below) | 6 of 60 | 1 of 1 |

The first two designs caught the one real mistake, but buried it under dozens of false alarms.
Asking the judge to label the message on its own, then comparing, worked much better:

```python
from llama_course.classify import predict
from llama_course.conversation import message_text
from llama_course.data import load_messages
from llama_course.llm import get_llm

JUDGE_PROMPT = """Categories:
- Repair: work on a bike or part the customer already has.
- Order: buying, stock, prices or delivery of something new.
- Feedback: praise, complaints or suggestions, with no request for work or a purchase.

Customer message: {message}

Which category fits best? Reply with one word: Repair, Order or Feedback."""

df = load_messages("challenge")
small = get_llm("laptop-16gb")
predicted = [predict(small, m) or "invalid" for m in df["message"]]

judge = get_llm("laptop-8b")
for message, label in zip(df["message"], predicted):
    reply = message_text(judge.create_chat_completion(
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(message=message)}],
        max_tokens=4, temperature=0.0,
    )).strip()
    if not reply.lower().startswith(label.lower()):
        print(f"Check by hand: 3B said {label}, 8B said {reply}: {message}")
```

On the last run this printed 6 messages. One was the 3B model's only mistake on the set. The other
5 were correct labels the 8B model disagreed with. Reading 6 messages instead of 60 is the saving.

## Test the judge before you trust it

The challenge set has correct labels, so you can measure the judge before using it on messages
that have none. Count two things:

- Real mistakes it caught. Here: 1 of 1.
- Correct labels it flagged. Here: 5 of 59.

Do this every time you change the judge prompt or model. The yes/no prompt above looked
reasonable and flagged everything.

## Limits

- One run, one data set. The numbers above are from the 60 challenge messages at temperature 0.
- The judge can be wrong. Where the two models agree, both can still be wrong, and this check
  will not show it.
- A judge doubles the cost: one extra model call per message.

## Where next

- [Lesson 3: Project](../../lessons/03_project/): the classifier being checked.
- [Lesson 7: Fine-tuning](../../lessons/07_finetuning/): ROUGE, an automatic score for generated text.
