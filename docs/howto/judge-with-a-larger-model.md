# Judge output with a larger model

Use a larger model to score a smaller model's output. This is called
LLM-as-judge, and it's useful when you don't have human labels for every
example.

## When to use it

You trained a classifier (lesson 3) or fine-tuned a model (lesson 7) and want
to know how well it works. You could:

- Read every output by hand. Accurate, but slow.
- Use automatic metrics (accuracy, ROUGE). Fast, but crude.
- Use a larger model to judge the output. A middle ground.

LLM-as-judge is the third option. You give the larger model the input, the
smaller model's output, and ask it to score the output. The larger model acts as
a proxy for human judgment.

## Example: judge the classifier

Here's how to use the 8B model to judge the 3B model's classifications:

```python
from llama_course.classify import classify, FEW_SHOT_EXAMPLES
from llama_course.conversation import message_text
from llama_course.data import load_messages
from llama_course.llm import get_llm

# Load the 3B model and classify the challenge set
llm_3b = get_llm("laptop-16gb")
df = load_messages("challenge")
preds_3b = [classify(llm_3b, msg, FEW_SHOT_EXAMPLES) for msg in df["message"]]

# Load the 8B model and use it as a judge
llm_8b = get_llm("laptop-8b")

judge_prompt = """You are a judge. A customer sent a message to a bike shop, and
an assistant classified it into one of three categories: Repair, Order, or
Feedback. Your job is to decide whether the classification is correct.

Customer message: {message}
Assistant's classification: {predicted}
True classification: {true}

Is the classification correct? Reply with only "yes" or "no"."""

correct = 0
for i, row in df.iterrows():
    prompt = judge_prompt.format(
        message=row["message"],
        predicted=preds_3b[i],
        true=row["category"]
    )
    messages = [{"role": "user", "content": prompt}]
    reply = message_text(llm_8b.create_chat_completion(
        messages=messages, max_tokens=5, temperature=0.0
    )).strip().lower()
    if reply.startswith("yes"):
        correct += 1

print(f"Judge says: {correct}/{len(df)} correct ({correct/len(df):.1%})")
```

## What the judge can and can't do

The judge is good at:

- Catching obvious mistakes (the 3B model said "Repair" for a message that
  clearly asks to buy something).
- Handling ambiguous cases where the true label is debatable.

The judge is not a substitute for:

- Human judgment on edge cases. The 8B model can be wrong too.
- Automatic metrics on large datasets. The judge is slow (one model call per
  example).
- Ground truth. The judge is only as good as its own understanding of the task.

## When the judge disagrees with the labels

If the judge says a label is wrong but the human-labeled data says it's right,
check the human label first. Human labels can be wrong, especially for ambiguous
messages. If the judge is right, update the labels. If the judge is wrong, the
judge prompt needs clarification.

## Where next

- [Lesson 3: Project](../../lessons/03_project/): the classifier being judged.
- [Lesson 7: Fine-tuning](../../lessons/07_finetuning/): ROUGE scores for
  generated text, another automatic metric.
- [Batch classify](../../lessons/03_project/batch_classify.py): score many
  messages efficiently before judging them.