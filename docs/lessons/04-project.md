# Lesson 4: Email classifier project

You sort emails into Priority, Updates or Promotions with a Llama model, and measure how often it is
right.

```bash
python lessons/04_project/classify_emails.py            # rules only (default)
python lessons/04_project/classify_emails.py --few-shot # add six worked examples
python lessons/04_project/classify_emails.py --save     # write results_<profile>.md
```

## The ideas

**The prompt states the rules.** The system prompt defines each category by what the reader must do:
Priority needs action soon, Updates is information only, Promotions tries to sell something. It
also says to judge by what the email asks, not by urgent-sounding words. The hard emails in
`data/` are written to catch a model that goes by the words.

**The schema limits the answer.** The reply must be JSON with a `category` that is one of the three
labels. The model cannot invent a fourth. If the reply is still unusable, `classify` returns a
default label, while `predict` returns `None` so the evaluation can count the failure.

**Score on emails the prompt was not built on.** `data/test.csv` has 33 emails the model never
sees while you design the prompt. Report accuracy for easy and hard emails separately.

**Examples can hurt.** The few-shot variant adds six worked examples as earlier turns of the chat.
On the 129 training-split emails it scored 88.4%, against 98.4% for rules only. Most of the lost
accuracy was Priority emails labelled Updates. That is why the default has no examples.

| 3B model | Rules only | Six examples |
|----------|-----------|--------------|
| Training split (129 emails) | 98.4% | 88.4% |
| Test split (33 emails) | 100% | 90.9% |

The prompt was compared on the training split and the test set was scored afterwards. A 100% score
on 33 synthetic emails means the test set is too easy to tell good prompts apart. To learn more,
add harder emails (how-to guide in the docs index).

## Code map

| File | What it shows |
|------|---------------|
| `src/llama_course/classify.py` | `build_messages`, `predict`, `classify`, `metrics`, `evaluate` |
| `src/llama_course/data.py` | `load_emails("train" or "test")`, which checks columns and labels |
| `lessons/04_project/results_laptop-16gb.md` | Saved output for the 3B model |

## Common mistakes

- **Tuning on the test set.** Compare prompts on training emails. Look at the test set once.
- **Reporting one number.** 100% overall can hide failures on hard emails. Split the score.
- **Trusting a tiny test set.** One email is 3 points here. Add data before you believe a 3-point
  difference.
- **Ignoring unusable replies.** A model that fails to answer is wrong. The report counts them.

## Exercise

Compare the two prompts yourself on the training emails. Use `predict` with and without
`FEW_SHOT_EXAMPLES`, and score with `score_predictions`.

<details>
<summary>Answer</summary>

```python
from llama_course.classify import FEW_SHOT_EXAMPLES, INVALID, predict, score_predictions
from llama_course.data import load_emails
from llama_course.llm import get_llm

llm, df = get_llm(), load_emails("train")
for examples in ((), FEW_SHOT_EXAMPLES):
    preds = [predict(llm, e, examples) or INVALID for e in df["email"]]
    print(len(examples), score_predictions(df, preds)["accuracy"])
```

It prints about 0.984 with no examples and 0.884 with six (3B model, temperature 0).
</details>

## Go deeper

- [scikit-learn: model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html):
  accuracy, precision, recall and the confusion matrix.
- [Pydantic JSON Schema](https://docs.pydantic.dev/latest/concepts/json_schema/): schemas from
  Python classes (used in lesson 7).
- [Prompt Engineering Guide](https://www.promptingguide.ai/): when examples help and when they
  don't.
- [Llama 3 model card and prompt formats](https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-3/):
  how the model expects instructions.
