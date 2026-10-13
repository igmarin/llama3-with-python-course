# Lesson 3: Customer-message classifier project

You sort messages sent to the Northwind Bikes shop into **Repair**, **Order** or **Feedback**, so each
one reaches the right person, and you measure how often the model is right.

```bash
python lessons/03_project/classify_messages.py               # rules plus six examples (default)
python lessons/03_project/classify_messages.py --rules-only  # rules without examples
python lessons/03_project/classify_messages.py --save        # write results_<profile>.md
```

## The ideas

**The prompt states the rules.** The system prompt defines each category by what the customer wants
now. Repair is work on a bike or part they already have. Order is buying, reserving, paying,
returning, stock or delivery. Feedback is praise, complaints or suggestions that ask for nothing.
The hard messages in `data/` are written to catch a model that goes by the words: praise followed by
a repair request, a mechanic's advice that leads to a purchase, an "URGENT" note that asks for
nothing.

**The schema limits the answer.** The reply must be JSON with a `category` that is one of the three
labels. The model cannot invent a fourth. If the reply is still unusable, `classify` returns a
fallback label, while `predict` returns `None` so the evaluation can count the failure.

**Score on messages the prompt was not built on.** `data/test.csv` has 33 messages the model never
sees while you design the prompt. Report accuracy for easy and hard messages separately.

**Examples are a trade-off.** The default prompt adds six worked examples as earlier turns of the
chat. Compared on the 129 training-split messages with the 3B model:

| 3B model, training split | Rules only | Rules plus six examples |
|--------------------------|-----------|--------------------------|
| All 129 messages | 92.2% | 95.3% |
| 27 hard messages | 88.9% | 81.5% |

The examples fixed many Order messages that the rules-only prompt called Feedback, but they made
the model worse on hard messages: it copied surface patterns from the examples. Overall accuracy was
the metric chosen in advance, so the default keeps the examples. If hard messages matter most to
you, the choice flips.

On the 33 test messages, both prompts scored 100%, and so did the 8B model. A test set that every
setup passes cannot rank them. The training-split numbers above, and the 8B model's 97.7% overall
and 96.3% on hard training messages, are more informative. To learn more, add harder messages
([how-to](../howto/add-your-own-messages.md)).

## Code map

| File | What it shows |
|------|---------------|
| `src/llama_course/classify.py` | `SYSTEM_PROMPT`, `FEW_SHOT_EXAMPLES`, `predict`, `classify`, `metrics`, `evaluate` |
| `src/llama_course/data.py` | `load_messages("train" or "test")`, which checks columns and labels |
| `lessons/03_project/results_*.md` | Saved output for each profile and prompt |

## Common mistakes

- **Tuning on the test set.** Compare prompts on training messages. Look at the test set once.
- **Reporting one number.** Overall accuracy can hide failures on hard messages. Split the score.
- **Trusting a tiny test set.** One message is 3 points here, and every setup scored 100%.
- **Ignoring unusable replies.** A model that fails to answer is wrong. The report counts them.

## Exercise

Reproduce the comparison above: score both prompts on the training messages with `predict` and
`score_predictions`.

<details>
<summary>Answer</summary>

```python
from llama_course.classify import FEW_SHOT_EXAMPLES, INVALID, predict, score_predictions
from llama_course.data import load_messages
from llama_course.llm import get_llm

llm, df = get_llm(), load_messages("train")
for examples in ((), FEW_SHOT_EXAMPLES):
    preds = [predict(llm, m, examples) or INVALID for m in df["message"]]
    result = score_predictions(df, preds)
    print(len(examples), result["accuracy"], result["accuracy_by_difficulty"])
```

With the 3B model at temperature 0 it prints about 0.922 overall and 0.889 on hard messages with no
examples, and 0.953 and 0.815 with six.
</details>

## Go deeper

- [scikit-learn: model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html):
  accuracy, precision, recall and the confusion matrix.
- [Pydantic JSON Schema](https://docs.pydantic.dev/latest/concepts/json_schema/): schemas from
  Python classes (used in lesson 6).
- [Prompt Engineering Guide](https://www.promptingguide.ai/): when examples help and when they
  don't.
- [Llama 3 model card and prompt formats](https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-3/):
  how the model expects instructions.
