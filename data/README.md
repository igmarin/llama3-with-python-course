# Course datasets

All data here is synthetic. It was written for this course, contains no real
people or messages, and is released under [CC0 1.0](LICENSE) (public domain).

| File | Rows | Columns | Used for |
|------|------|---------|----------|
| `messages.csv` | 162 | `message`, `category`, `difficulty` | Message classifier (lesson 3) and fine-tuning (lesson 7) |
| `train.csv` / `test.csv` | 129 / 33 | same | Fixed 80/20 split of the file above |
| `summaries.csv` | 31 | `text`, `summary` | ROUGE evaluation: `summary` is the reference |
| `inventory_text.csv` | 32 | `text`, `expected_json` | JSON extraction (lessons 2.3 and 6.2) |
| `docs/*.md` | 5 files | Markdown | RAG lesson (lesson 5): the shop's policies |

The course uses one made-up company throughout: Northwind Bikes, a bicycle shop.

## Customer messages

- `category` is one of `Repair` (work on a bike or part the customer already has), `Order`
  (buying, reserving, paying, returning, stock or delivery) or `Feedback` (praise, complaints or
  suggestions that ask for nothing). There are 54 of each.
- `difficulty` is `hard` for 33 rows (about 20%). Hard messages use words that point to the wrong
  class: praise followed by a repair request, a mechanic's advice that leads to a purchase, an
  "URGENT" note that asks for nothing. Report accuracy on easy and hard rows separately.
- `train.csv` and `test.csv` come from `python scripts/split_messages.py`. The split is seeded
  and stratified by category and difficulty. Change the data, then run the script again.

## Delivery notes

`inventory_text.csv` holds supplier delivery notes. `expected_json` maps each part to its quantity,
for example `{"inner tubes": 24}`. Keys are plural part names as written in the note (`tandems`,
not `tandem`). A small model may name a part differently, so compare quantities or normalize keys
before judging a reply wrong.

## Loading

```python
from llama_course.data import load_messages, load_summaries, load_inventory

train = load_messages("train")
```

`load_messages` checks the columns, rejects empty messages, and rejects unknown labels.

## RAG documents

`docs/` holds five short policy pages for a made-up bike shop. `helmets.md` ends with a
planted prompt-injection sentence ("ignore your previous instructions...") so the lesson can
show how the prompt resists it. Do not remove it.
