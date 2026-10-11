# Course datasets

All data here is synthetic. It was written for this course, contains no real
people or messages, and is released under CC0 (public domain).

| File | Rows | Columns | Used for |
|------|------|---------|----------|
| `email_categories_data.csv` | 162 | `email`, `category`, `difficulty` | Email classifier (lesson 4) and fine-tuning (lesson 3) |
| `train.csv` / `test.csv` | 129 / 33 | same | Fixed 80/20 split of the file above |
| `summaries.csv` | 31 | `text`, `summary` | ROUGE evaluation: `summary` is the reference |
| `inventory_text.csv` | 32 | `text`, `expected_json` | JSON extraction (lesson 2.3) |

## Emails

- `category` is one of `Priority` (needs action soon), `Updates` (information only),
  or `Promotions` (marketing). There are 54 of each.
- `difficulty` is `hard` for 33 rows (about 20%). Hard emails have a tone or keywords that
  point to the wrong class: a calm message that needs action today, an "URGENT" notice that
  needs no action, a "Final notice" that is an ad. Report accuracy on easy and hard rows
  separately.
- `train.csv` and `test.csv` come from `python scripts/split_emails.py`. The split is seeded
  and stratified by category and difficulty. Change the data, then run the script again.

## Inventory

`expected_json` maps each item to its count, for example `{"apples": 15}`. Keys are the
plural item names as written in the sentence (`sofas`, not `sofa`). A small model may name an
item differently, so compare counts or normalize keys before judging a reply wrong.

## Loading

```python
from llama_course.data import load_emails, load_summaries, load_inventory

train = load_emails("train")
```

`load_emails` checks the columns, rejects empty emails, and rejects unknown labels.
