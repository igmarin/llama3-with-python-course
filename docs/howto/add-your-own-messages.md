# Add your own messages

Put your own labelled customer messages into the classifier project and score the model on them.

## Steps

1. Append rows to [`data/messages.csv`](../../data/messages.csv). Each row is `message,category,difficulty`:

   ```csv
   "My front light flickers when I ride over cobbles. Can you check the wiring?",Repair,easy
   "Do you have a 160 mm disc rotor for a Shimano brake in stock?",Order,easy
   "The new bike racks outside the shop are a great idea.",Feedback,easy
   ```

   - `category` must be exactly `Repair`, `Order` or `Feedback`.
   - `difficulty` is `easy`, or `hard` for messages whose wording points to the wrong category
     (praise followed by a repair request, an "URGENT" note that asks for nothing).
   - Put the message in quotes if it contains a comma.
   - Keep messages unique. Do not use real customers' messages.

2. Rebuild the training and test files:

   ```bash
   python scripts/split_messages.py
   ```

   You see `train: 132 rows, test: 33 rows`. The script picks the test messages at random, but with a
   fixed seed, so the same data always gives the same split. Adding rows can still change which
   messages land in the test set.

3. Check the data:

   ```bash
   pytest tests/test_data.py
   ```

   You see `10 passed`.

4. Score the model:

   ```bash
   python lessons/03_project/classify_messages.py
   ```

   Adding the three messages above gave 100% again. To see whether a change helped, compare against
   a run on the same data (a baseline), because the test set may have changed.

## Tips

- To find out where the model fails, add hard messages. A test set where everything passes cannot
  rank two prompts or two models. Hard messages can also go into
  [`data/challenge.csv`](../../data/challenge.csv), which is never split; score it with
  `--split challenge`. Keep the three categories even there too.
- Add rows to the three categories evenly. The data tests expect the classes to stay within 5% of
  each other.
- Do not compare prompts on the test messages. Compare on `train.csv`, then score the test set once.

## If something fails

- `ValueError: Unknown labels ['Spam']; expected ['Repair', 'Order', 'Feedback']`: a row uses a
  label the project does not know. Fix the spelling, or change `LABELS` in [`src/llama_course/data.py`](../../src/llama_course/data.py)
  and the prompt in [`src/llama_course/classify.py`](../../src/llama_course/classify.py) together.
- `Missing columns` or `Found an empty message`: a row is missing a field or has an extra comma
  outside quotes.
