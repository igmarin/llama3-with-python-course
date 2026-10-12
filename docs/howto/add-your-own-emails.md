# Add your own emails

Put your own labelled emails into the classifier project and score the model on them.

## Steps

1. Append rows to `data/email_categories_data.csv`. Each row is `email,category,difficulty`:

   ```csv
   "Reminder: your electricity bill of $82 is due Friday. Pay online to avoid a late fee.",Priority,easy
   "Your gym sent its updated class timetable for the summer.",Updates,easy
   "Summer sale: all swimwear 40% off until Sunday!",Promotions,easy
   ```

   - `category` must be exactly `Priority`, `Updates` or `Promotions`.
   - `difficulty` is `easy`, or `hard` for emails whose wording points to the wrong category (an
     urgent-sounding promotion, a calm message that needs action today).
   - Put the email in quotes if it contains a comma.
   - Keep emails unique. Do not use real people's messages.

2. Rebuild the training and test files:

   ```bash
   python scripts/split_emails.py
   ```

   You see `train: 132 rows, test: 33 rows`. The split is random with a fixed seed, so adding rows
   can change which emails land in the test set.

3. Check the data:

   ```bash
   pytest tests/test_data.py
   ```

   You see `10 passed`.

4. Score the model:

   ```bash
   python lessons/04_project/classify_emails.py
   ```

   Adding three easy emails here gave 100% again. Compare against a baseline you ran on the same
   data, since the test set may have changed.

## Tips

- To find out where the model fails, add hard emails. A test set where everything passes cannot rank
  two prompts or two models.
- Add rows to both categories evenly. The data tests expect the three classes to stay within 5%.
- Do not compare prompts on the test emails. Compare on `train.csv`, then score the test set once.

## If something fails

- `ValueError: Unknown labels ['Spam']; expected ['Priority', 'Updates', 'Promotions']`: a row uses a
  label the project does not know. Fix the spelling, or change `LABELS` in `src/llama_course/data.py`
  and the prompt in `src/llama_course/classify.py` together.
- `Missing columns` or `Found an empty email`: a row is missing a field or has an extra comma
  outside quotes.
