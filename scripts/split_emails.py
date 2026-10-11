"""Split data/email_categories_data.csv into train.csv and test.csv.

The split is stratified by (category, difficulty) and seeded, so every run gives
the same files and both splits contain easy and hard emails of every category.

    python scripts/split_emails.py
"""

import pandas as pd

from llama_course.data import DATA_DIR

TEST_FRACTION = 0.2
SEED = 42


def main() -> None:
    df = pd.read_csv(DATA_DIR / "email_categories_data.csv")
    test = (
        df.groupby(["category", "difficulty"], group_keys=False)
        .sample(frac=TEST_FRACTION, random_state=SEED)
    )
    train = df.drop(test.index)
    train.sample(frac=1, random_state=SEED).to_csv(DATA_DIR / "train.csv", index=False)
    test.sample(frac=1, random_state=SEED).to_csv(DATA_DIR / "test.csv", index=False)
    print(f"train: {len(train)} rows, test: {len(test)} rows")


if __name__ == "__main__":
    main()
