import numpy as np
import pandas as pd

KEY = ["main_ident", "timeslot_datetime_from", "channel_id"]


def check_duplicates(df: pd.DataFrame, name: str, ignore: list[str] = ()) -> None:
    cols = [c for c in df.columns if c not in ignore]
    print(f"== {name}: {len(df)} rows")
    print("  full-row duplicates:", df.duplicated(subset=cols).sum())
    print("  key duplicates (ident, time, channel):", df.duplicated(subset=KEY).sum())

    sizes = df.groupby(KEY).size()
    print("  unique keys:", len(sizes))
    print("  copies per key:", sizes.value_counts().sort_index().to_dict())

    if "share_15_54" in df.columns:
        conflicting = (df.groupby(KEY)["share_15_54"].nunique(dropna=False) > 1).sum()
        print("  keys with conflicting target:", conflicting)


train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

check_duplicates(train, "train")
check_duplicates(test, "test", ignore=["row_id"])
print("test row_id unique:", test["row_id"].is_unique)
print("train/test key overlap:", len(train[KEY].drop_duplicates().merge(test[KEY])))

data = train.drop_duplicates().reset_index(drop=True)
print(f"\ntrain after drop_duplicates: {len(data)} rows")
print(data)
