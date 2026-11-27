"""Exploratory data analysis of the prime-time audience share data.

Run from the repo root:  python3 src/data_explore.py

Key findings (as of the current data download):
  * train.csv: ~48% of rows are exact duplicates (join fan-out per timeslot, copies
    are adjacent, 2..32 copies). Targets never conflict -> drop_duplicates() is safe.
  * f_7..f_11 are 100% missing in test for every channel (and ch9 has them missing in
    train too). Anything learned from them cannot be used at predict time.
  * main_ident = <channel_id><YYMMDDHHMMSS programme start><4-digit suffix>. The
    start gives "minutes since programme start"; f_1 ~ elapsed / duration (progress).
  * No ch3 programme in test appears in train (ident includes the air date), so
    programme-level target encoding on main_ident is useless.
  * Timeslots are sparse samples (~6 per evening per channel), not a full grid;
    one day (2023-05-31) is missing.
  * share_15_54_3mo_mean is a smoothed trailing level (~past 90 days, same hour,
    best match with ~1 month lag); corr 0.61 with ch3 target. As a predictor (MAE
    5.43) it loses to a plain hour x weekday mean (MAE 4.54) on the last 2 months.
  * ch3 share is driven by time: peaks 18-19h on weekdays (~31%), weekends early
    evening much lower (~18%). Christmas 2022 flattens the profile (16h up, 19h down).
  * ident suffix 0010 (fixed 19:30 news-type starts) -> ch3 ~31% vs 21% for 0020.
  * Missingness is informative: ch3 share 26.8% when ch3__f_12 missing vs 19.4%.
  * f_10/f_11 look like genre/country (Czech strings), f_12 small-int with long tail,
    f_9 binary. Czech origin / Drama / Publicistický -> high ch3 share, USA lower.
  * Test ch3 3mo_mean level (21.4) is lower than the train average (22.9); test has
    few 19h slots, so hour mix differs from train.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)

TARGET = "share_15_54"
MO3 = "share_15_54_3mo_mean"
KEY = ["main_ident", "timeslot_datetime_from", "channel_id"]
CHANNELS = [3, 4, 9, 54]
SCORED = 3
CAT_FEATS = (10, 11)


def section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def fcols(df: pd.DataFrame, ch: int | None = None, i: int | None = None) -> list[str]:
    """Columns ch{C}__f_{i}, optionally filtered by channel and/or feature index."""
    out = []
    for c in df.columns:
        if "__f_" not in c:
            continue
        c_ch, c_i = c[2:].split("__f_")
        if (ch is None or int(c_ch) == ch) and (i is None or int(c_i) == i):
            out.append(c)
    return out


# --------------------------------------------------------------------------- 1. duplicates
def check_duplicates(df: pd.DataFrame, name: str, ignore: list[str] = ()) -> None:
    cols = [c for c in df.columns if c not in ignore]
    print(f"== {name}: {len(df)} rows")
    print("  full-row duplicates:", df.duplicated(subset=cols).sum())
    print("  key duplicates (ident, time, channel):", df.duplicated(subset=KEY).sum())

    sizes = df.groupby(KEY).size()
    print("  unique keys:", len(sizes))
    print("  copies per key:", sizes.value_counts().sort_index().to_dict())

    if TARGET in df.columns:
        conflicting = (df.groupby(KEY)[TARGET].nunique(dropna=False) > 1).sum()
        print("  keys with conflicting target:", conflicting)

    if sizes.max() > 1:
        idx = df.reset_index(drop=True).reset_index().groupby(KEY)["index"].agg(["min", "max", "size"])
        idx = idx[idx["size"] > 1]
        print("  duplicate copies adjacent in file:", ((idx["max"] - idx["min"] + 1) == idx["size"]).mean())
        per_slot = df.groupby(KEY).size().rename("n").reset_index()
        per_slot = per_slot.groupby("timeslot_datetime_from")["n"].nunique()
        print("  timeslots where all channels have the same copy count:", (per_slot == 1).mean())


# --------------------------------------------------------------------------- 2. derived columns
def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["t"] = pd.to_datetime(df["timeslot_datetime_from"], errors="coerce")
    df["date"] = df["t"].dt.normalize()
    df["hour"] = df["t"].dt.hour
    df["dow"] = df["t"].dt.dayofweek
    df["month"] = df["t"].dt.to_period("M")

    ident = df["main_ident"].astype(str)
    ch = df["channel_id"].astype(str)
    rest = pd.Series([a[len(b):] if a.startswith(b) else "" for a, b in zip(ident, ch)], index=df.index)
    df["ident_prefix_ok"] = [a.startswith(b) for a, b in zip(ident, ch)]
    df["prog_start"] = pd.to_datetime(rest.str[:12], format="%y%m%d%H%M%S", errors="coerce")
    df["ident_suffix"] = rest.str[12:]
    df["elapsed_min"] = (df["t"] - df["prog_start"]).dt.total_seconds() / 60
    return df


# --------------------------------------------------------------------------- 3. schema
def schema(train: pd.DataFrame, test: pd.DataFrame) -> None:
    section("SCHEMA")
    print("train shape:", train.shape, " test shape:", test.shape)
    print("only in train:", sorted(set(train.columns) - set(test.columns)))
    print("only in test:", sorted(set(test.columns) - set(train.columns)))
    print("\ndtypes (train):", train.dtypes.value_counts().to_dict())
    mismatched = {
        c: (str(train[c].dtype), str(test[c].dtype))
        for c in train.columns.intersection(test.columns)
        if train[c].dtype != test[c].dtype
    }
    print("dtype mismatches train vs test:", mismatched)
    print("\nrows per channel:")
    print(pd.DataFrame({"train": train["channel_id"].value_counts(), "test": test["channel_id"].value_counts()}))


# --------------------------------------------------------------------------- 4. missingness
def missingness(train: pd.DataFrame, test: pd.DataFrame) -> None:
    section("MISSINGNESS (% of rows, per channel block x feature)")
    for name, df in (("train", train), ("test", test)):
        tab = pd.DataFrame(
            {f"ch{ch}": [df[f"ch{ch}__f_{i}"].isna().mean() * 100 for i in range(1, 13)] for ch in CHANNELS},
            index=[f"f_{i}" for i in range(1, 13)],
        )
        print(f"\n{name}:\n{tab.round(1)}")

    fully_missing_test = [c for c in fcols(test) if test[c].isna().all()]
    print("\ncolumns 100% missing in test:", fully_missing_test)
    useless = [c for c in fully_missing_test if train[c].notna().any()]
    print("-> present in train but useless at predict time:", useless)

    # does missingness of the own-channel block relate to the target?
    c3 = train[train["channel_id"] == SCORED]
    for i in (3, 5, 12):
        col = f"ch3__f_{i}"
        print(f"ch3 share when {col} missing vs present:",
              c3.groupby(c3[col].isna())[TARGET].mean().round(2).to_dict())


# --------------------------------------------------------------------------- 5. time coverage
def time_coverage(train: pd.DataFrame, test: pd.DataFrame) -> None:
    section("TIME COVERAGE")
    for name, df in (("train", train), ("test", test)):
        print(f"\n{name}: {df['t'].min()} -> {df['t'].max()}, unparsable dates: {df['t'].isna().sum()}")
        all_days = pd.date_range(df["date"].min(), df["date"].max())
        missing = sorted(set(all_days) - set(df["date"]))
        print("  missing days:", [d.date() for d in missing])
        print("  hours:", sorted(df["hour"].unique().tolist()), " minutes:", sorted(df["t"].dt.minute.unique().tolist()))
        print("  rows outside 16:00-23:55:", ((df["hour"] < 16) | (df["hour"] > 23)).sum())
        print("  timeslots by number of channels present:",
              df.groupby("timeslot_datetime_from").size().value_counts().sort_index().to_dict())
        spd = df[df["channel_id"] == SCORED].groupby("date").size()
        print(f"  ch3 slots per evening: mean {spd.mean():.1f}, min {spd.min()}, max {spd.max()}")
        print("  rows per hour (ch3):", df[df["channel_id"] == SCORED]["hour"].value_counts().sort_index().to_dict())


# --------------------------------------------------------------------------- 6. main_ident
def main_ident_analysis(train: pd.DataFrame, test: pd.DataFrame) -> None:
    section("MAIN_IDENT = <channel><YYMMDDHHMMSS programme start><suffix>")
    for name, df in (("train", train), ("test", test)):
        print(f"\n{name}:")
        print("  prefix == channel_id:", df["ident_prefix_ok"].mean())
        print("  unparsable programme start:", df["prog_start"].isna().sum())
        print("  suffix counts:", df["ident_suffix"].value_counts().to_dict())
        print("  minutes since programme start:", df["elapsed_min"].describe().round(1).to_dict())
        print("  slot starts before programme (elapsed < 0):", (df["elapsed_min"] < 0).sum(),
              "(min", df["elapsed_min"].min(), ") - programme starts inside the 5-min slot")

    c3 = train[train["channel_id"] == SCORED]
    print("\nch3 share by ident suffix:")
    print(c3.groupby("ident_suffix")[TARGET].agg(["mean", "std", "size"]).round(2))
    bins = [-10, 0, 15, 30, 60, 90, 120, 300]
    print("\nch3 share by minutes since programme start:")
    print(c3.groupby(pd.cut(c3["elapsed_min"], bins), observed=True)[TARGET].agg(["mean", "std", "size"]).round(2))

    per_prog = c3.groupby("main_ident")[TARGET].agg(["size", "std"])
    print(f"\nch3: {len(per_prog)} programmes, rows/programme {per_prog['size'].mean():.2f}, "
          f"within-programme std {per_prog['std'].mean():.2f} vs overall std {c3[TARGET].std():.2f}")
    t3 = test[test["channel_id"] == SCORED]
    print("test ch3 idents seen in train:", t3["main_ident"].isin(train["main_ident"]).mean())

    for ch in CHANNELS:
        d = train[train["channel_id"] == ch]
        print(f"ch{ch}: spearman(f_1, elapsed_min) = {d[f'ch{ch}__f_1'].corr(d['elapsed_min'], method='spearman'):.2f}")
    print("-> f_1 behaves like programme progress (elapsed / duration); values > 1 = overrun:",
          sum((train[c] > 1).sum() for c in fcols(train, i=1)))


# --------------------------------------------------------------------------- 7. target
def target_analysis(train: pd.DataFrame) -> None:
    section("TARGET share_15_54")
    print(train.groupby("channel_id")[TARGET].describe().round(2))
    print("\nvalues < 0 or > 100:", ((train[TARGET] < 0) | (train[TARGET] > 100)).sum())
    print("exact zeros per channel:", train.loc[train[TARGET] == 0, "channel_id"].value_counts().to_dict())

    c3 = train[train["channel_id"] == SCORED]
    print("\nch3 quantiles:", c3[TARGET].quantile([0, .001, .01, .05, .5, .95, .99, .999, 1]).round(2).to_dict())
    print("ch3 skew:", round(float(c3[TARGET].skew()), 2), " kurtosis:", round(float(c3[TARGET].kurt()), 2))

    print("\nch3 by hour:")
    print(c3.groupby("hour")[TARGET].agg(["mean", "median", "std", "size"]).round(2).T)
    print("\nch3 by weekday (0=Mon):")
    print(c3.groupby("dow")[TARGET].agg(["mean", "median", "std", "size"]).round(2).T)
    print("\nch3 by hour x weekday (mean):")
    print(c3.pivot_table(index="hour", columns="dow", values=TARGET, aggfunc="mean").round(1))
    print("\nch3 by month:")
    print(c3.groupby("month")[TARGET].agg(["mean", "std", "size"]).round(2).T)

    # Christmas / New Year: test covers it, train has one such period (2022/23)
    xmas = c3[(c3["t"] >= "2022-12-20") & (c3["t"] < "2023-01-07")]
    ref = c3[(c3["t"] >= "2022-11-15") & (c3["t"] < "2022-12-20")]
    print(f"\nch3 Christmas 20 Dec 2022 - 6 Jan 2023: mean {xmas[TARGET].mean():.2f} (n={len(xmas)}) "
          f"vs 15 Nov - 19 Dec 2022: {ref[TARGET].mean():.2f}")
    print("  by hour, Christmas vs reference:")
    print(pd.DataFrame({"xmas": xmas.groupby("hour")[TARGET].mean(), "ref": ref.groupby("hour")[TARGET].mean()}).round(2).T)

    # season comparison with test window (Nov - mid Jan) in 2022/23
    same_season = c3[(c3["t"] >= "2022-11-01") & (c3["t"] < "2023-01-14")]
    print(f"\nch3 Nov 2022 - 13 Jan 2023 (test-like season): mean {same_season[TARGET].mean():.2f}, "
          f"overall {c3[TARGET].mean():.2f}")

    weekly = c3.set_index("t")[TARGET].resample("W").mean()
    print("\nch3 weekly mean: min", weekly.idxmin().date(), round(float(weekly.min()), 2),
          " max", weekly.idxmax().date(), round(float(weekly.max()), 2), " std", round(float(weekly.std()), 2))

    # extreme slots relative to the smoothed level
    ratio = c3[TARGET] / c3[MO3]
    print("\nch3 share / 3mo_mean quantiles:", ratio.quantile([0, .01, .5, .99, 1]).round(2).to_dict())
    cols = ["timeslot_datetime_from", "main_ident", TARGET, MO3, "ch3__f_10", "ch3__f_11"]
    print("top 10 ch3 slots:")
    print(c3.nlargest(10, TARGET)[cols])
    print("bottom 10 ch3 slots:")
    print(c3.nsmallest(10, TARGET)[cols])


# --------------------------------------------------------------------------- 8. 3-month mean
def three_month_mean(train: pd.DataFrame, test: pd.DataFrame) -> None:
    section("share_15_54_3mo_mean")
    c3 = train[train["channel_id"] == SCORED].sort_values("t")
    print("corr with target per channel:",
          {ch: round(float(d[TARGET].corr(d[MO3])), 3) for ch, d in train.groupby("channel_id")})
    print("distinct 3mo values per ch3 evening:", round(float(c3.groupby("date")[MO3].nunique().mean()), 2),
          "-> varies within an evening (time-of-day specific)")

    # which window reproduces it best? trailing / centered / leading mean of the target, same hour
    results = {}
    for name, (lo, hi) in {"past 90d": (-90, 0), "centered 90d": (-45, 45),
                           "future 90d": (0, 90), "past 90d lag 30d": (-120, -30)}.items():
        est = pd.Series(np.nan, index=c3.index)
        for _, g in c3.groupby("hour"):
            tv, yv = g["t"].to_numpy(), g[TARGET].to_numpy()
            for idx, t in zip(g.index, tv):
                m = (tv >= t + np.timedelta64(lo, "D")) & (tv < t + np.timedelta64(hi, "D")) & (tv != t)
                if m.any():
                    est[idx] = yv[m].mean()
        results[name] = round(float(c3[MO3].corr(est)), 3)
    print("corr(3mo_mean, rolling target mean at same hour):", results)

    print("\ntest 3mo_mean per channel:")
    print(pd.DataFrame({"train": train.groupby("channel_id")[MO3].mean(),
                        "test": test.groupby("channel_id")[MO3].mean()}).round(2))


# --------------------------------------------------------------------------- 9. baselines
def baselines(train: pd.DataFrame) -> None:
    section("NAIVE BASELINES (ch3, time split: validate on last 2 months of train)")
    c3 = train[train["channel_id"] == SCORED]
    cut = c3["t"].max() - pd.DateOffset(months=2)
    fit, val = c3[c3["t"] < cut], c3[c3["t"] >= cut]
    y = val[TARGET]

    def mae(p) -> float:
        return round(float(np.mean(np.abs(y - p))), 3)

    hour_dow = fit.groupby(["hour", "dow"])[TARGET].mean()
    preds = {
        "global mean": np.full(len(val), fit[TARGET].mean()),
        "global median": np.full(len(val), fit[TARGET].median()),
        "hour mean": val["hour"].map(fit.groupby("hour")[TARGET].mean()),
        "hour x weekday mean": pd.Series(list(zip(val["hour"], val["dow"])), index=val.index).map(hour_dow),
        "3mo_mean": val[MO3],
    }
    for k, p in preds.items():
        print(f"  {k:22s} MAE {mae(p)}")
    print(f"  (validation from {cut.date()}, n={len(val)})")


# --------------------------------------------------------------------------- 10. features
def numeric_features(train: pd.DataFrame, test: pd.DataFrame) -> None:
    section("NUMERIC f_* (pooled over channels): train vs test distribution")
    rows = []
    for i in range(1, 13):
        if i in CAT_FEATS:
            continue
        tr = pd.concat([train[c] for c in fcols(train, i=i)])
        te = pd.concat([test[c] for c in fcols(test, i=i)])
        te_ok = te.notna().any()
        rows.append({
            "f": f"f_{i}", "n_unique": tr.nunique(),
            "tr_min": tr.min(), "tr_p01": tr.quantile(.01), "tr_med": tr.median(),
            "tr_p99": tr.quantile(.99), "tr_max": tr.max(),
            "te_med": te.median() if te_ok else np.nan, "te_max": te.max() if te_ok else np.nan, "te_missing%": te.isna().mean() * 100,
        })
    print(pd.DataFrame(rows).set_index("f").round(2))

    f12 = pd.concat([train[c] for c in fcols(train, i=12)])
    print("\nf_12 value counts (small integer, heavy tail -> maybe categorical / count):")
    print(f12.value_counts().sort_index().to_dict())
    f9 = pd.concat([train[c] for c in fcols(train, i=9)])
    print("f_9 values:", f9.value_counts(dropna=False).to_dict())

    # relationships between features of one channel
    own = [f"ch3__f_{i}" for i in range(1, 13) if i not in CAT_FEATS]
    print("\nspearman correlation among ch3 numeric features:")
    print(train[train["channel_id"] == SCORED][own].corr(method="spearman").round(2))


def categorical_features(train: pd.DataFrame, test: pd.DataFrame) -> None:
    section("CATEGORICAL f_10 (genre?) / f_11 (country?)")
    c3 = train[train["channel_id"] == SCORED]
    for i in CAT_FEATS:
        for ch in CHANNELS:
            col = f"ch{ch}__f_{i}"
            print(f"{col}: {train[col].nunique()} levels, test non-null: {test[col].notna().sum()}")
        print(f"\nch3 share by own ch3__f_{i}:")
        print(c3.groupby(f"ch3__f_{i}", dropna=False)[TARGET].agg(["mean", "std", "size"])
              .sort_values("size", ascending=False).head(15).round(2))


def feature_target_corr(train: pd.DataFrame) -> None:
    section("SPEARMAN CORRELATION WITH ch3 TARGET (ch3 rows)")
    c3 = train[train["channel_id"] == SCORED]
    num = [c for c in fcols(train) if train[c].dtype != object] + [MO3, "elapsed_min", "hour", "dow"]
    cor = c3[num].corrwith(c3[TARGET], method="spearman").dropna().sort_values(key=np.abs, ascending=False)
    print(cor.head(25).round(3).to_string())


# --------------------------------------------------------------------------- 11. competitors
def competitors(train: pd.DataFrame) -> None:
    section("COMPETITOR CONTEXT")
    wide = train.pivot_table(index="timeslot_datetime_from", columns="channel_id", values=TARGET)
    print("corr of shares between channels at the same timeslot:")
    print(wide.corr().round(2))
    print("\nsum of 4 shares per timeslot (all 4 present):",
          wide.dropna().sum(axis=1).describe().round(2).to_dict())

    c3 = train[train["channel_id"] == SCORED]
    print("\nch3 share by which competitors are present in the timeslot:")
    present = train.groupby("timeslot_datetime_from")["channel_id"].apply(lambda s: ",".join(map(str, sorted(s))))
    print(c3.groupby(c3["timeslot_datetime_from"].map(present))[TARGET].agg(["mean", "size"]).round(2))

    print("\nch3 share by ch4 programme progress (ch4__f_1, a programme just starting on ch4?):")
    print(c3.groupby(pd.cut(c3["ch4__f_1"], [-.01, .1, .3, .6, .9, 5]), observed=True)[TARGET]
          .agg(["mean", "size"]).round(2))
    print("\nch3 share by competitor ch4 genre (ch4__f_10, top 10):")
    print(c3.groupby("ch4__f_10")[TARGET].agg(["mean", "size"]).sort_values("size", ascending=False).head(10).round(2))


# --------------------------------------------------------------------------- main
def main() -> None:
    train_raw = pd.read_csv("./data/train.csv")
    test_raw = pd.read_csv("./data/test.csv")

    section("DUPLICATES")
    check_duplicates(train_raw, "train")
    check_duplicates(test_raw, "test", ignore=["row_id"])
    print("test row_id unique:", test_raw["row_id"].is_unique)
    print("train/test key overlap:", len(train_raw[KEY].drop_duplicates().merge(test_raw[KEY])))

    train = add_derived(train_raw.drop_duplicates().reset_index(drop=True))
    test = add_derived(test_raw)
    print(f"\ntrain after drop_duplicates: {len(train)} rows")

    schema(train_raw, test_raw)
    missingness(train, test)
    time_coverage(train, test)
    main_ident_analysis(train, test)
    target_analysis(train)
    three_month_mean(train, test)
    baselines(train)
    numeric_features(train, test)
    categorical_features(train, test)
    feature_target_corr(train)
    competitors(train)


if __name__ == "__main__":
    main()
