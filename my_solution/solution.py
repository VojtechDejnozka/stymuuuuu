"""Prime-time audience share: channel 3 share_15_54 forecast.

Gradient boosting (scikit-learn HistGradientBoostingRegressor, absolute-error loss) trained on
channel 3 rows. Features:
  * calendar: hour, minute of evening, weekday, season, Czech public holidays, Christmas period;
  * programme timing decoded from main_ident = <channel><YYMMDDHHMMSS start><suffix>;
  * share_15_54_3mo_mean and the anonymised f_1..f_6, f_12 of all four channels;
  * competitor context: 3-month means and programme timing of the other channels in the same timeslot.
f_7..f_11 are never used (empty in the test period). Development and validation: notebooks/model.ipynb.
"""
from __future__ import annotations

from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).parent

TARGET = "share_15_54"
MO3 = "share_15_54_3mo_mean"
CHANNELS = [3, 4, 9, 54]
SCORED = 3
USE_F = [1, 2, 3, 4, 5, 6, 12]
CTX_VARS = [MO3, "elapsed", "prog_start_min", "suffix"]

CALENDAR = ["hour", "minute_of_evening", "dow", "weekend", "doy_sin", "doy_cos", "holiday", "xmas", "summer"]
IDENT = ["elapsed", "prog_start_min", "suffix"]
LEVEL = [MO3, "n_channels"]
F_COLS = [f"ch{c}__f_{i}" for c in CHANNELS for i in USE_F]
CTX_COLS = [f"ctx_{v}_ch{c}" for c in CHANNELS if c != SCORED for v in CTX_VARS]
FEATURES = CALENDAR + IDENT + LEVEL + F_COLS + CTX_COLS

PARAMS = dict(loss="absolute_error", learning_rate=0.03, max_iter=800, max_leaf_nodes=15,
              min_samples_leaf=30, l2_regularization=1.0, early_stopping=False)
MIN_ROWS = 50  # below this many usable ch3 rows, fall back to simpler predictions


# --------------------------------------------------------------------------- features
def _ident_strings(s: pd.Series) -> pd.Series:
    """main_ident as a digit string, whatever dtype it arrives in."""
    if pd.api.types.is_integer_dtype(s):
        return s.astype("Int64").astype(str)
    if pd.api.types.is_float_dtype(s):
        return s.map(lambda v: f"{v:.0f}" if np.isfinite(v) else "")
    return s.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)


def _decode_ident(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """main_ident = <channel_id><YYMMDDHHMMSS programme start><4-digit suffix>."""
    ident = _ident_strings(df["main_ident"])
    ch = df["channel_id"].map(lambda v: f"{v:.0f}" if pd.notna(v) else "\x00")
    rest = pd.Series([a[len(b):] if a.startswith(b) else "" for a, b in zip(ident, ch)], index=df.index)
    start = pd.to_datetime(rest.str[:12], format="%y%m%d%H%M%S", errors="coerce")
    return start, pd.to_numeric(rest.str[12:], errors="coerce")


def _holiday_dates(t: pd.Series) -> pd.DatetimeIndex:
    years = t.dt.year.dropna().astype(int).unique().tolist()
    if not years:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(pd.to_datetime(list(holidays.Czechia(years=years).keys())))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw train/test frame -> frame with all FEATURES. Row-order independent; index preserved."""
    df = df.copy()
    for col in ["main_ident", "timeslot_datetime_from", "channel_id", MO3, *F_COLS]:
        if col not in df:
            df[col] = np.nan
    df["channel_id"] = pd.to_numeric(df["channel_id"], errors="coerce")
    missing_ch = df["channel_id"].isna()
    if missing_ch.any():  # recover the channel from the main_ident prefix
        ident = _ident_strings(df.loc[missing_ch, "main_ident"])
        prefix = ident.str[:2].where(ident.str[:2] == "54", ident.str[:1])
        df.loc[missing_ch, "channel_id"] = pd.to_numeric(prefix, errors="coerce").where(lambda s: s.isin(CHANNELS))
    for col in [MO3, *F_COLS]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    t = pd.to_datetime(df["timeslot_datetime_from"], errors="coerce", format="mixed")
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_localize(None)
    start, suffix = _decode_ident(df)
    df["elapsed"] = (t - start).dt.total_seconds() / 60
    df["prog_start_min"] = (start.dt.hour - 16) * 60 + start.dt.minute + start.dt.second / 60
    df["suffix"] = suffix

    valid = t.notna()
    df["hour"] = t.dt.hour
    df["minute_of_evening"] = (t.dt.hour - 16) * 60 + t.dt.minute
    df["dow"] = t.dt.dayofweek
    df["weekend"] = (df["dow"] >= 5).astype(float).where(valid)
    doy = t.dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["holiday"] = t.dt.normalize().isin(_holiday_dates(t)).astype(float).where(valid)
    month_day = t.dt.month * 100 + t.dt.day
    df["xmas"] = ((month_day >= 1220) | (month_day <= 106)).astype(float).where(valid)
    df["summer"] = t.dt.month.isin([7, 8]).astype(float).where(valid)

    # competitor context: the other channels' rows in the same timeslot
    slot = t.astype(str).where(valid, "__invalid__" + df.index.astype(str))
    ctx = (df.assign(_slot=slot).dropna(subset=["channel_id"])
             .groupby(["_slot", "channel_id"])[CTX_VARS].first().unstack("channel_id"))
    ctx.columns = [f"ctx_{v}_ch{c:.0f}" for v, c in ctx.columns]
    ctx = ctx.reindex(columns=CTX_COLS)
    df = df.join(ctx, on=slot)
    df["n_channels"] = df.groupby(slot)["channel_id"].transform("size").astype(float)

    df[FEATURES] = df[FEATURES].astype(float).replace([np.inf, -np.inf], np.nan)
    return df


def clean_train(raw: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows (join fan-out in the raw data) and rows without a valid target."""
    df = raw.drop_duplicates().copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    return df[df[TARGET].between(0, 100)].reset_index(drop=True)


# --------------------------------------------------------------------------- model
class Model:
    def __init__(self) -> None:
        self.seed = 42
        self.params = dict(PARAMS, random_state=self.seed)
        self.model = None
        self.hour_dow = pd.Series(dtype=float)
        self.global_mean = 0.0
        self.half_width = np.nan

    def fit(self, train: pd.DataFrame) -> None:
        df = build_features(clean_train(train))
        ch3 = df[df["channel_id"] == SCORED]
        if len(ch3) == 0:
            ch3 = df
        self.global_mean = float(ch3[TARGET].mean()) if len(ch3) else 0.0
        self.hour_dow = ch3.groupby(["hour", "dow"])[TARGET].mean()
        if len(ch3) >= MIN_ROWS:
            self.model = HistGradientBoostingRegressor(**self.params).fit(ch3[FEATURES], ch3[TARGET])
            self.half_width = self._conformal_half_width(ch3)

    def _conformal_half_width(self, df: pd.DataFrame) -> float:
        """80% interval half-width: 80th pct of |error| on the last 3 months, model fitted before them."""
        t = pd.to_datetime(df["timeslot_datetime_from"], errors="coerce", format="mixed")
        cut = t.max() - pd.DateOffset(months=3)
        inner, cal = df[t < cut], df[t >= cut]
        if len(inner) < MIN_ROWS or len(cal) < 10:
            return float(np.quantile(np.abs(df[TARGET] - df[TARGET].median()), 0.8))
        m = HistGradientBoostingRegressor(**self.params).fit(inner[FEATURES], inner[TARGET])
        return float(np.quantile(np.abs(cal[TARGET] - m.predict(cal[FEATURES])), 0.8))

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        if len(test) == 0:
            return np.zeros(0, dtype=float)
        df = build_features(test.reset_index(drop=True))
        if self.model is not None:
            pred = pd.Series(self.model.predict(df[FEATURES]), index=df.index)
        else:
            pred = pd.Series(np.nan, index=df.index)
        fallback = pd.Series(list(zip(df["hour"], df["dow"])), index=df.index).map(self.hour_dow)
        pred = pred.fillna(fallback).fillna(df[MO3]).fillna(self.global_mean).clip(0, 100)
        pred[df["channel_id"].notna() & (df["channel_id"] != SCORED)] = 0.0  # unscored channels
        return pred.to_numpy(dtype=float)

    def predict_interval(self, test: pd.DataFrame) -> pd.DataFrame:
        """Extra, not used by the scorer: point prediction with an 80% interval per row."""
        p = self.predict(test)
        hw = self.half_width if np.isfinite(self.half_width) else 0.0
        return pd.DataFrame({"pred": p, "lo80": np.clip(p - hw, 0, 100), "hi80": np.clip(p + hw, 0, 100)})
