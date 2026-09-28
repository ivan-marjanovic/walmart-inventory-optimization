#!/usr/bin/env python3
"""
Walk-forward demand forecasts for the inventory model.

One global LightGBM model forecasts 1 to 5 weeks ahead for every modeled
item-state. It uses a Tweedie objective, which suits zero-heavy weekly
demand. Five weeks covers the longest protection interval in the stress test
(a 4-week lead time plus a 1-week review period).

Walk-forward design, so no forecast uses information from its own future:
    - Forecasts are made from every week in the segment and holdout windows.
    - The model is retrained every 13 weeks, on data observed by that date only.
    - Features use data at or before the forecast origin, except planned
      inputs known in advance: the target week's shelf price, SNAP days,
      and events.

Segment-window forecasts give the policy step a full year of forecast errors
to size safety stock from. Holdout forecasts are the test.

Baselines: 4- and 13-week moving averages, and Croston's method with the
Syntetos-Boylan correction (SBA), the standard for intermittent demand.

Inputs:  data/processed/m5_state_weekly.parquet, data/processed/segmentation.csv
Outputs: data/processed/forecasts.parquet  one row per item-state, origin, and horizon
         reports/forecast_accuracy.md       accuracy tables for the README

Usage (from the repo root):
    python forecast.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

WEEKLY = Path("data/processed/m5_state_weekly.parquet")
SEGMENTS = Path("data/processed/segmentation.csv")
FORECASTS = Path("data/processed/forecasts.parquet")
REPORT = Path("reports/forecast_accuracy.md")

HOLDOUT_WEEKS = 52        # must match segment.py
SEGMENT_WEEKS = 52
MAX_HORIZON = 5           # 4-week lead time (stress test max) + 1-week review
BASE_INTERVAL = 3         # base case: 2-week lead time + 1-week review
RETRAIN_EVERY = 13
TRAIN_WINDOW_WEEKS = 156  # each model trains on the most recent three years of origins
MIN_ORIGIN = 13           # skip the first quarter, when rolling features are thin
SBA_ALPHA = 0.1

PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.2,
    "learning_rate": 0.1,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": 42,
    "verbose": -1,
}
MAX_ROUNDS = 1000
EARLY_STOPPING = 30

ORIGIN_FEATURES = ["lag_0", "lag_1", "lag_2", "lag_3", "mean_4", "mean_8", "mean_13",
                   "mean_26", "mean_52", "std_13", "zero_share_13", "age_weeks"]
TARGET_FEATURES = ["horizon", "lag_52_target", "price_target", "price_ratio_13",
                   "snap_days_target", "event_target", "week_of_year_target", "n_stores_target"]
STATIC_FEATURES = ["state", "cat", "dept"]
FEATURES = ORIGIN_FEATURES + TARGET_FEATURES + STATIC_FEATURES
METHODS = ["lgbm", "ma4", "ma13", "sba"]


class CheckFailed(Exception):
    pass


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'pass' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    if not ok:
        raise CheckFailed(name)


def load_panel() -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Dense week x series arrays for the modeled item-states. NaN means not on sale."""
    seg = pd.read_csv(SEGMENTS)
    series = (seg[seg["in_model"]]
              [["item_id", "state_id", "dept_id", "cat_id", "abc", "xyz", "demand_pattern"]]
              .sort_values(["state_id", "item_id"]).reset_index(drop=True))
    weeks = duckdb.sql(f"SELECT DISTINCT week_start FROM read_parquet('{WEEKLY}') ORDER BY 1") \
        .df()["week_start"].to_numpy()
    weekly = duckdb.sql(f"""
        SELECT item_id, state_id, week_start, units, avg_sell_price, snap_days,
               CAST(has_event AS INTEGER) AS has_event, n_stores_listed
        FROM read_parquet('{WEEKLY}')
    """).df().merge(series[["item_id", "state_id"]], on=["item_id", "state_id"])

    s_idx = pd.MultiIndex.from_frame(series[["item_id", "state_id"]]) \
        .get_indexer(pd.MultiIndex.from_frame(weekly[["item_id", "state_id"]]))
    w_idx = np.searchsorted(weeks, weekly["week_start"].to_numpy())
    shape = (len(weeks), len(series))

    def grid(col: str) -> np.ndarray:
        a = np.full(shape, np.nan, dtype=np.float32)
        a[w_idx, s_idx] = weekly[col].to_numpy(dtype=np.float32)
        return a

    panel = {c: grid(c) for c in ["units", "avg_sell_price", "snap_days", "has_event", "n_stores_listed"]}
    return series, weeks, panel


def rolling(a: np.ndarray, k: int, stat: str = "mean", min_periods: int = 1) -> np.ndarray:
    r = pd.DataFrame(a).rolling(k, min_periods=min_periods)
    return getattr(r, stat)().to_numpy(dtype=np.float32)


def lag(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(a, np.nan)
    out[k:] = a[:-k]
    return out


def origin_features(panel: dict) -> dict:
    """Features known at the end of each week, for every series."""
    u = panel["units"]
    f = {f"lag_{k}": (u if k == 0 else lag(u, k)) for k in range(4)}
    for k in (4, 8, 13, 26, 52):
        f[f"mean_{k}"] = rolling(u, k)
    f["std_13"] = rolling(u, 13, "std", min_periods=2)
    zero = np.where(np.isnan(u), np.nan, (u == 0).astype(np.float32))
    f["zero_share_13"] = rolling(zero, 13)
    first = np.argmax(~np.isnan(u), axis=0)
    age = (np.arange(u.shape[0])[:, None] - first[None, :]).astype(np.float32)
    f["age_weeks"] = np.where(age < 0, np.nan, age)
    f["price_mean_13"] = rolling(panel["avg_sell_price"], 13)  # helper for the price ratio
    return f


def sba(units: np.ndarray, alpha: float = SBA_ALPHA) -> np.ndarray:
    """Croston's method with the Syntetos-Boylan correction, as of each week."""
    W, S = units.shape
    size = np.full(S, np.nan)      # smoothed nonzero demand size
    interval = np.full(S, np.nan)  # smoothed weeks between nonzero demands
    since = np.ones(S)             # weeks since the last nonzero demand, counting this one
    out = np.zeros((W, S), dtype=np.float32)
    for w in range(W):
        x = units[w]
        listed = ~np.isnan(x)
        pos = listed & (x > 0)
        first = pos & np.isnan(size)
        size[first], interval[first] = x[first], since[first]
        upd = pos & ~first
        size[upd] += alpha * (x[upd] - size[upd])
        interval[upd] += alpha * (since[upd] - interval[upd])
        since[pos] = 1
        since[listed & ~pos] += 1
        out[w] = np.nan_to_num((1 - alpha / 2) * size / interval)
    return out


def build_rows(series: pd.DataFrame, weeks: np.ndarray, panel: dict, f: dict) -> tuple[np.ndarray, dict]:
    """Stack one row per (series, origin, horizon) where the item is on sale at both ends."""
    u, price = panel["units"], panel["avg_sell_price"]
    W = u.shape[0]
    listed = ~np.isnan(u)
    woy = ((pd.DatetimeIndex(weeks).dayofyear.to_numpy() - 1) // 7 + 1).astype(np.float32)
    static = {c: pd.Categorical(series[col]).codes.astype(np.float32)
              for c, col in [("state", "state_id"), ("cat", "cat_id"), ("dept", "dept_id")]}

    index = []
    for h in range(1, MAX_HORIZON + 1):
        origins = np.arange(MIN_ORIGIN, W - h)
        oi, s = np.nonzero(listed[origins] & listed[origins + h])
        index.append((h, origins[oi], s))
    n = sum(len(o) for _, o, _ in index)
    X = np.empty((n, len(FEATURES)), dtype=np.float32)  # filled in place to keep peak memory down
    meta = {k: np.empty(n, dtype=np.int32) for k in ["s", "o", "h", "t"]}
    meta["y"] = np.empty(n, dtype=np.float32)

    row = 0
    for h, o, s in index:
        t = o + h
        rows = slice(row, row + len(o))
        cols = [f[name][o, s] for name in ORIGIN_FEATURES] + [
            np.full(len(o), h, dtype=np.float32),
            np.where(t >= 52, u[np.maximum(t - 52, 0), s], np.nan),  # same week last year
            price[t, s],
            price[t, s] / f["price_mean_13"][o, s],
            panel["snap_days"][t, s],
            panel["has_event"][t, s],
            woy[t],
            panel["n_stores_listed"][t, s],
        ] + [static[c][s] for c in STATIC_FEATURES]
        for j, col in enumerate(cols):
            X[rows, j] = col
        for k, v in zip(["s", "o", "h", "t", "y"], [s, o, h, t, u[t, s]]):
            meta[k][rows] = v
        row += len(o)
    return X, meta


def walk_forward(X: np.ndarray, m: dict, W: int) -> tuple[np.ndarray, lgb.Booster, int]:
    first_eval = W - HOLDOUT_WEEKS - SEGMENT_WEEKS - 1  # its 1-week-ahead target opens the segment window
    starts = list(range(first_eval, W - 1, RETRAIN_EVERY))
    cat_idx = [FEATURES.index(c) for c in STATIC_FEATURES]
    pred = np.full(len(X), np.nan, dtype=np.float32)
    rounds = None

    def dataset(mask: np.ndarray) -> lgb.Dataset:
        return lgb.Dataset(X[mask], m["y"][mask], feature_name=FEATURES,
                           categorical_feature=cat_idx, free_raw_data=True)

    for k, o_k in enumerate(starts):
        t0 = time.time()
        train = (m["t"] <= o_k) & (m["o"] >= o_k - TRAIN_WINDOW_WEEKS)
        check(f"model {k + 1} trains only on data observed by its start week",
              int(m["t"][train].max()) <= o_k)
        if rounds is None:
            # Pick the number of trees once, using the 13 weeks before the first origin as validation.
            fit, val = train & (m["t"] <= o_k - 13), train & (m["t"] > o_k - 13)
            probe = lgb.train(PARAMS, dataset(fit), MAX_ROUNDS, valid_sets=[dataset(val)],
                              callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)])
            rounds = max(probe.best_iteration, 50)
            print(f"  early stopping picked {rounds} trees")
        booster = lgb.train(PARAMS, dataset(train), rounds)
        block = (m["o"] >= o_k) & (m["o"] < o_k + RETRAIN_EVERY)
        pred[block] = booster.predict(X[block])
        print(f"  model {k + 1}/{len(starts)}: {train.sum():,} training rows, "
              f"{block.sum():,} forecasts, {time.time() - t0:.0f}s")
    return pred, booster, first_eval


def interval_totals(df: pd.DataFrame, weeks: int) -> pd.DataFrame:
    """Sum forecasts and actuals over the first `weeks` horizons, where all are present."""
    d = df[df["h"] <= weeks]
    g = d.groupby(["item_id", "state_id", "origin_start"], observed=True)
    tot = g[["actual"] + METHODS].sum()
    return tot[g.size() == weeks].reset_index()


def accuracy(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    def one(d: pd.DataFrame) -> pd.Series:
        a = d["actual"].sum()
        return pd.Series({mth: np.abs(d["actual"] - d[mth]).sum() / a for mth in METHODS})
    return one(df).to_frame("All").T if by is None else df.groupby(by, observed=True).apply(one)


def bias(df: pd.DataFrame) -> pd.Series:
    a = df["actual"].sum()
    return pd.Series({mth: (df[mth] - df["actual"]).sum() / a for mth in METHODS})


def md_table(t: pd.DataFrame) -> str:
    lines = ["| | " + " | ".join(t.columns) + " |", "|---" * (len(t.columns) + 1) + "|"]
    for idx, row in t.iterrows():
        lines.append(f"| **{idx}** | " + " | ".join(f"{v:.1%}" for v in row) + " |")
    return "\n".join(lines)


def report(out: pd.DataFrame, series: pd.DataFrame, booster: lgb.Booster, rounds_note: str) -> None:
    hold = out[out["window"] == "holdout"]
    week1 = hold[hold["h"] == 1]
    base = interval_totals(hold, BASE_INTERVAL).merge(
        series[["item_id", "state_id", "abc", "demand_pattern"]], on=["item_id", "state_id"])

    overall = pd.DataFrame({
        "WMAPE, 1 week ahead": accuracy(week1).iloc[0],
        "Bias, 1 week ahead": bias(week1),
        f"WMAPE, {BASE_INTERVAL}-week total": accuracy(base).iloc[0],
        f"Bias, {BASE_INTERVAL}-week total": bias(base),
    }).T
    by_abc = accuracy(base, "abc")
    by_pattern = (accuracy(base, "demand_pattern")
                  .reindex(["smooth", "erratic", "intermittent", "lumpy"]).dropna(how="all"))
    gain = pd.Series(booster.feature_importance("gain"), index=FEATURES)
    top = (gain / gain.sum()).sort_values(ascending=False).head(10)

    labels = {"lgbm": "LightGBM", "ma4": "4-wk avg", "ma13": "13-wk avg", "sba": "Croston (SBA)"}
    tables = [(t.rename(columns=labels)) for t in (overall, by_abc, by_pattern)]
    print("\nHoldout accuracy (lower WMAPE is better; bias above 0 means over-forecasting):")
    for title, t in zip(["Overall", f"WMAPE of {BASE_INTERVAL}-week totals by ABC",
                         f"WMAPE of {BASE_INTERVAL}-week totals by demand pattern"], tables):
        print(f"\n{title}:\n" + t.map(lambda v: f"{v:.1%}").to_string())
    print("\nTop features by share of gain:\n" + top.map(lambda v: f"{v:.1%}").to_string())

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join([
        "# Forecast accuracy (holdout year)",
        "",
        f"- Walk-forward: retrained every {RETRAIN_EVERY} weeks on data observed by each forecast origin",
        f"- {rounds_note}",
        f"- {BASE_INTERVAL}-week totals match the base-case protection interval (2-week lead time + 1-week review)",
        "- WMAPE = sum of absolute errors / sum of actual demand; bias above 0 means over-forecasting",
        "",
        "## Overall", "", md_table(tables[0]), "",
        f"## WMAPE of {BASE_INTERVAL}-week totals by ABC class", "", md_table(tables[1]), "",
        f"## WMAPE of {BASE_INTERVAL}-week totals by demand pattern", "", md_table(tables[2]), "",
        "## Top LightGBM features (share of gain)", "",
        "\n".join(f"- {name}: {v:.1%}" for name, v in top.items()), "",
    ]))


def main() -> None:
    missing = [p for p in (WEEKLY, SEGMENTS) if not p.exists()]
    if missing:
        sys.exit("Missing inputs (run segment.py first):\n  " + "\n  ".join(map(str, missing)))
    start = time.time()

    print("Loading weekly panel...")
    series, weeks, panel = load_panel()
    W = len(weeks)
    print(f"  {len(series):,} item-states x {W} weeks")

    print("Building features and baselines...")
    f = origin_features(panel)
    sba_fc = sba(panel["units"])
    X, m = build_rows(series, weeks, panel, f)
    print(f"  {len(X):,} rows x {X.shape[1]} features")

    print("Walk-forward training...")
    try:
        pred, booster, first_eval = walk_forward(X, m, W)
        ev = m["o"] >= first_eval
        s, o, t = m["s"][ev], m["o"][ev], m["t"][ev]
        out = pd.DataFrame({
            "item_id": pd.Categorical(series["item_id"].to_numpy()[s]),
            "state_id": pd.Categorical(series["state_id"].to_numpy()[s]),
            "origin_start": weeks[o],
            "target_start": weeks[t],
            "h": m["h"][ev].astype(np.int8),
            "window": np.where(t >= W - HOLDOUT_WEEKS, "holdout", "segment"),
            "actual": m["y"][ev],
            "lgbm": pred[ev],
            "ma4": f["mean_4"][o, s],
            "ma13": f["mean_13"][o, s],
            "sba": sba_fc[o, s],
        })
        print("Checking forecasts...")
        check("every forecast row has a LightGBM prediction", int(out["lgbm"].isna().sum()) == 0)
        check("no negative forecasts", bool((out[METHODS] >= 0).all().all()))
        per_window = out.groupby("window").size()
        check("forecasts cover both the segment and holdout years",
              set(per_window.index) == {"segment", "holdout"},
              ", ".join(f"{w} {n:,}" for w, n in per_window.items()))
    except CheckFailed as e:
        sys.exit(f"\nCheck failed ({e}), so nothing was saved.")

    out.to_parquet(FORECASTS, index=False)
    report(out, series, booster, f"Tree count chosen by early stopping on pre-backtest data: "
                                 f"{booster.current_iteration()}")
    print(f"\nSaved {FORECASTS} and {REPORT}. Done in {(time.time() - start) / 60:.1f} min.")


if __name__ == "__main__":
    main()
