#!/usr/bin/env python3
"""
Segment item-states for the inventory model: ABC by dollar usage, XYZ by
demand variability, and Syntetos-Boylan demand patterns to flag intermittent
items.

Time windows (full weeks only; M5 weeks run Saturday to Friday):
    history   everything before the segment window, for forecast training
    segment   the 52 weeks before the holdout, used for all classification here
    holdout   the last 52 weeks, reserved for backtesting and never used here

Classifying on the segment window only avoids look-ahead bias: the holdout
year is where the inventory policy gets tested, so nothing from it can leak
into how items are classified.

Inputs:  data/processed/m5_store_daily.parquet and m5_state_daily.parquet (from load_m5.py)
Outputs: data/processed/m5_state_weekly.parquet  item x state x week
         data/processed/segmentation.csv          one row per item-state
         reports/segmentation_summary.md          tables ready for the README

Usage (from the repo root):
    python segment.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

STORE_DAILY = Path("data/processed/m5_store_daily.parquet")
STATE_DAILY = Path("data/processed/m5_state_daily.parquet")
WEEKLY = Path("data/processed/m5_state_weekly.parquet")
SEGMENTS = Path("data/processed/segmentation.csv")
SUMMARY = Path("reports/segmentation_summary.md")

HOLDOUT_WEEKS = 52
SEGMENT_WEEKS = 52
MIN_HISTORY_WEEKS = 52  # training weeks that must exist before the segment window

ABC_CUTOFFS = (0.80, 0.95)  # cumulative dollar share: A to 80%, B to 95%, C the rest
XYZ_CUTOFFS = (0.5, 1.0)    # coefficient of variation of weekly demand (common convention)
ADI_CUTOFF = 1.32           # Syntetos, Boylan and Croston (2005)
CV2_CUTOFF = 0.49


class CheckFailed(Exception):
    pass


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'pass' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    if not ok:
        raise CheckFailed(name)


def tmp(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def build_weekly(con: duckdb.DuckDBPyConnection) -> None:
    print("Rolling days up to full weeks...")
    con.execute(f"""
        CREATE TABLE full_weeks AS
        SELECT wm_yr_wk, MIN(date) AS week_start
        FROM (SELECT DISTINCT wm_yr_wk, date FROM read_parquet('{STORE_DAILY}'))
        GROUP BY wm_yr_wk
        HAVING COUNT(*) = 7
    """)
    # One row per item-state-week in which at least one store had the item on sale.
    con.execute(f"""
        COPY (
            SELECT s.item_id, s.state_id,
                   ANY_VALUE(s.dept_id) AS dept_id,
                   ANY_VALUE(s.cat_id) AS cat_id,
                   s.wm_yr_wk, w.week_start,
                   CAST(SUM(s.units) AS INTEGER) AS units,
                   SUM(s.units * s.unit_cost) AS cost_dollars,
                   SUM(s.units * s.sell_price) AS revenue_dollars,
                   AVG(s.unit_cost) AS avg_unit_cost,
                   AVG(s.sell_price) AS avg_sell_price,
                   COUNT(DISTINCT s.store_id) AS n_stores_listed,
                   COUNT(DISTINCT s.date) FILTER (WHERE s.snap = 1) AS snap_days,
                   BOOL_OR(s.event_name_1 IS NOT NULL) AS has_event
            FROM read_parquet('{STORE_DAILY}') s
            JOIN full_weeks w ON w.wm_yr_wk = s.wm_yr_wk
            GROUP BY s.item_id, s.state_id, s.wm_yr_wk, w.week_start
        ) TO '{tmp(WEEKLY)}' (FORMAT parquet, COMPRESSION zstd)
    """)


def check_weekly(con: duckdb.DuckDBPyConnection) -> None:
    print("Checking weekly table...")
    n_weeks = con.execute("SELECT COUNT(*) FROM full_weeks").fetchone()[0]
    needed = HOLDOUT_WEEKS + SEGMENT_WEEKS + MIN_HISTORY_WEEKS
    check("enough full weeks for history, segment, and holdout", n_weeks >= needed,
          f"{n_weeks} weeks, need {needed}")
    weekly_units = con.execute(f"SELECT SUM(units) FROM read_parquet('{tmp(WEEKLY)}')").fetchone()[0]
    daily_units, partial_days = con.execute(f"""
        SELECT SUM(units) FILTER (WHERE wm_yr_wk IN (SELECT wm_yr_wk FROM full_weeks)),
               COUNT(DISTINCT date) FILTER (WHERE wm_yr_wk NOT IN (SELECT wm_yr_wk FROM full_weeks))
        FROM read_parquet('{STORE_DAILY}')
    """).fetchone()
    check("weekly rollup keeps every unit", weekly_units == daily_units,
          f"{weekly_units:,} vs {daily_units:,}; {partial_days} days in partial weeks left out")


def item_stats(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, dict]:
    n = con.execute("SELECT COUNT(*) FROM full_weeks").fetchone()[0]
    con.execute(f"""
        CREATE TABLE week_window AS
        SELECT wm_yr_wk, week_start,
               CASE WHEN idx >= {n - HOLDOUT_WEEKS} THEN 'holdout'
                    WHEN idx >= {n - HOLDOUT_WEEKS - SEGMENT_WEEKS} THEN 'segment'
                    ELSE 'history' END AS win
        FROM (SELECT *, ROW_NUMBER() OVER (ORDER BY wm_yr_wk) - 1 AS idx FROM full_weeks)
    """)
    windows = {
        win: (start, end)
        for win, start, end in con.execute("""
            SELECT win, MIN(week_start), MAX(week_start) + INTERVAL 6 DAY
            FROM week_window GROUP BY win
        """).fetchall()
    }
    stats = con.execute(f"""
        SELECT f.item_id, f.state_id,
               ANY_VALUE(f.dept_id) AS dept_id,
               ANY_VALUE(f.cat_id) AS cat_id,
               COUNT(*) FILTER (WHERE win = 'segment') AS segment_weeks_listed,
               COUNT(*) FILTER (WHERE win = 'holdout') AS holdout_weeks_listed,
               MIN(n_stores_listed) FILTER (WHERE win <> 'history') AS min_stores,
               MAX(n_stores_listed) FILTER (WHERE win <> 'history') AS max_stores,
               COALESCE(SUM(cost_dollars) FILTER (WHERE win = 'segment'), 0) AS annual_cost_dollars,
               AVG(units) FILTER (WHERE win = 'segment') AS mean_weekly_units,
               STDDEV_POP(units) FILTER (WHERE win = 'segment') AS sd_weekly_units,
               COUNT(*) FILTER (WHERE win = 'segment' AND units > 0) AS nonzero_weeks,
               AVG(units) FILTER (WHERE win = 'segment' AND units > 0) AS mean_nonzero_size,
               STDDEV_POP(units) FILTER (WHERE win = 'segment' AND units > 0) AS sd_nonzero_size
        FROM read_parquet('{tmp(WEEKLY)}') f
        JOIN week_window w ON w.wm_yr_wk = f.wm_yr_wk
        GROUP BY f.item_id, f.state_id
    """).df()
    seg_start, seg_end = windows["segment"]
    zero_days = con.execute(f"""
        SELECT item_id, state_id, AVG(CASE WHEN units = 0 THEN 1.0 ELSE 0.0 END) AS pct_zero_days
        FROM read_parquet('{STATE_DAILY}')
        WHERE date BETWEEN DATE '{seg_start}' AND DATE '{seg_end:%Y-%m-%d}'
        GROUP BY item_id, state_id
    """).df()
    return stats.merge(zero_days, on=["item_id", "state_id"], how="left"), windows


def classify(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Why an item-state is left out of the model, if it is (first reason wins).
    reasons = [
        (df["segment_weeks_listed"] < SEGMENT_WEEKS, "short history"),
        (df["holdout_weeks_listed"] < HOLDOUT_WEEKS, "not listed through holdout"),
        (df["min_stores"] != df["max_stores"], "store count changed"),
        (df["mean_weekly_units"].fillna(0) == 0, "no demand"),
    ]
    df["exclusion"] = np.select([c for c, _ in reasons], [r for _, r in reasons], default="")
    df["in_model"] = df["exclusion"] == ""

    # ABC within each state, since each state's DC manages its own inventory.
    # An item is A if the items ranked above it hold less than 80% of dollars.
    df = df.sort_values(["state_id", "annual_cost_dollars"], ascending=[True, False])
    total = df.groupby("state_id")["annual_cost_dollars"].transform("sum")
    df["dollar_share"] = df["annual_cost_dollars"] / total
    prior = df.groupby("state_id")["dollar_share"].cumsum() - df["dollar_share"]
    a, b = ABC_CUTOFFS
    df["abc"] = np.select([prior < a, prior < b], ["A", "B"], default="C")

    # XYZ on the variability of weekly demand.
    df["cv"] = (df["sd_weekly_units"] / df["mean_weekly_units"]).replace(np.inf, np.nan)
    x, y = XYZ_CUTOFFS
    df["xyz"] = np.select([df["cv"] < x, df["cv"] < y, df["cv"] >= y], ["X", "Y", "Z"], default="")

    # Syntetos-Boylan: how often demand occurs (ADI) and how much its size varies (CV squared).
    df["adi"] = (df["segment_weeks_listed"] / df["nonzero_weeks"]).replace(np.inf, np.nan)
    df["cv2"] = (df["sd_nonzero_size"] / df["mean_nonzero_size"]) ** 2
    frequent = df["adi"] < ADI_CUTOFF
    steady = df["cv2"] < CV2_CUTOFF
    df["demand_pattern"] = np.select(
        [df["segment_weeks_listed"] == 0, df["nonzero_weeks"] == 0, frequent & steady, frequent, steady],
        ["", "no demand", "smooth", "erratic", "intermittent"],
        default="lumpy",
    )
    df["pct_zero_weeks"] = 1 - df["nonzero_weeks"] / df["segment_weeks_listed"].replace(0, np.nan)

    columns = [
        "item_id", "state_id", "dept_id", "cat_id", "in_model", "exclusion",
        "abc", "xyz", "demand_pattern", "annual_cost_dollars", "dollar_share",
        "mean_weekly_units", "cv", "adi", "cv2", "pct_zero_weeks", "pct_zero_days",
        "segment_weeks_listed", "holdout_weeks_listed", "min_stores", "max_stores",
    ]
    return df[columns].sort_values(["state_id", "annual_cost_dollars"], ascending=[True, False])


def check_segments(seg: pd.DataFrame) -> None:
    print("Checking segmentation...")
    share_sums = seg.groupby("state_id")["dollar_share"].sum()
    check("dollar shares sum to 100% in each state", bool(np.allclose(share_sums, 1.0)),
          ", ".join(f"{s} {v:.4f}" for s, v in share_sums.items()))
    m = seg[seg["in_model"]]
    check("at least one item-state is modeled", len(m) > 0, f"{len(m):,}")
    unlabeled = int(((m["xyz"] == "") | (m["demand_pattern"] == "no demand")).sum())
    check("every modeled item-state has a full classification", unlabeled == 0, f"{unlabeled} missing")


def matrix(m: pd.DataFrame, values: pd.Series | None = None) -> pd.DataFrame:
    if values is None:
        t = pd.crosstab(m["abc"], m["xyz"], margins=True, margins_name="Total")
    else:
        t = pd.crosstab(m["abc"], m["xyz"], values=values, aggfunc="sum",
                        margins=True, margins_name="Total") / values.sum()
    return t.reindex(index=["A", "B", "C", "Total"], columns=["X", "Y", "Z", "Total"]).fillna(0)


def md_table(t: pd.DataFrame, fmt) -> str:
    """Markdown table; fmt is one formatter, or a dict of formatters by column."""
    fmts = fmt if isinstance(fmt, dict) else {c: fmt for c in t.columns}
    lines = ["| | " + " | ".join(map(str, t.columns)) + " |", "|---" * (len(t.columns) + 1) + "|"]
    for idx, row in t.iterrows():
        lines.append(f"| **{idx}** | " + " | ".join(fmts[c](row[c]) for c in t.columns) + " |")
    return "\n".join(lines)


def summarize(seg: pd.DataFrame, windows: dict) -> None:
    m = seg[seg["in_model"]]
    counts = matrix(m)
    dollars = matrix(m, m["annual_cost_dollars"])
    patterns = (m["demand_pattern"].value_counts()
                .reindex(["smooth", "erratic", "intermittent", "lumpy"]).fillna(0).astype(int))
    excluded = seg.loc[~seg["in_model"], "exclusion"].value_counts()
    (seg_start, seg_end), (hold_start, hold_end) = windows["segment"], windows["holdout"]

    print(f"\nSegment window {seg_start} to {seg_end:%Y-%m-%d}; holdout {hold_start} to {hold_end:%Y-%m-%d}")
    print(f"Modeled {len(m):,} of {len(seg):,} item-states. Excluded: "
          + ", ".join(f"{r} {n:,}" for r, n in excluded.items()))
    print("\nItem-states by ABC/XYZ:")
    print(counts.astype(int).to_string())
    print("\nShare of annual cost dollars:")
    print(dollars.map(lambda v: f"{v:.1%}").to_string())
    print("\nDemand patterns (weekly):")
    print(patterns.to_string())
    print(f"\nZero-sales share: {m['pct_zero_days'].mean():.1%} of days vs "
          f"{m['pct_zero_weeks'].mean():.1%} of weeks")

    pattern_table = pd.DataFrame({"Item-states": patterns, "Share": patterns / patterns.sum()})
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join([
        "# Segmentation summary",
        "",
        f"- Segment window: {seg_start} to {seg_end:%Y-%m-%d} ({SEGMENT_WEEKS} weeks)",
        f"- Holdout, not used for classification: {hold_start} to {hold_end:%Y-%m-%d} ({HOLDOUT_WEEKS} weeks)",
        f"- Item-states modeled: {len(m):,} of {len(seg):,}",
        "- Excluded: " + ", ".join(f"{r} ({n:,})" for r, n in excluded.items()),
        "",
        "## Item-states by ABC/XYZ",
        "",
        md_table(counts, lambda v: f"{int(v):,}"),
        "",
        "## Share of annual cost dollars",
        "",
        md_table(dollars, lambda v: f"{v:.1%}"),
        "",
        "## Weekly demand patterns (Syntetos-Boylan)",
        "",
        md_table(pattern_table, {"Item-states": lambda v: f"{int(v):,}", "Share": lambda v: f"{v:.1%}"}),
        "",
        f"On average, {m['pct_zero_days'].mean():.1%} of days have zero sales, "
        f"versus {m['pct_zero_weeks'].mean():.1%} of weeks.",
        "",
    ]))


def main() -> None:
    missing = [p for p in (STORE_DAILY, STATE_DAILY) if not p.exists()]
    if missing:
        sys.exit("Missing inputs (run load_m5.py first):\n  " + "\n  ".join(map(str, missing)))

    con = duckdb.connect()
    Path("data/tmp").mkdir(parents=True, exist_ok=True)
    con.execute("SET temp_directory = 'data/tmp'")
    con.execute("SET preserve_insertion_order = false")

    try:
        build_weekly(con)
        check_weekly(con)
        stats, windows = item_stats(con)
        seg = classify(stats)
        check_segments(seg)
    except CheckFailed as e:
        tmp(WEEKLY).unlink(missing_ok=True)
        sys.exit(f"\nCheck failed ({e}), so nothing was saved.")

    os.replace(tmp(WEEKLY), WEEKLY)
    seg.to_csv(SEGMENTS, index=False)
    summarize(seg, windows)
    print(f"\nSaved {WEEKLY}, {SEGMENTS}, and {SUMMARY}")


if __name__ == "__main__":
    main()
