#!/usr/bin/env python3
"""
Load the M5 data into analysis-ready Parquet tables, with unit costs from the
Walmart 10-K calibration.

Inputs (run walmart_10k_calibration.py first):
    data/raw/m5/calendar.csv
    data/raw/m5/sell_prices.csv
    data/raw/m5/sales_train_evaluation.csv
    data/processed/walmart_calibration.csv

Outputs:
    data/processed/m5_store_daily.parquet  item x store x day, only days the item was on sale
    data/processed/m5_state_daily.parquet  item x state x day, stores rolled up (the "DC" view)

Unit cost = that fiscal year's Walmart cost ratio * the item's shelf price that week.
Outputs are written to temporary files and only saved if every check passes.

Usage (from the repo root):
    python load_m5.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import duckdb

RAW = Path("data/raw/m5")
CALENDAR = RAW / "calendar.csv"
PRICES = RAW / "sell_prices.csv"
SALES = RAW / "sales_train_evaluation.csv"
CALIBRATION = Path("data/processed/walmart_calibration.csv")
OUT_STORE = Path("data/processed/m5_store_daily.parquet")
OUT_STATE = Path("data/processed/m5_state_daily.parquet")
SPILL_DIR = Path("data/tmp")  # DuckDB spills here if a step outgrows memory

EXPECTED_SERIES = 30_490  # 3,049 items x 10 stores
EXPECTED_DAYS = 1_941     # d_1 (2011-01-29) through d_1941 (2016-05-22)
MAX_DROPPED_UNIT_SHARE = 0.001  # units sold on days with no shelf price


class CheckFailed(Exception):
    pass


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'pass' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    if not ok:
        raise CheckFailed(name)


def tmp(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def load_inputs(con: duckdb.DuckDBPyConnection) -> list[str]:
    print("Loading inputs...")
    # Walmart's fiscal year ends Jan 31: February onward belongs to next year's FY.
    con.execute(f"""
        CREATE TABLE calendar AS
        SELECT d,
               CAST(date AS DATE) AS date,
               wm_yr_wk,
               CASE WHEN month(CAST(date AS DATE)) >= 2
                    THEN year(CAST(date AS DATE)) + 1
                    ELSE year(CAST(date AS DATE)) END AS fiscal_year,
               event_name_1, event_type_1, event_name_2, event_type_2,
               snap_CA, snap_TX, snap_WI
        FROM read_csv('{CALENDAR}', header = true)
    """)
    con.execute(f"""
        CREATE TABLE prices AS
        SELECT store_id, item_id, wm_yr_wk, sell_price
        FROM read_csv('{PRICES}', header = true)
    """)
    con.execute(f"""
        CREATE TABLE calibration AS
        SELECT fiscal_year, cost_ratio FROM read_csv('{CALIBRATION}', header = true)
    """)
    con.execute(f"CREATE TABLE sales_wide AS SELECT * FROM read_csv('{SALES}', header = true)")
    return [row[0] for row in con.execute("DESCRIBE sales_wide").fetchall() if row[0].startswith("d_")]


def check_inputs(con: duckdb.DuckDBPyConnection, day_cols: list[str]) -> None:
    print("Checking inputs...")
    n_series = con.execute("SELECT COUNT(*) FROM sales_wide").fetchone()[0]
    check("sales has one row per item-store", n_series == EXPECTED_SERIES,
          f"{n_series:,} rows, expected {EXPECTED_SERIES:,}")
    check("sales has every day column", len(day_cols) == EXPECTED_DAYS,
          f"{len(day_cols):,} days, expected {EXPECTED_DAYS:,}")

    # Lookup tables must be unique on their join keys, or the joins would duplicate sales.
    dup_days = con.execute("SELECT COUNT(*) - COUNT(DISTINCT d) FROM calendar").fetchone()[0]
    check("calendar is unique by day", dup_days == 0, f"{dup_days} duplicates")
    dup_prices = con.execute("""
        SELECT COUNT(*) - COUNT(DISTINCT (store_id, item_id, wm_yr_wk)) FROM prices
    """).fetchone()[0]
    check("prices are unique by store, item, and week", dup_prices == 0, f"{dup_prices} duplicates")

    known = {row[0] for row in con.execute("SELECT d FROM calendar").fetchall()}
    missing = [d for d in day_cols if d not in known]
    check("every sales day is in the calendar", not missing, f"{len(missing)} missing")

    needed = {row[0] for row in con.execute(
        "SELECT DISTINCT fiscal_year FROM calendar WHERE d IN (SELECT UNNEST($1))", [day_cols]
    ).fetchall()}
    ratios = dict(con.execute("SELECT fiscal_year, cost_ratio FROM calibration").fetchall())
    gaps = sorted(needed - ratios.keys())
    check("calibration covers every fiscal year in the sales window", not gaps,
          f"missing FY {gaps}" if gaps else f"FY{min(needed)}-FY{max(needed)}")
    bad = [fy for fy, r in ratios.items() if not 0 < r < 1]
    check("cost ratios are between 0 and 1", not bad, f"bad years: {bad}" if bad else "")


def build_store_table(con: duckdb.DuckDBPyConnection) -> None:
    print("Building item x store x day table (the slow step, usually a few minutes)...")
    # The inner join to prices keeps only days the item was on sale at that store.
    # Zeros before an item launches are not demand, so they shouldn't reach the model.
    con.execute(f"""
        COPY (
            WITH sales_long AS (
                UNPIVOT sales_wide ON COLUMNS('^d_[0-9]+$') INTO NAME d VALUE units
            )
            SELECT c.date, c.fiscal_year, c.wm_yr_wk,
                   s.item_id, s.dept_id, s.cat_id, s.store_id, s.state_id,
                   CAST(s.units AS INTEGER) AS units,
                   p.sell_price,
                   k.cost_ratio * p.sell_price AS unit_cost,
                   CASE s.state_id WHEN 'CA' THEN c.snap_CA
                                   WHEN 'TX' THEN c.snap_TX
                                   WHEN 'WI' THEN c.snap_WI END AS snap,
                   c.event_name_1, c.event_type_1, c.event_name_2, c.event_type_2
            FROM sales_long s
            JOIN calendar c ON c.d = s.d
            JOIN prices p ON p.store_id = s.store_id
                         AND p.item_id = s.item_id
                         AND p.wm_yr_wk = c.wm_yr_wk
            JOIN calibration k ON k.fiscal_year = c.fiscal_year
        ) TO '{tmp(OUT_STORE)}' (FORMAT parquet, COMPRESSION zstd)
    """)


def check_store_table(con: duckdb.DuckDBPyConnection, day_cols: list[str]) -> int:
    print("Checking item x store x day table...")
    store = f"read_parquet('{tmp(OUT_STORE)}')"
    n_rows, n_neg, n_null_cost, n_bad_cost, total_units = con.execute(f"""
        SELECT COUNT(*),
               COUNT(*) FILTER (WHERE units < 0),
               COUNT(*) FILTER (WHERE unit_cost IS NULL),
               COUNT(*) FILTER (WHERE unit_cost <= 0 OR unit_cost >= sell_price),
               SUM(units)
        FROM {store}
    """).fetchone()
    raw_units = con.execute(
        "SELECT SUM(units) FROM (UNPIVOT sales_wide ON COLUMNS('^d_[0-9]+$') INTO NAME d VALUE units)"
    ).fetchone()[0]

    max_rows = EXPECTED_SERIES * len(day_cols)
    check("row count is plausible", 0 < n_rows <= max_rows,
          f"{n_rows:,} of {max_rows:,} item-store-days had the item on sale")
    check("no negative sales", n_neg == 0, f"{n_neg} rows")
    check("every row has a unit cost", n_null_cost == 0, f"{n_null_cost} rows")
    check("unit cost is positive and below shelf price", n_bad_cost == 0, f"{n_bad_cost} rows")
    dropped = raw_units - total_units
    share = dropped / raw_units if raw_units else 0
    check("almost no sales fell on days without a price", share <= MAX_DROPPED_UNIT_SHARE,
          f"{dropped:,} of {raw_units:,} units ({share:.4%})")
    return total_units


def build_state_table(con: duckdb.DuckDBPyConnection) -> None:
    print("Rolling stores up to states...")
    con.execute(f"""
        COPY (
            SELECT date, item_id, state_id,
                   ANY_VALUE(fiscal_year) AS fiscal_year,
                   ANY_VALUE(wm_yr_wk) AS wm_yr_wk,
                   ANY_VALUE(dept_id) AS dept_id,
                   ANY_VALUE(cat_id) AS cat_id,
                   CAST(SUM(units) AS INTEGER) AS units,
                   AVG(sell_price) AS avg_sell_price,
                   AVG(unit_cost) AS avg_unit_cost,
                   COUNT(*) AS n_stores_listed,
                   ANY_VALUE(snap) AS snap,
                   ANY_VALUE(event_name_1) AS event_name_1,
                   ANY_VALUE(event_type_1) AS event_type_1,
                   ANY_VALUE(event_name_2) AS event_name_2,
                   ANY_VALUE(event_type_2) AS event_type_2
            FROM read_parquet('{tmp(OUT_STORE)}')
            GROUP BY date, item_id, state_id
        ) TO '{tmp(OUT_STATE)}' (FORMAT parquet, COMPRESSION zstd)
    """)


def check_state_table(con: duckdb.DuckDBPyConnection, store_units: int) -> None:
    print("Checking item x state x day table...")
    state_units, max_stores = con.execute(
        f"SELECT SUM(units), MAX(n_stores_listed) FROM read_parquet('{tmp(OUT_STATE)}')"
    ).fetchone()
    check("rollup keeps every unit", state_units == store_units,
          f"{state_units:,} vs {store_units:,}")
    check("no state has more than 4 stores listed", max_stores <= 4, f"max {max_stores}")


def main() -> None:
    missing = [p for p in (CALENDAR, PRICES, SALES, CALIBRATION) if not p.exists()]
    if missing:
        hint = "\nRun walmart_10k_calibration.py first." if CALIBRATION in missing else ""
        sys.exit("Missing input files:\n  " + "\n  ".join(map(str, missing)) + hint)

    start = time.time()
    SPILL_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET temp_directory = '{SPILL_DIR}'")
    con.execute("SET preserve_insertion_order = false")

    try:
        day_cols = load_inputs(con)
        check_inputs(con, day_cols)
        build_store_table(con)
        store_units = check_store_table(con, day_cols)
        build_state_table(con)
        check_state_table(con, store_units)
    except CheckFailed as e:
        for p in (tmp(OUT_STORE), tmp(OUT_STATE)):
            p.unlink(missing_ok=True)
        sys.exit(f"\nCheck failed ({e}), so nothing was saved.")

    for out in (OUT_STORE, OUT_STATE):
        os.replace(tmp(out), out)

    print()
    for out in (OUT_STORE, OUT_STATE):
        n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out}')").fetchone()[0]
        print(f"Saved {out}  ({n:,} rows, {out.stat().st_size / 1e6:,.1f} MB)")
    first, last, items = con.execute(
        f"SELECT MIN(date), MAX(date), COUNT(DISTINCT item_id) FROM read_parquet('{OUT_STORE}')"
    ).fetchone()
    print(f"Sales window {first} to {last}, {items:,} items. Done in {time.time() - start:.0f}s.")


if __name__ == "__main__":
    main()
