#!/usr/bin/env python3
"""
Walmart 10-K calibration for the inventory planning model.

Pulls Walmart's annual net sales, cost of sales, and year-end inventories from
SEC's XBRL API, then derives two parameters:

    cost_ratio = cost of sales / net sales
        -> unit cost of an item = cost_ratio * its M5 shelf price
    dio_days   = average inventory / cost of sales * 365
        -> the rule-of-thumb baseline, in days of cover

Data quality check: 1 - cost_ratio must match the gross profit margin Walmart
reported in its annual reports. If it doesn't, the pull picked up the wrong
numbers, and nothing downstream should use them.

Usage (run from the repo root):
    export SEC_USER_AGENT="Your Name your.email@example.com"   # SEC requires this
    python walmart_10k_calibration.py              # uses the cached JSON if present
    python walmart_10k_calibration.py --refresh    # force a fresh download
    python walmart_10k_calibration.py --list-tags  # see which XBRL tags Walmart used
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

CIK = "0000104169"  # Walmart Inc.
URL = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK}.json"
CACHE = Path("data/raw/walmart_companyfacts.json")
OUTPUT = Path("data/processed/walmart_calibration.csv")

# Walmart's fiscal year ends January 31 and is named for that calendar year
# (FY2016 = Feb 1, 2015 to Jan 31, 2016). FY2011 through FY2017 covers all of
# M5 (Jan 29, 2011 to Jun 19, 2016).
FISCAL_YEARS = range(2011, 2018)

# Candidate XBRL tags in priority order. Net sales must exclude membership and
# other income, because Walmart computes its gross profit rate on net sales.
# Run --list-tags to confirm these against the actual data.
NET_SALES_TAGS = [
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
]
COST_TAGS = ["CostOfGoodsSold", "CostOfGoodsAndServicesSold", "CostOfRevenue"]
INVENTORY_TAGS = ["InventoryNet"]

# Gross profit margin (%) from the five-year financial summaries in Walmart's
# fiscal 2016 and fiscal 2017 annual reports (Exhibit 13 of each 10-K).
REPORTED_GM_PCT = {2012: 24.5, 2013: 24.3, 2014: 24.3, 2015: 24.3, 2016: 24.6, 2017: 24.9}
TOLERANCE_PP = 0.15  # reported margins are rounded to one decimal place
DIO_SANITY_RANGE = (20, 90)  # days; outside this, the inventory pull is suspect


def load_companyfacts(refresh: bool) -> dict:
    """Load Walmart's companyfacts JSON, downloading only when needed."""
    if CACHE.exists() and not refresh:
        return json.loads(CACHE.read_text())
    user_agent = os.environ.get("SEC_USER_AGENT")
    if not user_agent:
        sys.exit('Set SEC_USER_AGENT first, e.g. export SEC_USER_AGENT="Your Name you@example.com"')
    resp = requests.get(URL, headers={"User-Agent": user_agent}, timeout=60)
    resp.raise_for_status()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(resp.text)
    return resp.json()


def _entries(facts: dict, tag: str) -> list[dict]:
    return facts.get("facts", {}).get("us-gaap", {}).get(tag, {}).get("units", {}).get("USD", [])


def annual_values(facts: dict, tag: str) -> dict[int, dict]:
    """Full-year values from 10-K filings, keyed by fiscal year.

    Each 10-K reports three years of results, so the same year appears in
    several filings. Keep the most recently filed value, since it reflects
    any restatement.
    """
    out: dict[int, dict] = {}
    for e in _entries(facts, tag):
        if not e.get("form", "").startswith("10-K") or "start" not in e:
            continue
        start, end = date.fromisoformat(e["start"]), date.fromisoformat(e["end"])
        if not 350 <= (end - start).days <= 380:  # drop quarterly durations
            continue
        fy = end.year
        if fy not in out or e["filed"] > out[fy]["filed"]:
            out[fy] = {"start": start, "end": end, "val": e["val"], "filed": e["filed"], "tag": tag}
    return out


def instant_values(facts: dict, tag: str) -> dict[date, dict]:
    """Point-in-time balances (like inventory) from 10-K filings, keyed by date."""
    out: dict[date, dict] = {}
    for e in _entries(facts, tag):
        if not e.get("form", "").startswith("10-K") or "start" in e:
            continue
        end = date.fromisoformat(e["end"])
        if end not in out or e["filed"] > out[end]["filed"]:
            out[end] = {"val": e["val"], "filed": e["filed"], "tag": tag}
    return out


def pick(series_fn, facts: dict, tags: list[str]) -> dict:
    """Merge candidate tags: for each period, the first tag in the list wins."""
    merged: dict = {}
    for tag in tags:
        for key, rec in series_fn(facts, tag).items():
            merged.setdefault(key, rec)
    return merged


def build_table(facts: dict) -> pd.DataFrame:
    sales = pick(annual_values, facts, NET_SALES_TAGS)
    cost = pick(annual_values, facts, COST_TAGS)
    inventory = pick(instant_values, facts, INVENTORY_TAGS)

    rows = []
    for fy in FISCAL_YEARS:
        s, c = sales.get(fy), cost.get(fy)
        if s is None or c is None:
            print(f"FY{fy}: missing net sales or cost of sales, skipped", file=sys.stderr)
            continue
        if (s["start"], s["end"]) != (c["start"], c["end"]):
            print(f"FY{fy}: sales and cost periods don't match, skipped", file=sys.stderr)
            continue

        inv_end = inventory.get(c["end"])
        inv_start = inventory.get(c["start"] - timedelta(days=1))  # prior year-end
        avg_inv = (inv_start["val"] + inv_end["val"]) / 2 if inv_start and inv_end else math.nan

        cost_ratio = c["val"] / s["val"]
        gm_pct = 100 * (1 - cost_ratio)
        dio = avg_inv / c["val"] * 365

        reported = REPORTED_GM_PCT.get(fy)
        if reported is None:
            check = "no reference"
        else:
            check = "pass" if abs(gm_pct - reported) <= TOLERANCE_PP else "FAIL"

        rows.append({
            "fiscal_year": fy,
            "period_start": c["start"],
            "period_end": c["end"],
            "net_sales_musd": s["val"] / 1e6,
            "cost_of_sales_musd": c["val"] / 1e6,
            "avg_inventory_musd": avg_inv / 1e6,
            "cost_ratio": cost_ratio,
            "gross_margin_pct": gm_pct,
            "reported_gm_pct": reported,
            "gm_check": check,
            "dio_days": dio,
            "weeks_of_cover": dio / 7,
            "net_sales_tag": s["tag"],
            "cost_tag": c["tag"],
        })
    return pd.DataFrame(rows)


def list_tags(facts: dict) -> None:
    """Show revenue, cost, and inventory tags that have data in the M5 years."""
    keywords = ("Revenue", "Sales", "CostOf", "Inventory")
    years = range(FISCAL_YEARS.start - 1, FISCAL_YEARS.stop)
    for tag, body in sorted(facts.get("facts", {}).get("us-gaap", {}).items()):
        if not any(k in tag for k in keywords):
            continue
        label = (body.get("label") or "")[:45]
        fys = sorted(fy for fy in annual_values(facts, tag) if fy in FISCAL_YEARS)
        if fys:
            print(f"{tag:<55} FY{fys[0]}-FY{fys[-1]} ({len(fys)} yrs)  {label}")
            continue
        dates = sorted(d for d in instant_values(facts, tag) if d.year in years)
        if dates:
            print(f"{tag:<55} balances {dates[0]} to {dates[-1]}  {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate cost ratio and DIO from Walmart's 10-K data.")
    parser.add_argument("--refresh", action="store_true", help="re-download from SEC even if cached")
    parser.add_argument("--list-tags", action="store_true", help="show candidate XBRL tags and exit")
    args = parser.parse_args()

    facts = load_companyfacts(args.refresh)
    if args.list_tags:
        list_tags(facts)
        return

    table = build_table(facts)
    if table.empty:
        sys.exit("No usable years found. Run with --list-tags to check the tag names.")

    shown = ["fiscal_year", "net_sales_musd", "cost_of_sales_musd", "gross_margin_pct",
             "reported_gm_pct", "gm_check", "dio_days", "weeks_of_cover"]
    print(table[shown].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    lo, hi = DIO_SANITY_RANGE
    for r in table.itertuples():
        if math.isnan(r.dio_days):
            print(f"Warning: FY{r.fiscal_year} is missing an inventory balance.", file=sys.stderr)
        elif not lo <= r.dio_days <= hi:
            print(f"Warning: FY{r.fiscal_year} DIO is {r.dio_days:.1f} days; check the inventory pull.",
                  file=sys.stderr)

    if (table["gm_check"] == "FAIL").any():
        sys.exit(
            "\nGross margin check FAILED, so nothing was saved. The pull doesn't match what "
            "Walmart reported.\nRun --list-tags. A margin about 0.5 points too high usually means "
            "total revenues\n(including membership income) was picked up instead of net sales."
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False)
    print(f"\nGross margin check passed for every year with a reported figure. Saved {OUTPUT}")


if __name__ == "__main__":
    main()
