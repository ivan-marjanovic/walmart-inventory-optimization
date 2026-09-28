#!/usr/bin/env python3
"""
Simulate inventory policies over the holdout year and compare them at equal fill rate.

Every week, each item-state places an order that brings its inventory position
(on hand plus on order) up to a target level S. Orders arrive after the lead time,
demand that can't be met from stock is lost, and service is measured as fill rate.

Three policies, so the savings can be split into their two sources:
    baseline    4-week moving average + a uniform number of weeks of cover
    lgbm_cover  LightGBM forecast     + the same uniform weeks of cover
    lgbm_ss     LightGBM forecast     + safety stock sized from forecast errors and optimized

Error-based safety stock uses only the year before the holdout, so the test year
never leaks in. Each item's errors over the protection interval (lead time plus
one week of review) give its bias and spread. Standardized errors are pooled
within each demand pattern, so intermittent items get their own skewed error
distribution instead of a normal curve.

Safety stock is then allocated to minimize inventory dollars for a given overall
fill rate. Minimizing sum(unit cost * safety stock) + lambda * expected units short
gives each item the rule F(z_i) = 1 - c_i / lambda: its chance of not stocking out
falls as its unit cost rises. Cheap items get deep buffers and expensive items run
lean, because a unit of fill rate is cheapest to buy on cheap items. A service
floor keeps any item from being starved: every item gets at least an 80% chance
of not stocking out in a given week, whatever its cost.
    S = forecast over the interval + item bias + z_i * item spread

Each policy is swept across its setting (weeks of cover, or target fill rate)
to trace an inventory vs. fill-rate curve. Comparisons are made at matched fill rate.

Inputs:  data/processed/forecasts.parquet, m5_state_weekly.parquet, segmentation.csv,
         walmart_calibration.csv (for the sanity check)
Outputs: data/processed/policy_curves.csv   every simulated setting
         reports/policy_results.md          headline tables for the README
         reports/frontier.png               the inventory vs. fill-rate chart

Usage (from the repo root):
    python simulate.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FORECASTS = Path("data/processed/forecasts.parquet")
WEEKLY = Path("data/processed/m5_state_weekly.parquet")
SEGMENTS = Path("data/processed/segmentation.csv")
CALIBRATION = Path("data/processed/walmart_calibration.csv")
CURVES = Path("data/processed/policy_curves.csv")
REPORT = Path("reports/policy_results.md")
CHART = Path("reports/frontier.png")

HOLDOUT_WEEKS = 52   # must match segment.py and forecast.py
WARMUP_WEEKS = 8     # simulated before the holdout so stock levels settle; not scored
BASE_LEAD_TIME = 2
LEAD_TIMES = [1, 2, 3, 4]
BASE_HOLDING_RATE = 0.25
HOLDING_RATES = [0.15, 0.25, 0.35]
FILL_TARGETS = [0.95, 0.97, 0.98, 0.99]
HEADLINE_FILL = 0.98

COVER_GRID = np.round(np.arange(0, 8.01, 0.25), 2)  # extra weeks of cover
SERVICE_GRID = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.96, 0.97,
                0.98, 0.985, 0.99, 0.993, 0.995, 0.997, 0.999]  # service level of a median-cost item
SERVICE_FLOOR = 0.80  # minimum per-item chance of not stocking out; set to 0 to turn off
PATTERNS = ["smooth", "erratic", "intermittent", "lumpy"]
POLICIES = {
    "baseline": "4-wk avg + weeks of cover",
    "lgbm_cover": "LightGBM + weeks of cover",
    "lgbm_ss": "LightGBM + optimized safety stock",
}


class CheckFailed(Exception):
    pass


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'pass' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    if not ok:
        raise CheckFailed(name)


def load() -> tuple[pd.DataFrame, np.ndarray, dict]:
    series = (pd.read_csv(SEGMENTS).query("in_model")
              [["item_id", "state_id", "abc", "demand_pattern"]]
              .sort_values(["state_id", "item_id"]).reset_index(drop=True))
    series["s"] = np.arange(len(series))
    weeks = duckdb.sql(f"SELECT DISTINCT week_start FROM read_parquet('{WEEKLY}') ORDER BY 1") \
        .df()["week_start"].to_numpy()
    W, S = len(weeks), len(series)

    wk = duckdb.sql(f"""
        SELECT item_id, state_id, week_start, units, avg_unit_cost FROM read_parquet('{WEEKLY}')
    """).df().merge(series[["item_id", "state_id", "s"]], on=["item_id", "state_id"])
    w = np.searchsorted(weeks, wk["week_start"].to_numpy())
    demand = np.full((W, S), np.nan)
    cost = np.full((W, S), np.nan)
    demand[w, wk["s"]] = wk["units"]
    cost[w, wk["s"]] = wk["avg_unit_cost"]

    fc = duckdb.sql(f"""
        SELECT CAST(item_id AS VARCHAR) AS item_id, CAST(state_id AS VARCHAR) AS state_id,
               origin_start, h, lgbm, ma4
        FROM read_parquet('{FORECASTS}')
    """).df().merge(series[["item_id", "state_id", "s"]], on=["item_id", "state_id"])
    o = np.searchsorted(weeks, fc["origin_start"].to_numpy())
    lgbm = np.full((W, S, fc["h"].max() + 1), np.nan)  # [origin, series, horizon]
    lgbm[o, fc["s"], fc["h"]] = fc["lgbm"]
    ma4 = np.full((W, S), np.nan)
    first = fc["h"].to_numpy() == 1
    ma4[o[first], fc["s"].to_numpy()[first]] = fc["ma4"].to_numpy()[first]
    return series, weeks, {"demand": demand, "cost": cost, "lgbm": lgbm, "ma4": ma4}


def error_model(data: dict, series: pd.DataFrame, L: int, hold0: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Per-item bias and spread of interval errors, plus pooled standardized errors by pattern.

    Uses only forecast origins whose whole protection interval ends before the holdout.
    """
    H = L + 1
    origins = np.arange(0, hold0 - H)  # last target week = origin + H <= hold0 - 1
    fcst = data["lgbm"][origins][:, :, 1:H + 1].sum(axis=2)  # NaN where any horizon is missing
    actual = sum(data["demand"][origins + h] for h in range(1, H + 1))
    err = actual - fcst
    used = ~np.isnan(err).all(axis=1)
    check(f"L={L}: safety stock uses only errors from before the holdout",
          int((origins[used] + H).max()) < hold0,
          f"{int(used.sum())} forecast origins")
    bias = np.nanmean(err, axis=0)
    spread = np.nanstd(err, axis=0)
    z = (err - bias) / np.where(spread > 0, spread, np.nan)
    pools = {}
    for pattern in PATTERNS:
        cols = series.index[series["demand_pattern"] == pattern].to_numpy()
        pool = z[:, cols].ravel()
        pool = pool[~np.isnan(pool)]
        if len(pool):
            pools[pattern] = pool
    return bias, spread, pools


def safety_factors(service: float, unit_cost: np.ndarray, series: pd.DataFrame, pools: dict) -> np.ndarray:
    """Optimal z per item: F(z_i) = 1 - c_i / lambda, with lambda set so a median-cost item gets `service`."""
    lam = np.nanmedian(unit_cost) / (1 - service)
    floor = min(SERVICE_FLOOR, service)  # the floor never exceeds the median item's own target
    prob = np.clip(1 - unit_cost / lam, max(floor, 0.01), 0.9995)
    z = np.zeros(len(series))
    for pattern, pool in pools.items():
        idx = (series["demand_pattern"] == pattern).to_numpy()
        z[idx] = np.quantile(pool, prob[idx])
    return z


def simulate(target: np.ndarray, demand: np.ndarray, cost: np.ndarray, L: int) -> dict:
    """Weekly order-up-to with lost sales. Returns per-item totals over the scored weeks."""
    T, n = demand.shape
    on_hand = np.maximum(target[0], 0.0)  # start at the target level with nothing on order
    start_stock = on_hand.copy()
    arrivals = np.zeros((T + L + 1, n))
    received = np.zeros(n)
    sold_all = np.zeros(n)
    out = {k: np.zeros(n) for k in ["demand", "sales", "inv_dollar_weeks", "cogs", "inv_units_weeks"]}
    for w in range(T):
        on_hand += arrivals[w]
        received += arrivals[w]
        position = on_hand + arrivals[w + 1:w + L].sum(axis=0)
        order = np.ceil(np.maximum(target[w] - position, 0.0) - 1e-9).clip(min=0)
        arrivals[w + L] += order
        begin = on_hand.copy()
        sold = np.minimum(on_hand, demand[w])
        on_hand -= sold
        sold_all += sold
        if w >= WARMUP_WEEKS:
            avg_units = (begin + on_hand) / 2
            out["demand"] += demand[w]
            out["sales"] += sold
            out["inv_units_weeks"] += avg_units
            out["inv_dollar_weeks"] += avg_units * cost[w]
            out["cogs"] += sold * cost[w]
    out["balance_ok"] = bool(np.allclose(start_stock + received - sold_all, on_hand))
    return out


def summarize(res: dict, idx: np.ndarray | slice = slice(None)) -> dict:
    d, s = res["demand"][idx].sum(), res["sales"][idx].sum()
    inv = res["inv_dollar_weeks"][idx].sum() / HOLDOUT_WEEKS
    cogs = res["cogs"][idx].sum() / HOLDOUT_WEEKS
    return {"fill_rate": s / d if d else np.nan, "avg_inventory": inv,
            "weeks_of_supply": inv / cogs if cogs else np.nan}


def run_lead_time(data: dict, series: pd.DataFrame, L: int, hold0: int, start: int) -> dict:
    """Sweep every policy at one lead time. Returns {policy: [(setting, per-item results), ...]}."""
    H = L + 1
    dec = np.arange(start, data["demand"].shape[0])  # decision weeks; each uses the prior week's forecast
    fcst = np.nansum(data["lgbm"][dec - 1][:, :, 1:H + 1], axis=2)
    ma4 = np.nan_to_num(data["ma4"][dec - 1])
    demand, cost = data["demand"][dec], data["cost"][dec]
    bias, spread, pools = error_model(data, series, L, hold0)
    unit_cost = np.nanmean(data["cost"][hold0 - HOLDOUT_WEEKS:hold0], axis=0)  # year before the holdout

    runs = {name: [] for name in POLICIES}
    for k in COVER_GRID:
        runs["baseline"].append((k, simulate((H + k) * ma4, demand, cost, L)))
        runs["lgbm_cover"].append((k, simulate(fcst * (1 + k / H), demand, cost, L)))
    for p in SERVICE_GRID:
        z = safety_factors(p, unit_cost, series, pools)
        runs["lgbm_ss"].append((p, simulate(fcst + bias + z * spread, demand, cost, L)))

    balanced = all(r["balance_ok"] for sweep in runs.values() for _, r in sweep)
    check(f"L={L}: stock flows balance in every simulation", balanced)
    for name, sweep in runs.items():
        fills = np.array([summarize(r)["fill_rate"] for _, r in sweep])
        dips = float(np.max(np.maximum.accumulate(fills) - fills))
        check(f"L={L}: {name} fill rate rises with its setting", dips < 0.002,
              f"{fills.min():.1%} to {fills.max():.1%}")
    return runs


def at_fill(sweep: list, target: float, groups: dict | None = None) -> dict | None:
    """Interpolate a policy's results at a target overall fill rate, or None if out of range."""
    fills = [summarize(r)["fill_rate"] for _, r in sweep]
    for i in range(len(sweep) - 1):
        lo, hi = fills[i], fills[i + 1]
        if lo <= target <= hi and hi > lo:
            wt = (target - lo) / (hi - lo)
            a, b = sweep[i][1], sweep[i + 1][1]
            mixed = {k: (1 - wt) * a[k] + wt * b[k] for k in ["demand", "sales", "inv_dollar_weeks", "cogs"]}
            result = {"setting": (1 - wt) * sweep[i][0] + wt * sweep[i + 1][0], **summarize(mixed)}
            has_demand = mixed["demand"] > 0
            item_fill = mixed["sales"][has_demand] / mixed["demand"][has_demand]
            result["items_below_90"] = float((item_fill < 0.90).mean())
            if groups:
                result["groups"] = {g: summarize(mixed, idx) for g, idx in groups.items()}
            return result
    return None


def pct(v: float) -> str:
    return "n/a" if v is None or np.isnan(v) else f"{v:.1%}"


def money(v: float) -> str:
    if v is None or np.isnan(v):
        return "n/a"
    return f"${v / 1e6:,.2f}M" if abs(v) >= 1e6 else f"${v / 1e3:,.0f}K"


def md(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def report(all_runs: dict, series: pd.DataFrame, weeks: np.ndarray, hold0: int) -> None:
    base = all_runs[BASE_LEAD_TIME]
    groups = {f"ABC {c}": series.index[series["abc"] == c].to_numpy() for c in "ABC"}
    groups |= {p: series.index[series["demand_pattern"] == p].to_numpy()
               for p in PATTERNS if (series["demand_pattern"] == p).any()}

    # 1. Headline: inventory at matched fill rate, and where the savings come from.
    headline = []
    for f in FILL_TARGETS:
        pts = {name: at_fill(base[name], f) for name in POLICIES}
        inv = {name: (p["avg_inventory"] if p else np.nan) for name, p in pts.items()}
        b = inv["baseline"]
        headline.append([f"{f:.0%}", money(inv["baseline"]), money(inv["lgbm_cover"]), money(inv["lgbm_ss"]),
                         pct((b - inv["lgbm_ss"]) / b), pct((b - inv["lgbm_cover"]) / b),
                         pct((inv["lgbm_cover"] - inv["lgbm_ss"]) / b)])
    head_cols = ["Fill rate", "Baseline", "LightGBM + cover", "LightGBM + safety stock",
                 "Total savings", "From forecast", "From safety stock"]

    # 2. Segments at the headline fill rate.
    b_pt = at_fill(base["baseline"], HEADLINE_FILL, groups)
    p_pt = at_fill(base["lgbm_ss"], HEADLINE_FILL, groups)
    check(f"all three policies reach the {HEADLINE_FILL:.0%} headline fill rate",
          b_pt is not None and p_pt is not None and at_fill(base["lgbm_cover"], HEADLINE_FILL) is not None,
          "extend COVER_GRID or SERVICE_GRID if not")
    seg_rows = []
    for g in groups:
        bg, pg = b_pt["groups"][g], p_pt["groups"][g]
        seg_rows.append([g, money(bg["avg_inventory"]), pct(bg["fill_rate"]),
                         money(pg["avg_inventory"]), pct(pg["fill_rate"]),
                         pct((bg["avg_inventory"] - pg["avg_inventory"]) / bg["avg_inventory"])])
    seg_cols = ["Segment", "Baseline inventory", "Baseline fill", "Policy inventory", "Policy fill",
                "Inventory change"]

    # 3. Annual holding cost savings at the headline fill rate.
    saved = b_pt["avg_inventory"] - p_pt["avg_inventory"]
    hold_rows = [[f"{r:.0%}", money(b_pt["avg_inventory"] * r), money(p_pt["avg_inventory"] * r), money(saved * r)]
                 for r in HOLDING_RATES]

    # 4. Lead-time stress test.
    lt_rows = []
    for L, runs in all_runs.items():
        b, c, p = (at_fill(runs[n], HEADLINE_FILL) for n in POLICIES)
        if b and p:
            lt_rows.append([f"{L} week{'s' if L > 1 else ''}", money(b["avg_inventory"]),
                            money(p["avg_inventory"]),
                            pct((b["avg_inventory"] - p["avg_inventory"]) / b["avg_inventory"]),
                            f"{b['weeks_of_supply']:.1f} vs {p['weeks_of_supply']:.1f}"])
        else:
            lt_rows.append([f"{L} weeks", "n/a", "n/a", "n/a", "n/a"])

    # 5. Sanity check against Walmart's reported inventory level.
    walmart = None
    if CALIBRATION.exists():
        cal = pd.read_csv(CALIBRATION)
        mid = pd.Timestamp(weeks[hold0 + HOLDOUT_WEEKS // 2])
        fy = mid.year + 1 if mid.month >= 2 else mid.year
        if "weeks_of_cover" in cal and (cal["fiscal_year"] == fy).any():
            walmart = (fy, float(cal.loc[cal["fiscal_year"] == fy, "weeks_of_cover"].iloc[0]))

    # Chart: the frontier at the base lead time.
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"baseline": "#8c8c8c", "lgbm_cover": "#5b8def", "lgbm_ss": "#1b9e5a"}
    for name, sweep in base.items():
        pts = [summarize(r) for _, r in sweep]
        ax.plot([q["avg_inventory"] / 1e6 for q in pts], [q["fill_rate"] * 100 for q in pts],
                marker="o", ms=3, lw=2, color=colors[name], label=POLICIES[name])
    ax.axhline(HEADLINE_FILL * 100, ls="--", lw=1, color="#444")
    ax.annotate(f"{(b_pt['avg_inventory'] - p_pt['avg_inventory']) / b_pt['avg_inventory']:.0%} less inventory "
                f"at {HEADLINE_FILL:.0%} fill rate",
                xy=(p_pt["avg_inventory"] / 1e6, HEADLINE_FILL * 100),
                xytext=(10, -28), textcoords="offset points", fontsize=9)
    b99 = at_fill(base["baseline"], 0.99)
    if b99:
        ax.set_xlim(0, b99["avg_inventory"] / 1e6 * 1.3)  # focus on the region planners care about
    ax.set_xlabel("Average inventory at cost ($M)")
    ax.set_ylabel("Fill rate (%)")
    ax.set_ylim(max(80, ax.get_ylim()[0]), 100)
    ax.set_title(f"Inventory vs. fill rate, holdout year ({BASE_LEAD_TIME}-week lead time)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    CHART.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(CHART, dpi=150)

    # Terminal and markdown output.
    print(f"\nInventory at matched fill rate ({BASE_LEAD_TIME}-week lead time):")
    print(pd.DataFrame(headline, columns=head_cols).to_string(index=False))
    print(f"\nBy segment at {HEADLINE_FILL:.0%} overall fill rate:")
    print(pd.DataFrame(seg_rows, columns=seg_cols).to_string(index=False))
    print(f"\nAnnual holding cost at {HEADLINE_FILL:.0%} fill rate:")
    print(pd.DataFrame(hold_rows, columns=["Holding rate", "Baseline", "Policy", "Savings"]).to_string(index=False))
    print(f"\nLead-time stress test at {HEADLINE_FILL:.0%} fill rate:")
    print(pd.DataFrame(lt_rows, columns=["Lead time", "Baseline", "Policy", "Savings",
                                         "Weeks of supply"]).to_string(index=False))
    tail = (f"Item-states below 90% fill at {HEADLINE_FILL:.0%} overall: baseline {b_pt['items_below_90']:.1%}, "
            f"policy {p_pt['items_below_90']:.1%}")
    print("\n" + tail)
    sanity = (f"Simulated weeks of supply at {HEADLINE_FILL:.0%} fill: baseline {b_pt['weeks_of_supply']:.1f}, "
              f"policy {p_pt['weeks_of_supply']:.1f}")
    if walmart:
        sanity += (f". Walmart's FY{walmart[0]} 10-K implies {walmart[1]:.1f} weeks across all inventory, "
                   f"including shelf, backroom, and in-transit stock the simulation doesn't model.")
    print("\n" + sanity)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join([
        "# Inventory policy results (holdout year)",
        "",
        f"- Weekly order-up-to policy, {BASE_LEAD_TIME}-week base lead time, lost sales, "
        f"{len(series):,} item-states",
        "- Policies compared at matched fill rate; savings split into better forecasting "
        "and better safety stock",
        "- Safety stock sized only from forecast errors in the year before the holdout, then allocated "
        "to minimize inventory dollars at each fill rate, with an "
        f"{SERVICE_FLOOR:.0%} minimum per-item service floor",
        "",
        "![Inventory vs. fill rate](frontier.png)",
        "",
        "## Average inventory at matched fill rate", "", md(headline, head_cols), "",
        f"## By segment at {HEADLINE_FILL:.0%} overall fill rate", "", md(seg_rows, seg_cols), "",
        f"## Annual holding cost at {HEADLINE_FILL:.0%} fill rate", "",
        md(hold_rows, ["Holding rate", "Baseline", "Policy", "Savings"]), "",
        f"## Lead-time stress test at {HEADLINE_FILL:.0%} fill rate", "",
        md(lt_rows, ["Lead time", "Baseline", "Policy", "Savings", "Weeks of supply (baseline vs policy)"]), "",
        "## Long-tail service", "", tail, "",
        "## Sanity check", "", sanity, "",
    ]))


def main() -> None:
    missing = [p for p in (FORECASTS, WEEKLY, SEGMENTS) if not p.exists()]
    if missing:
        sys.exit("Missing inputs (run forecast.py first):\n  " + "\n  ".join(map(str, missing)))
    t0 = time.time()
    print("Loading forecasts, demand, and costs...")
    series, weeks, data = load()
    W = len(weeks)
    hold0 = W - HOLDOUT_WEEKS
    start = hold0 - WARMUP_WEEKS
    print(f"  {len(series):,} item-states, holdout {pd.Timestamp(weeks[hold0]).date()} "
          f"to {pd.Timestamp(weeks[-1]).date()}, {WARMUP_WEEKS}-week warm-up")

    try:
        all_runs = {}
        for L in LEAD_TIMES:
            print(f"Simulating {L}-week lead time...")
            all_runs[L] = run_lead_time(data, series, L, hold0, start)
        rows = [{"lead_time": L, "policy": name, "setting": setting, **summarize(r)}
                for L, runs in all_runs.items() for name, sweep in runs.items() for setting, r in sweep]
        report(all_runs, series, weeks, hold0)
    except CheckFailed as e:
        sys.exit(f"\nCheck failed ({e}), so nothing was saved.")

    pd.DataFrame(rows).to_csv(CURVES, index=False)
    print(f"\nSaved {CURVES}, {REPORT}, and {CHART}. Done in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main()
