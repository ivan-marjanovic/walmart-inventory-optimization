# Forecast accuracy (holdout year)

- Walk-forward: retrained every 13 weeks on data observed by each forecast origin
- Tree count chosen by early stopping on pre-backtest data: 179
- 3-week totals match the base-case protection interval (2-week lead time + 1-week review)
- WMAPE = sum of absolute errors / sum of actual demand; bias above 0 means over-forecasting

## Overall

| | LightGBM | 4-wk avg | 13-wk avg | Croston (SBA) |
|---|---|---|---|---|
| **WMAPE, 1 week ahead** | 22.8% | 25.7% | 30.4% | 32.8% |
| **Bias, 1 week ahead** | -2.0% | -0.8% | -1.9% | -5.6% |
| **WMAPE, 3-week total** | 19.6% | 22.5% | 26.5% | 28.4% |
| **Bias, 3-week total** | -2.4% | -1.0% | -2.1% | -5.7% |

## WMAPE of 3-week totals by ABC class

| | LightGBM | 4-wk avg | 13-wk avg | Croston (SBA) |
|---|---|---|---|---|
| **A** | 18.1% | 21.1% | 25.3% | 27.1% |
| **B** | 22.4% | 24.7% | 28.5% | 30.7% |
| **C** | 28.6% | 31.5% | 33.9% | 36.0% |

## WMAPE of 3-week totals by demand pattern

| | LightGBM | 4-wk avg | 13-wk avg | Croston (SBA) |
|---|---|---|---|---|
| **smooth** | 18.3% | 20.9% | 23.6% | 24.2% |
| **erratic** | 28.9% | 34.5% | 45.1% | 43.5% |
| **intermittent** | 28.2% | 31.9% | 45.7% | 61.0% |
| **lumpy** | 34.2% | 41.1% | 58.8% | 69.4% |

## Top LightGBM features (share of gain)

- mean_4: 55.4%
- mean_8: 15.0%
- lag_0: 12.4%
- mean_13: 7.5%
- mean_52: 3.9%
- price_ratio_13: 1.0%
- lag_52_target: 0.9%
- mean_26: 0.8%
- price_target: 0.5%
- week_of_year_target: 0.4%
