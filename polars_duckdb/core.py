"""Volatility analytics — rolling variance/std via DuckDB (GARCH fit stays in arch)."""

import duckdb
import numpy as np
import polars as pl


def simulate_returns(n: int = 1000, seed: int = 42) -> pl.Series:
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 0.1, 0.8, 0.1
    errors = rng.normal(size=n)
    vol = np.zeros(n)
    rets = np.zeros(n)
    for t in range(1, n):
        vol[t] = np.sqrt(omega + alpha * errors[t - 1] ** 2 + beta * vol[t - 1] ** 2)
        rets[t] = vol[t] * rng.normal()
    return pl.Series("returns", rets.tolist())


def rolling_variance(series: pl.Series, window: int = 20) -> pl.Series:
    w = window - 1
    work = series.to_frame("r").with_row_index("idx")
    return duckdb.sql(f"""
        SELECT
            VAR_SAMP(r) OVER (
                ORDER BY idx ROWS BETWEEN {w} PRECEDING AND CURRENT ROW
            ) AS rolling_var
        FROM work
        ORDER BY idx
    """).pl()["rolling_var"]


def rolling_std(series: pl.Series, window: int = 20) -> pl.Series:
    w = window - 1
    work = series.to_frame("r").with_row_index("idx")
    return duckdb.sql(f"""
        SELECT
            STDDEV_SAMP(r) OVER (
                ORDER BY idx ROWS BETWEEN {w} PRECEDING AND CURRENT ROW
            ) AS rolling_std
        FROM work
        ORDER BY idx
    """).pl()["rolling_std"]
