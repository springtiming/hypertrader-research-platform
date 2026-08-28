"""Small, explicit performance metrics for periodic simple returns."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """A compact performance summary with values expressed as decimal fractions."""

    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float


def as_float_vector(
    values: ArrayLike,
    *,
    name: str,
    minimum_size: int = 1,
) -> FloatArray:
    """Validate and copy a one-dimensional finite numeric vector."""

    raw = np.asarray(values)
    if raw.dtype.kind == "b":
        raise ValueError(f"{name} cannot contain boolean values")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size < minimum_size:
        raise ValueError(f"{name} must contain at least {minimum_size} values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def compound_equity(
    returns: ArrayLike,
    *,
    initial_equity: float = 1.0,
) -> FloatArray:
    """Build an equity curve including the initial point."""

    periodic = as_float_vector(returns, name="returns")
    if not math.isfinite(initial_equity) or initial_equity <= 0.0:
        raise ValueError("initial_equity must be positive and finite")
    if np.any(periodic <= -1.0):
        raise ValueError("simple returns must be greater than -1")
    equity = np.empty(periodic.size + 1, dtype=np.float64)
    equity[0] = initial_equity
    equity[1:] = initial_equity * np.cumprod(1.0 + periodic)
    if not np.all(np.isfinite(equity)):
        raise ValueError("equity curve must remain finite")
    return equity


def annualized_sharpe(
    returns: ArrayLike,
    *,
    periods_per_year: float,
    risk_free_rate_per_period: float = 0.0,
) -> float:
    """Calculate sample-standard-deviation annualized Sharpe."""

    periodic = as_float_vector(returns, name="returns", minimum_size=2)
    if not math.isfinite(periods_per_year) or periods_per_year <= 0.0:
        raise ValueError("periods_per_year must be positive and finite")
    if not math.isfinite(risk_free_rate_per_period):
        raise ValueError("risk_free_rate_per_period must be finite")
    excess = periodic - risk_free_rate_per_period
    standard_deviation = float(np.std(excess, ddof=1))
    if math.isclose(standard_deviation, 0.0, rel_tol=0.0, abs_tol=1e-15):
        return 0.0
    return float(np.mean(excess) / standard_deviation * math.sqrt(periods_per_year))


def maximum_drawdown(equity: ArrayLike) -> float:
    """Return the worst peak-to-trough drawdown as a positive decimal fraction."""

    curve = as_float_vector(equity, name="equity", minimum_size=2)
    if np.any(curve <= 0.0):
        raise ValueError("equity must remain positive")
    running_peak = np.maximum.accumulate(curve)
    drawdowns = 1.0 - curve / running_peak
    return float(np.max(drawdowns))


def summarize_performance(
    returns: ArrayLike,
    *,
    periods_per_year: float,
    initial_equity: float = 1.0,
) -> PerformanceMetrics:
    """Summarize a periodic simple-return stream."""

    periodic = as_float_vector(returns, name="returns", minimum_size=2)
    curve = compound_equity(periodic, initial_equity=initial_equity)
    growth = float(curve[-1] / curve[0])
    total_return = growth - 1.0
    try:
        annualized_return = growth ** (periods_per_year / periodic.size) - 1.0
    except OverflowError as exc:
        raise ValueError("annualized performance metrics must remain finite") from exc
    annualized_volatility = float(np.std(periodic, ddof=1) * math.sqrt(periods_per_year))
    if not math.isfinite(annualized_return) or not math.isfinite(annualized_volatility):
        raise ValueError("annualized performance metrics must remain finite")
    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        sharpe_ratio=annualized_sharpe(
            periodic,
            periods_per_year=periods_per_year,
        ),
        max_drawdown=maximum_drawdown(curve),
    )


__all__ = [
    "FloatArray",
    "PerformanceMetrics",
    "annualized_sharpe",
    "as_float_vector",
    "compound_equity",
    "maximum_drawdown",
    "summarize_performance",
]
