"""A deterministic vector backtest with an explicit cost model."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .metrics import (
    FloatArray,
    PerformanceMetrics,
    as_float_vector,
    compound_equity,
    summarize_performance,
)


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Configuration for one research-only vector backtest."""

    initial_equity: float = 100_000.0
    periods_per_year: float = 252.0
    fee_bps: float = 1.0
    slippage_bps: float = 1.0
    max_abs_position: float = 1.0
    close_final_position: bool = True

    def __post_init__(self) -> None:
        numeric_values = (
            self.initial_equity,
            self.periods_per_year,
            self.fee_bps,
            self.slippage_bps,
            self.max_abs_position,
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("backtest configuration values must be finite")
        if self.initial_equity <= 0.0 or self.periods_per_year <= 0.0:
            raise ValueError("initial_equity and periods_per_year must be positive")
        if self.fee_bps < 0.0 or self.slippage_bps < 0.0:
            raise ValueError("cost assumptions cannot be negative")
        if self.max_abs_position <= 0.0:
            raise ValueError("max_abs_position must be positive")


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Immutable arrays and a compact performance summary."""

    asset_returns: FloatArray
    positions: FloatArray
    turnover: FloatArray
    costs: FloatArray
    strategy_returns: FloatArray
    equity_curve: FloatArray
    metrics: PerformanceMetrics


def _readonly(values: FloatArray) -> FloatArray:
    values.setflags(write=False)
    return values


def run_backtest(
    prices: ArrayLike,
    positions: ArrayLike,
    *,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run a causal backtest over precomputed target positions.

    A position at index i is held from price i to price i + 1. Therefore the
    position vector must have exactly one fewer element than the price vector.
    """

    settings = config or BacktestConfig()
    price_vector = as_float_vector(prices, name="prices", minimum_size=3)
    position_vector = as_float_vector(positions, name="positions")
    if np.any(price_vector <= 0.0):
        raise ValueError("prices must be positive")
    if position_vector.size != price_vector.size - 1:
        raise ValueError("positions must have exactly len(prices) - 1 values")
    if np.any(np.abs(position_vector) > settings.max_abs_position + 1e-12):
        raise ValueError("positions exceed max_abs_position")

    asset_returns = price_vector[1:] / price_vector[:-1] - 1.0
    if not np.all(np.isfinite(asset_returns)):
        raise ValueError("derived asset returns must remain finite")
    previous_positions = np.concatenate(
        (np.array([0.0], dtype=np.float64), position_vector[:-1])
    )
    turnover = np.abs(position_vector - previous_positions)
    if settings.close_final_position:
        turnover[-1] += abs(position_vector[-1])
    cost_rate = (settings.fee_bps + settings.slippage_bps) / 10_000.0
    costs = turnover * cost_rate
    strategy_returns = position_vector * asset_returns - costs
    if np.any(strategy_returns <= -1.0):
        raise ValueError("configuration can lose all capital within one period")
    equity_curve = compound_equity(
        strategy_returns,
        initial_equity=settings.initial_equity,
    )
    metrics = summarize_performance(
        strategy_returns,
        periods_per_year=settings.periods_per_year,
        initial_equity=settings.initial_equity,
    )
    return BacktestResult(
        asset_returns=_readonly(asset_returns.copy()),
        positions=_readonly(position_vector.copy()),
        turnover=_readonly(turnover),
        costs=_readonly(costs),
        strategy_returns=_readonly(strategy_returns),
        equity_curve=_readonly(equity_curve),
        metrics=metrics,
    )


__all__ = ["BacktestConfig", "BacktestResult", "run_backtest"]
