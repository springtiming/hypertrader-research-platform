"""Deterministic synthetic inputs for examples and tests."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .metrics import FloatArray, as_float_vector

FloatMatrix = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SyntheticExperiment:
    prices: FloatArray
    candidate_positions: FloatMatrix
    candidate_labels: tuple[str, ...]
    seed: int


def generate_synthetic_prices(
    *,
    observations: int = 512,
    seed: int = 7,
    start_price: float = 100.0,
) -> FloatArray:
    """Generate positive prices from a transparent regime-plus-noise process."""

    if not isinstance(observations, int) or isinstance(observations, bool):
        raise ValueError("observations must be an integer")
    if observations < 64:
        raise ValueError("observations must be at least 64")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not math.isfinite(start_price) or start_price <= 0.0:
        raise ValueError("start_price must be positive and finite")
    generator = np.random.default_rng(seed)
    phase = np.linspace(0.0, 8.0 * math.pi, observations, endpoint=False)
    regime = 0.0008 * np.sin(phase) + 0.0004 * np.sin(phase / 3.0)
    noise = generator.normal(loc=0.0, scale=0.008, size=observations)
    log_returns = np.clip(0.0001 + regime + noise, -0.15, 0.15)
    prices = start_price * np.exp(np.concatenate(([0.0], np.cumsum(log_returns))))
    prices.setflags(write=False)
    return prices


def causal_momentum_positions(
    prices: ArrayLike,
    *,
    lookback: int,
    direction: int = 1,
) -> FloatArray:
    """Create toy positions using only prices available before each holding period."""

    price_vector = as_float_vector(prices, name="prices", minimum_size=3)
    if np.any(price_vector <= 0.0):
        raise ValueError("prices must be positive")
    if not isinstance(lookback, int) or isinstance(lookback, bool) or lookback < 1:
        raise ValueError("lookback must be a positive integer")
    if lookback >= price_vector.size - 1:
        raise ValueError("lookback must be shorter than the experiment")
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    positions = np.zeros(price_vector.size - 1, dtype=np.float64)
    for index in range(lookback, positions.size):
        signal = price_vector[index] / price_vector[index - lookback] - 1.0
        positions[index] = direction * float(np.sign(signal))
    positions.setflags(write=False)
    return positions


def generate_synthetic_experiment(
    *,
    observations: int = 512,
    seed: int = 7,
    lookbacks: tuple[int, ...] = (2, 4, 8, 16, 32),
) -> SyntheticExperiment:
    """Build a toy trial family suitable for multiple-testing demonstrations."""

    if not lookbacks or len(set(lookbacks)) != len(lookbacks):
        raise ValueError("lookbacks must be non-empty and unique")
    prices = generate_synthetic_prices(observations=observations, seed=seed)
    candidate_columns: list[FloatArray] = []
    candidate_labels: list[str] = []
    for direction, prefix in ((1, "momentum"), (-1, "contrarian")):
        for lookback in lookbacks:
            candidate_columns.append(
                causal_momentum_positions(
                    prices,
                    lookback=lookback,
                    direction=direction,
                )
            )
            candidate_labels.append(f"{prefix}_{lookback}")
    positions = np.column_stack(candidate_columns)
    positions.setflags(write=False)
    return SyntheticExperiment(
        prices=prices,
        candidate_positions=positions,
        candidate_labels=tuple(candidate_labels),
        seed=seed,
    )


__all__ = [
    "SyntheticExperiment",
    "causal_momentum_positions",
    "generate_synthetic_experiment",
    "generate_synthetic_prices",
]
