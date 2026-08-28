from __future__ import annotations

import numpy as np
import pytest

from hypertrader_research.synthetic import (
    causal_momentum_positions,
    generate_synthetic_experiment,
    generate_synthetic_prices,
)


def test_synthetic_prices_are_deterministic_and_positive() -> None:
    first = generate_synthetic_prices(observations=128, seed=11)
    second = generate_synthetic_prices(observations=128, seed=11)
    different = generate_synthetic_prices(observations=128, seed=12)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)
    assert np.all(first > 0.0)
    assert first.size == 129


def test_toy_positions_are_causal() -> None:
    prices = np.linspace(100.0, 130.0, 40)
    changed_future = prices.copy()
    changed_future[-1] = 1_000.0
    original = causal_momentum_positions(prices, lookback=4)
    modified = causal_momentum_positions(changed_future, lookback=4)
    assert np.array_equal(original[:-1], modified[:-1])


def test_direction_reverses_toy_positions() -> None:
    prices = np.linspace(100.0, 120.0, 20)
    long = causal_momentum_positions(prices, lookback=2, direction=1)
    short = causal_momentum_positions(prices, lookback=2, direction=-1)
    assert long == pytest.approx(-short)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"observations": 10},
        {"observations": 128.0},
        {"seed": True},
        {"start_price": 0.0},
    ],
)
def test_synthetic_prices_reject_invalid_config(kwargs: object) -> None:
    with pytest.raises(ValueError):
        generate_synthetic_prices(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("lookback", "direction"),
    [(0, 1), (20, 1), (2, 0)],
)
def test_toy_positions_reject_invalid_config(
    lookback: int,
    direction: int,
) -> None:
    with pytest.raises(ValueError):
        causal_momentum_positions(
            np.linspace(100.0, 120.0, 20),
            lookback=lookback,
            direction=direction,
        )


def test_toy_positions_reject_nonpositive_prices() -> None:
    with pytest.raises(ValueError, match="positive"):
        causal_momentum_positions([100.0, 0.0, 101.0], lookback=1)


def test_experiment_contains_labeled_candidate_matrix() -> None:
    experiment = generate_synthetic_experiment(
        observations=128,
        seed=5,
        lookbacks=(2, 8),
    )
    assert experiment.candidate_positions.shape == (128, 4)
    assert experiment.candidate_labels == (
        "momentum_2",
        "momentum_8",
        "contrarian_2",
        "contrarian_8",
    )


@pytest.mark.parametrize("lookbacks", [(), (2, 2)])
def test_experiment_rejects_invalid_lookback_family(
    lookbacks: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError):
        generate_synthetic_experiment(observations=128, lookbacks=lookbacks)
