from __future__ import annotations

import math

import numpy as np
import pytest

from hypertrader_research.metrics import (
    annualized_sharpe,
    as_float_vector,
    compound_equity,
    maximum_drawdown,
    summarize_performance,
)


def test_as_float_vector_copies_and_validates() -> None:
    source = np.array([1.0, 2.0])
    result = as_float_vector(source, name="sample", minimum_size=2)
    source[0] = 99.0
    assert result.tolist() == [1.0, 2.0]


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([[1.0]], "one-dimensional"),
        ([1.0], "at least 2"),
        ([1.0, float("nan")], "finite"),
        ([True, False], "boolean"),
    ],
)
def test_as_float_vector_rejects_invalid_values(
    values: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        as_float_vector(values, name="sample", minimum_size=2)


def test_compound_equity_matches_hand_calculation() -> None:
    result = compound_equity([0.10, -0.05], initial_equity=100.0)
    assert result == pytest.approx([100.0, 110.0, 104.5])


@pytest.mark.parametrize(
    ("returns", "initial_equity", "message"),
    [
        ([0.1], 0.0, "positive"),
        ([-1.0], 100.0, "greater than -1"),
    ],
)
def test_compound_equity_rejects_invalid_inputs(
    returns: list[float],
    initial_equity: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compound_equity(returns, initial_equity=initial_equity)


def test_compound_equity_rejects_numeric_overflow() -> None:
    with np.errstate(over="ignore"):
        with pytest.raises(ValueError, match="finite"):
            compound_equity([1e308, 1e308])


def test_annualized_sharpe_matches_independent_formula() -> None:
    returns = np.array([0.01, 0.02, -0.005, 0.015])
    expected = float(np.mean(returns) / np.std(returns, ddof=1) * math.sqrt(252.0))
    assert annualized_sharpe(returns, periods_per_year=252.0) == pytest.approx(expected)


def test_annualized_sharpe_handles_flat_returns() -> None:
    assert annualized_sharpe([0.01, 0.01], periods_per_year=252.0) == 0.0


@pytest.mark.parametrize(
    ("periods", "risk_free"),
    [(0.0, 0.0), (252.0, float("inf"))],
)
def test_annualized_sharpe_rejects_invalid_assumptions(
    periods: float,
    risk_free: float,
) -> None:
    with pytest.raises(ValueError):
        annualized_sharpe(
            [0.01, -0.01],
            periods_per_year=periods,
            risk_free_rate_per_period=risk_free,
        )


def test_maximum_drawdown_uses_the_full_path() -> None:
    assert maximum_drawdown([100.0, 120.0, 90.0, 110.0, 80.0]) == pytest.approx(1.0 / 3.0)


@pytest.mark.parametrize("equity", [[100.0], [100.0, 0.0]])
def test_maximum_drawdown_rejects_invalid_equity(equity: list[float]) -> None:
    with pytest.raises(ValueError):
        maximum_drawdown(equity)


def test_summarize_performance_is_internally_consistent() -> None:
    result = summarize_performance(
        [0.10, -0.05, 0.02],
        periods_per_year=3.0,
        initial_equity=100.0,
    )
    assert result.total_return == pytest.approx(1.10 * 0.95 * 1.02 - 1.0)
    assert result.annualized_return == pytest.approx(result.total_return)
    assert result.annualized_volatility > 0.0
    assert result.max_drawdown == pytest.approx(0.05)


def test_summarize_performance_rejects_annualization_overflow() -> None:
    with pytest.raises(ValueError, match="finite"):
        summarize_performance([1.0, 1.0], periods_per_year=1e308)
