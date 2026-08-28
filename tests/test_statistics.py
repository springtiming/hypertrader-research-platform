from __future__ import annotations

import math

import numpy as np
import pytest

from hypertrader_research.statistics import (
    _average_ranks,
    cscv_probability_of_backtest_overfitting,
    deflated_sharpe_ratio,
)


def _return_matrix(rows: int = 120, columns: int = 4) -> np.ndarray:
    time = np.arange(rows, dtype=np.float64)
    series = []
    for index in range(columns):
        series.append(
            0.0005 * (index + 1)
            + 0.008 * np.sin(time * (0.11 + index * 0.013) + index)
            + 0.003 * np.cos(time * (0.037 + index * 0.009))
        )
    return np.column_stack(series)


def test_deflated_sharpe_is_deterministic_and_bounded() -> None:
    matrix = _return_matrix()
    first = deflated_sharpe_ratio(matrix)
    second = deflated_sharpe_ratio(matrix)
    assert first == second
    assert 0.0 <= first.probability <= 1.0
    assert first.selected_index == int(np.argmax(first.periodic_sharpes))
    assert first.selected_annualized_sharpe == pytest.approx(
        first.selected_periodic_sharpe * math.sqrt(252.0)
    )


def test_deflated_sharpe_supports_preselected_and_effective_trials() -> None:
    result = deflated_sharpe_ratio(
        _return_matrix(),
        selected_index=0,
        effective_trial_count=1.0,
        null_sharpe_mean=0.1,
    )
    assert result.selected_index == 0
    assert result.effective_trial_count == 1.0
    assert result.expected_maximum_periodic_sharpe == 0.1


def test_deflated_sharpe_penalizes_more_trials() -> None:
    matrix = _return_matrix(columns=6)
    one_effective = deflated_sharpe_ratio(
        matrix,
        selected_index=5,
        effective_trial_count=1.0,
    )
    all_effective = deflated_sharpe_ratio(
        matrix,
        selected_index=5,
        effective_trial_count=6.0,
    )
    assert all_effective.expected_maximum_periodic_sharpe >= (
        one_effective.expected_maximum_periodic_sharpe
    )
    assert all_effective.probability <= one_effective.probability


@pytest.mark.parametrize(
    ("matrix", "kwargs"),
    [
        ([0.1, 0.2], {}),
        ([[0.1, 0.2]] * 4, {}),
        ([[0.1, 0.2]] * 5, {}),
        ([[0.1, float("nan")]] * 5, {}),
        ([[-1.0, 0.1]] * 5, {}),
        ([[True, False]] * 5, {}),
        (_return_matrix(), {"periods_per_year": 0.0}),
        (_return_matrix(), {"risk_free_rate_per_period": float("nan")}),
        (_return_matrix(), {"selected_index": 99}),
        (_return_matrix(), {"selected_index": True}),
        (_return_matrix(), {"effective_trial_count": 0.5}),
        (_return_matrix(), {"effective_trial_count": True}),
    ],
)
def test_deflated_sharpe_rejects_invalid_inputs(
    matrix: object,
    kwargs: dict[str, float | int],
) -> None:
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(matrix, **kwargs)


def test_average_ranks_handles_ties() -> None:
    ranks = _average_ranks(np.array([3.0, 1.0, 3.0, 2.0]))
    assert ranks == pytest.approx([3.5, 1.0, 3.5, 2.0])


def test_cscv_enumerates_every_legal_split() -> None:
    result = cscv_probability_of_backtest_overfitting(
        _return_matrix(rows=120, columns=5),
        partition_count=6,
    )
    assert result.split_count == math.comb(6, 3)
    assert sum(result.selection_counts) == result.split_count
    assert 0.0 <= result.probability_of_backtest_overfitting <= 1.0
    assert result.minimum_logit <= result.mean_logit <= result.maximum_logit
    assert all(
        len(split.in_sample_partitions) == 3
        and len(split.out_of_sample_partitions) == 3
        for split in result.splits
    )


def test_cscv_is_deterministic() -> None:
    matrix = _return_matrix(rows=96)
    first = cscv_probability_of_backtest_overfitting(matrix, partition_count=6)
    second = cscv_probability_of_backtest_overfitting(matrix, partition_count=6)
    assert first == second


@pytest.mark.parametrize(
    ("matrix", "partition_count", "max_splits"),
    [
        (_return_matrix(rows=96), 3, 20_000),
        (_return_matrix(rows=95), 6, 20_000),
        (_return_matrix(rows=96), 100, 20_000),
        (_return_matrix(rows=96), 16, 1),
        (_return_matrix(rows=96), 6.0, 20_000),
        (_return_matrix(rows=96), 6, 0),
    ],
)
def test_cscv_rejects_invalid_partition_requests(
    matrix: np.ndarray,
    partition_count: int,
    max_splits: int,
) -> None:
    with pytest.raises(ValueError):
        cscv_probability_of_backtest_overfitting(
            matrix,
            partition_count=partition_count,
            max_splits=max_splits,
        )


def test_cscv_rejects_all_tied_trials() -> None:
    base = 0.01 + 0.005 * np.sin(np.arange(96) * 0.2)
    matrix = np.column_stack([base, base])
    with pytest.raises(ValueError, match="non-tied"):
        cscv_probability_of_backtest_overfitting(matrix, partition_count=6)


def test_cscv_rejects_invalid_periods_per_year() -> None:
    with pytest.raises(ValueError, match="periods_per_year"):
        cscv_probability_of_backtest_overfitting(
            _return_matrix(rows=96),
            partition_count=6,
            periods_per_year=0.0,
        )
