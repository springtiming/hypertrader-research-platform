"""Multiple-testing diagnostics derived from published quantitative research."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from statistics import NormalDist
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatMatrix = NDArray[np.float64]
EULER_MASCHERONI = 0.5772156649015329


@dataclass(frozen=True, slots=True)
class DeflatedSharpeResult:
    selected_index: int
    observation_count: int
    trial_count: int
    effective_trial_count: float
    periodic_sharpes: tuple[float, ...]
    selected_periodic_sharpe: float
    selected_annualized_sharpe: float
    expected_maximum_periodic_sharpe: float
    skewness: float
    pearson_kurtosis: float
    test_statistic: float
    probability: float


@dataclass(frozen=True, slots=True)
class CscvSplit:
    in_sample_partitions: tuple[int, ...]
    out_of_sample_partitions: tuple[int, ...]
    selected_index: int
    selected_out_of_sample_rank: float
    relative_rank: float
    logit: float
    overfit: bool


@dataclass(frozen=True, slots=True)
class CscvPboResult:
    observation_count: int
    trial_count: int
    partition_count: int
    split_count: int
    overfit_split_count: int
    probability_of_backtest_overfitting: float
    mean_logit: float
    minimum_logit: float
    maximum_logit: float
    selection_counts: tuple[int, ...]
    splits: tuple[CscvSplit, ...]


def _as_return_matrix(values: ArrayLike, *, minimum_rows: int = 4) -> FloatMatrix:
    raw = np.asarray(values)
    if raw.dtype.kind == "b":
        raise ValueError("returns_matrix cannot contain boolean values")
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("returns_matrix must be two-dimensional")
    rows, columns = matrix.shape
    if rows < minimum_rows or columns < 2:
        raise ValueError(
            f"returns_matrix must contain at least {minimum_rows} rows and two trials"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("returns_matrix must contain only finite values")
    if np.any(matrix <= -1.0):
        raise ValueError("simple returns must be greater than -1")
    return matrix.copy()


def _population_sharpes(
    matrix: FloatMatrix,
    *,
    risk_free_rate_per_period: float,
) -> FloatMatrix:
    excess = matrix - risk_free_rate_per_period
    standard_deviations = np.std(excess, axis=0, ddof=0)
    means = np.mean(excess, axis=0)
    if (
        not np.all(np.isfinite(standard_deviations))
        or not np.all(np.isfinite(means))
        or np.any(standard_deviations <= 1e-15)
    ):
        raise ValueError("each trial must have positive return variance")
    return cast(FloatMatrix, means / standard_deviations)


def _expected_maximum_sharpe(
    *,
    null_mean: float,
    standard_deviation: float,
    effective_trial_count: float,
) -> float:
    if effective_trial_count <= 1.0 or standard_deviation == 0.0:
        return null_mean
    normal = NormalDist()
    first = normal.inv_cdf(1.0 - 1.0 / effective_trial_count)
    second = normal.inv_cdf(1.0 - 1.0 / (effective_trial_count * math.e))
    maximum_z = (1.0 - EULER_MASCHERONI) * first + EULER_MASCHERONI * second
    return null_mean + standard_deviation * maximum_z


def deflated_sharpe_ratio(
    returns_matrix: ArrayLike,
    *,
    periods_per_year: float = 252.0,
    risk_free_rate_per_period: float = 0.0,
    selected_index: int | None = None,
    effective_trial_count: float | None = None,
    null_sharpe_mean: float = 0.0,
) -> DeflatedSharpeResult:
    """Estimate the probability that selected Sharpe exceeds a multiple-test null.

    The calculation uses population moments and the classic iid form. The caller
    may supply a defensible effective trial count; otherwise every observed trial
    is treated as independent.
    """

    matrix = _as_return_matrix(returns_matrix, minimum_rows=5)
    if not math.isfinite(periods_per_year) or periods_per_year <= 0.0:
        raise ValueError("periods_per_year must be positive and finite")
    if not math.isfinite(risk_free_rate_per_period) or not math.isfinite(null_sharpe_mean):
        raise ValueError("Sharpe assumptions must be finite")
    rows, columns = matrix.shape
    sharpes = _population_sharpes(
        matrix,
        risk_free_rate_per_period=risk_free_rate_per_period,
    )
    if isinstance(selected_index, bool):
        raise ValueError("selected_index must be an integer index")
    chosen = int(np.argmax(sharpes)) if selected_index is None else selected_index
    if chosen < 0 or chosen >= columns:
        raise ValueError("selected_index is outside the trial matrix")
    if isinstance(effective_trial_count, bool):
        raise ValueError("effective_trial_count must be numeric")
    effective = float(columns) if effective_trial_count is None else effective_trial_count
    if not math.isfinite(effective) or not 1.0 <= effective <= float(columns):
        raise ValueError("effective_trial_count must be within [1, trial_count]")

    selected_returns = matrix[:, chosen]
    selected_mean = float(np.mean(selected_returns))
    selected_standard_deviation = float(np.std(selected_returns, ddof=0))
    standardized = (selected_returns - selected_mean) / selected_standard_deviation
    skewness = float(np.mean(standardized**3))
    pearson_kurtosis = float(np.mean(standardized**4))
    selected_sharpe = float(sharpes[chosen])
    cross_trial_deviation = float(np.std(sharpes, ddof=0))
    expected_maximum = _expected_maximum_sharpe(
        null_mean=null_sharpe_mean,
        standard_deviation=cross_trial_deviation,
        effective_trial_count=effective,
    )
    denominator_variance = (
        1.0
        - skewness * selected_sharpe
        + (pearson_kurtosis - 1.0) * selected_sharpe**2 / 4.0
    )
    if not math.isfinite(denominator_variance) or denominator_variance <= 0.0:
        raise ValueError("probabilistic Sharpe variance must be positive")
    test_statistic = (
        (selected_sharpe - expected_maximum) * math.sqrt(rows - 1)
    ) / math.sqrt(denominator_variance)
    probability = NormalDist().cdf(test_statistic)
    return DeflatedSharpeResult(
        selected_index=chosen,
        observation_count=rows,
        trial_count=columns,
        effective_trial_count=effective,
        periodic_sharpes=tuple(float(value) for value in sharpes),
        selected_periodic_sharpe=selected_sharpe,
        selected_annualized_sharpe=selected_sharpe * math.sqrt(periods_per_year),
        expected_maximum_periodic_sharpe=expected_maximum,
        skewness=skewness,
        pearson_kurtosis=pearson_kurtosis,
        test_statistic=test_statistic,
        probability=probability,
    )


def _average_ranks(values: FloatMatrix) -> FloatMatrix:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < order.size:
        end = start + 1
        anchor = values[order[start]]
        while end < order.size and values[order[end]] == anchor:
            end += 1
        ranks[order[start:end]] = ((start + 1) + end) / 2.0
        start = end
    return ranks


def cscv_probability_of_backtest_overfitting(
    returns_matrix: ArrayLike,
    *,
    partition_count: int = 8,
    periods_per_year: float = 252.0,
    max_splits: int = 20_000,
) -> CscvPboResult:
    """Enumerate combinatorially symmetric cross-validation splits.

    Contiguous observations are divided into an even number of partitions. For
    every half-partition training set, the best in-sample trial is ranked on the
    complementary out-of-sample observations.
    """

    matrix = _as_return_matrix(returns_matrix)
    rows, columns = matrix.shape
    if not isinstance(partition_count, int) or isinstance(partition_count, bool):
        raise ValueError("partition_count must be an integer")
    if not isinstance(max_splits, int) or isinstance(max_splits, bool) or max_splits < 1:
        raise ValueError("max_splits must be a positive integer")
    if partition_count < 4 or partition_count % 2 != 0:
        raise ValueError("partition_count must be an even integer of at least four")
    if rows < partition_count or rows % partition_count != 0:
        raise ValueError("observation count must be divisible by partition_count")
    if not math.isfinite(periods_per_year) or periods_per_year <= 0.0:
        raise ValueError("periods_per_year must be positive and finite")
    expected_splits = math.comb(partition_count, partition_count // 2)
    if expected_splits > max_splits:
        raise ValueError("requested CSCV enumeration exceeds max_splits")

    rows_per_partition = rows // partition_count
    partitions = tuple(
        np.arange(index * rows_per_partition, (index + 1) * rows_per_partition)
        for index in range(partition_count)
    )
    score_cache: dict[tuple[int, ...], FloatMatrix] = {}

    def scores(partition_indices: tuple[int, ...]) -> FloatMatrix:
        cached = score_cache.get(partition_indices)
        if cached is not None:
            return cached
        selected_rows = np.concatenate([partitions[index] for index in partition_indices])
        result = _population_sharpes(
            matrix[selected_rows],
            risk_free_rate_per_period=0.0,
        ) * math.sqrt(periods_per_year)
        score_cache[partition_indices] = result
        return result

    split_results: list[CscvSplit] = []
    selection_counts = [0] * columns
    all_partitions = set(range(partition_count))
    for in_sample in combinations(range(partition_count), partition_count // 2):
        out_of_sample = tuple(sorted(all_partitions.difference(in_sample)))
        in_scores = scores(in_sample)
        out_scores = scores(out_of_sample)
        if np.all(in_scores == in_scores[0]) or np.all(out_scores == out_scores[0]):
            raise ValueError("CSCV selection and ranking require non-tied trial scores")
        maximum = float(np.max(in_scores))
        selected_candidates = np.flatnonzero(in_scores == maximum)
        selected = int(selected_candidates[0])
        ranks = _average_ranks(out_scores)
        selected_rank = float(ranks[selected])
        relative_rank = selected_rank / (columns + 1.0)
        logit = math.log(relative_rank) - math.log1p(-relative_rank)
        selection_counts[selected] += 1
        split_results.append(
            CscvSplit(
                in_sample_partitions=in_sample,
                out_of_sample_partitions=out_of_sample,
                selected_index=selected,
                selected_out_of_sample_rank=selected_rank,
                relative_rank=relative_rank,
                logit=logit,
                overfit=logit <= 0.0,
            )
        )

    logits = tuple(item.logit for item in split_results)
    overfit_count = sum(item.overfit for item in split_results)
    return CscvPboResult(
        observation_count=rows,
        trial_count=columns,
        partition_count=partition_count,
        split_count=len(split_results),
        overfit_split_count=overfit_count,
        probability_of_backtest_overfitting=overfit_count / len(split_results),
        mean_logit=math.fsum(logits) / len(logits),
        minimum_logit=min(logits),
        maximum_logit=max(logits),
        selection_counts=tuple(selection_counts),
        splits=tuple(split_results),
    )


__all__ = [
    "CscvPboResult",
    "CscvSplit",
    "DeflatedSharpeResult",
    "cscv_probability_of_backtest_overfitting",
    "deflated_sharpe_ratio",
]
