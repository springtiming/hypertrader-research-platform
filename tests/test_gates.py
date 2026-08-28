from __future__ import annotations

from dataclasses import replace

import pytest

from hypertrader_research.gates import EvidenceGateConfig, evaluate_evidence_gate
from hypertrader_research.metrics import PerformanceMetrics
from hypertrader_research.statistics import CscvPboResult, DeflatedSharpeResult


def _dsr() -> DeflatedSharpeResult:
    return DeflatedSharpeResult(
        selected_index=0,
        observation_count=500,
        trial_count=3,
        effective_trial_count=3.0,
        periodic_sharpes=(0.1, 0.05, -0.02),
        selected_periodic_sharpe=0.1,
        selected_annualized_sharpe=1.58,
        expected_maximum_periodic_sharpe=0.04,
        skewness=0.0,
        pearson_kurtosis=3.0,
        test_statistic=2.0,
        probability=0.98,
    )


def _pbo() -> CscvPboResult:
    return CscvPboResult(
        observation_count=500,
        trial_count=3,
        partition_count=6,
        split_count=20,
        overfit_split_count=4,
        probability_of_backtest_overfitting=0.2,
        mean_logit=0.5,
        minimum_logit=-0.5,
        maximum_logit=1.0,
        selection_counts=(10, 6, 4),
        splits=(),
    )


def _metrics() -> PerformanceMetrics:
    return PerformanceMetrics(
        total_return=0.1,
        annualized_return=0.05,
        annualized_volatility=0.1,
        sharpe_ratio=0.8,
        max_drawdown=0.1,
    )


def _config() -> EvidenceGateConfig:
    return EvidenceGateConfig(
        minimum_observations=252,
        minimum_dsr_probability=0.95,
        maximum_pbo=0.5,
        maximum_drawdown=0.2,
    )


def test_gate_can_only_allow_further_research() -> None:
    result = evaluate_evidence_gate(
        dsr=_dsr(),
        pbo=_pbo(),
        performance=_metrics(),
        config=_config(),
    )
    assert result.eligible_for_further_research
    assert all(check.passed for check in result.checks)
    assert not result.alpha_validated
    assert not result.authorizes_paper_trading
    assert not result.authorizes_live_trading


@pytest.mark.parametrize(
    ("dsr", "pbo", "metrics", "failed_name"),
    [
        (replace(_dsr(), observation_count=100), _pbo(), _metrics(), "observation_count"),
        (
            replace(_dsr(), probability=0.5),
            _pbo(),
            _metrics(),
            "deflated_sharpe_probability",
        ),
        (
            _dsr(),
            replace(_pbo(), probability_of_backtest_overfitting=0.8),
            _metrics(),
            "probability_of_backtest_overfitting",
        ),
        (
            _dsr(),
            _pbo(),
            replace(_metrics(), max_drawdown=0.5),
            "maximum_drawdown",
        ),
    ],
)
def test_gate_explains_each_failure(
    dsr: DeflatedSharpeResult,
    pbo: CscvPboResult,
    metrics: PerformanceMetrics,
    failed_name: str,
) -> None:
    result = evaluate_evidence_gate(
        dsr=dsr,
        pbo=pbo,
        performance=metrics,
        config=_config(),
    )
    assert not result.eligible_for_further_research
    assert failed_name in {check.name for check in result.checks if not check.passed}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum_observations": 1},
        {"minimum_dsr_probability": -0.1},
        {"maximum_pbo": 1.1},
        {"maximum_drawdown": float("nan")},
    ],
)
def test_gate_config_rejects_invalid_thresholds(kwargs: dict[str, float | int]) -> None:
    values: dict[str, float | int] = {
        "minimum_observations": 252,
        "minimum_dsr_probability": 0.95,
        "maximum_pbo": 0.5,
        "maximum_drawdown": 0.2,
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        EvidenceGateConfig(**values)  # type: ignore[arg-type]
