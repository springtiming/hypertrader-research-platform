"""Explainable evidence checks that never authorize trading."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .metrics import PerformanceMetrics
from .statistics import CscvPboResult, DeflatedSharpeResult


@dataclass(frozen=True, slots=True)
class EvidenceGateConfig:
    minimum_observations: int
    minimum_dsr_probability: float
    maximum_pbo: float
    maximum_drawdown: float

    def __post_init__(self) -> None:
        if self.minimum_observations < 2:
            raise ValueError("minimum_observations must be at least two")
        probability_values = (
            self.minimum_dsr_probability,
            self.maximum_pbo,
            self.maximum_drawdown,
        )
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probability_values):
            raise ValueError("gate probabilities and drawdown must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class EvidenceCheck:
    name: str
    passed: bool
    observed: float
    threshold: float
    comparison: str


@dataclass(frozen=True, slots=True)
class EvidenceGateResult:
    eligible_for_further_research: bool
    checks: tuple[EvidenceCheck, ...]
    alpha_validated: bool = False
    authorizes_paper_trading: bool = False
    authorizes_live_trading: bool = False


def evaluate_evidence_gate(
    *,
    dsr: DeflatedSharpeResult,
    pbo: CscvPboResult,
    performance: PerformanceMetrics,
    config: EvidenceGateConfig,
) -> EvidenceGateResult:
    """Evaluate transparent screening checks for further research only."""

    checks = (
        EvidenceCheck(
            name="observation_count",
            passed=dsr.observation_count >= config.minimum_observations,
            observed=float(dsr.observation_count),
            threshold=float(config.minimum_observations),
            comparison="greater_than_or_equal",
        ),
        EvidenceCheck(
            name="deflated_sharpe_probability",
            passed=dsr.probability >= config.minimum_dsr_probability,
            observed=dsr.probability,
            threshold=config.minimum_dsr_probability,
            comparison="greater_than_or_equal",
        ),
        EvidenceCheck(
            name="probability_of_backtest_overfitting",
            passed=pbo.probability_of_backtest_overfitting <= config.maximum_pbo,
            observed=pbo.probability_of_backtest_overfitting,
            threshold=config.maximum_pbo,
            comparison="less_than_or_equal",
        ),
        EvidenceCheck(
            name="maximum_drawdown",
            passed=performance.max_drawdown <= config.maximum_drawdown,
            observed=performance.max_drawdown,
            threshold=config.maximum_drawdown,
            comparison="less_than_or_equal",
        ),
    )
    return EvidenceGateResult(
        eligible_for_further_research=all(check.passed for check in checks),
        checks=checks,
    )


__all__ = [
    "EvidenceCheck",
    "EvidenceGateConfig",
    "EvidenceGateResult",
    "evaluate_evidence_gate",
]
