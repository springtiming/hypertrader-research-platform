"""Public research-only API for HyperTrader Research Platform."""

from .backtest import BacktestConfig, BacktestResult, run_backtest
from .gates import (
    EvidenceCheck,
    EvidenceGateConfig,
    EvidenceGateResult,
    evaluate_evidence_gate,
)
from .metrics import (
    PerformanceMetrics,
    annualized_sharpe,
    compound_equity,
    maximum_drawdown,
    summarize_performance,
)
from .statistics import (
    CscvPboResult,
    DeflatedSharpeResult,
    cscv_probability_of_backtest_overfitting,
    deflated_sharpe_ratio,
)
from .synthetic import (
    SyntheticExperiment,
    causal_momentum_positions,
    generate_synthetic_experiment,
    generate_synthetic_prices,
)

__version__ = "0.1.0"

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "CscvPboResult",
    "DeflatedSharpeResult",
    "EvidenceCheck",
    "EvidenceGateConfig",
    "EvidenceGateResult",
    "PerformanceMetrics",
    "SyntheticExperiment",
    "annualized_sharpe",
    "causal_momentum_positions",
    "compound_equity",
    "cscv_probability_of_backtest_overfitting",
    "deflated_sharpe_ratio",
    "evaluate_evidence_gate",
    "generate_synthetic_experiment",
    "generate_synthetic_prices",
    "maximum_drawdown",
    "run_backtest",
    "summarize_performance",
]
