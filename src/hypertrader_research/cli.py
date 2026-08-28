"""Command-line demonstration over deterministic synthetic data."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

import numpy as np

from .backtest import BacktestConfig, BacktestResult, run_backtest
from .gates import EvidenceGateConfig, evaluate_evidence_gate
from .statistics import (
    cscv_probability_of_backtest_overfitting,
    deflated_sharpe_ratio,
)
from .synthetic import generate_synthetic_experiment


def build_synthetic_report(
    *,
    observations: int = 512,
    seed: int = 7,
    partition_count: int = 8,
) -> dict[str, Any]:
    """Run the complete demonstration without network or filesystem access."""

    experiment = generate_synthetic_experiment(
        observations=observations,
        seed=seed,
    )
    config = BacktestConfig(
        initial_equity=100_000.0,
        periods_per_year=252.0,
        fee_bps=1.0,
        slippage_bps=1.0,
        max_abs_position=1.0,
        close_final_position=True,
    )
    backtests: list[BacktestResult] = [
        run_backtest(
            experiment.prices,
            experiment.candidate_positions[:, index],
            config=config,
        )
        for index in range(experiment.candidate_positions.shape[1])
    ]
    returns_matrix = np.column_stack([result.strategy_returns for result in backtests])
    dsr = deflated_sharpe_ratio(
        returns_matrix,
        periods_per_year=config.periods_per_year,
    )
    pbo = cscv_probability_of_backtest_overfitting(
        returns_matrix,
        partition_count=partition_count,
        periods_per_year=config.periods_per_year,
    )
    selected_backtest = backtests[dsr.selected_index]
    gate_config = EvidenceGateConfig(
        minimum_observations=252,
        minimum_dsr_probability=0.95,
        maximum_pbo=0.50,
        maximum_drawdown=0.30,
    )
    gate = evaluate_evidence_gate(
        dsr=dsr,
        pbo=pbo,
        performance=selected_backtest.metrics,
        config=gate_config,
    )
    return {
        "schema_version": "hypertrader-synthetic-report-v1",
        "scope": {
            "synthetic_data_only": True,
            "alpha_validated": False,
            "paper_trading_supported": False,
            "live_trading_supported": False,
        },
        "experiment": {
            "seed": seed,
            "observation_count": observations,
            "trial_count": len(experiment.candidate_labels),
            "candidate_labels": list(experiment.candidate_labels),
            "selected_candidate": experiment.candidate_labels[dsr.selected_index],
            "cost_bps_per_unit_turnover": config.fee_bps + config.slippage_bps,
            "full_entry_exit_cost_bps": 2.0 * (config.fee_bps + config.slippage_bps),
        },
        "selected_performance": asdict(selected_backtest.metrics),
        "deflated_sharpe": {
            "selected_index": dsr.selected_index,
            "selected_annualized_sharpe": dsr.selected_annualized_sharpe,
            "effective_trial_count": dsr.effective_trial_count,
            "expected_maximum_periodic_sharpe": dsr.expected_maximum_periodic_sharpe,
            "probability": dsr.probability,
        },
        "cscv_pbo": {
            "partition_count": pbo.partition_count,
            "split_count": pbo.split_count,
            "overfit_split_count": pbo.overfit_split_count,
            "probability": pbo.probability_of_backtest_overfitting,
            "mean_logit": pbo.mean_logit,
        },
        "evidence_gate": asdict(gate),
        "interpretation": (
            "Passing this illustrative gate only permits further research. "
            "It does not validate alpha or authorize paper or live trading."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an evidence-gated demonstration over synthetic market data."
    )
    parser.add_argument("--observations", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--partitions", type=int, default=8)
    parser.add_argument("--compact", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = build_synthetic_report(
        observations=arguments.observations,
        seed=arguments.seed,
        partition_count=arguments.partitions,
    )
    print(
        json.dumps(
            report,
            indent=None if arguments.compact else 2,
            sort_keys=True,
        )
    )
    return 0


__all__ = ["build_synthetic_report", "main"]
