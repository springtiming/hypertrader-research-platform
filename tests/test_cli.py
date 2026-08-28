from __future__ import annotations

import json
from pathlib import Path

from hypertrader_research.cli import build_synthetic_report, main


def test_synthetic_report_is_explicit_about_scope() -> None:
    report = build_synthetic_report(
        observations=128,
        seed=3,
        partition_count=4,
    )
    assert report["schema_version"] == "hypertrader-synthetic-report-v1"
    assert report["scope"] == {
        "synthetic_data_only": True,
        "alpha_validated": False,
        "paper_trading_supported": False,
        "live_trading_supported": False,
    }
    assert report["experiment"]["trial_count"] == 10
    assert report["experiment"]["cost_bps_per_unit_turnover"] == 2.0
    assert report["experiment"]["full_entry_exit_cost_bps"] == 4.0
    assert report["cscv_pbo"]["split_count"] == 6
    assert report["evidence_gate"]["alpha_validated"] is False


def test_cli_prints_machine_readable_json(capsys: object) -> None:
    assert main(["--observations", "128", "--partitions", "4", "--compact"]) == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    report = json.loads(captured.out)
    assert report["scope"]["synthetic_data_only"] is True


def test_committed_example_matches_default_report() -> None:
    example_path = Path(__file__).resolve().parents[1] / "examples" / "synthetic_report.json"
    committed = json.loads(example_path.read_text(encoding="utf-8"))
    generated = json.loads(json.dumps(build_synthetic_report()))
    assert committed == generated
