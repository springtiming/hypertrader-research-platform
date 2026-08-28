# HyperTrader Research Platform

[![CI](https://github.com/springtiming/hypertrader-research-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/springtiming/hypertrader-research-platform/actions/workflows/ci.yml)

An evidence-gated quantitative research toolkit for reproducible backtesting and
multiple-testing diagnostics.

The repository is deliberately narrow:

- synthetic data only in committed examples;
- no exchange, wallet, account, database, deployment, or order-execution code;
- no proprietary strategy parameters or historical research artifacts;
- no validated alpha and no profitability claim.

The default demonstration actually rejects its selected synthetic candidate at
the Deflated Sharpe gate. That negative result is intentional: the project
demonstrates how to reject weak evidence, not how to manufacture a return claim.

## What this demonstrates

- A causal vector backtest with explicit turnover, fees, slippage, and forced
  final close semantics.
- Full-path maximum drawdown and clearly defined annualized performance metrics.
- Deflated Sharpe Ratio for selection bias, multiple testing, and non-normal
  returns.
- Combinatorially Symmetric Cross-Validation and Probability of Backtest
  Overfitting with complete split enumeration.
- Explainable, fail-closed evidence checks that can permit only further research.
- Deterministic synthetic fixtures, 87 tests, strict typing, linting, and more
  than 98% branch-aware test coverage on Python 3.11.

## Quick start

    python3.11 -m venv .venv
    .venv/bin/python -m pip install --require-hashes -r requirements-dev.lock
    .venv/bin/python -m pip install -e . --no-deps
    .venv/bin/python -m hypertrader_research

The command performs no network or filesystem reads. With the default seed it
produces a machine-readable report whose key result is:

    synthetic_data_only: true
    dsr_probability: 0.575227
    cscv_pbo: 0.257143
    eligible_for_further_research: false
    alpha_validated: false
    live_trading_supported: false

These values describe a toy experiment generated in memory. They are not market
performance and must not be interpreted as investment evidence.

## Architecture

    deterministic synthetic prices
                  |
                  v
    causal toy candidate positions
                  |
                  v
    cost-aware vector backtests ------> performance metrics
                  |
                  v
         synchronized T x N returns
                  |
          +-------+-------+
          |               |
          v               v
    Deflated Sharpe    CSCV / PBO
          |               |
          +-------+-------+
                  |
                  v
      explainable research-only gate
                  |
                  v
       further research or rejection

The core accepts in-memory arrays and has no persistence or network adapter. See
[architecture](docs/architecture.md) for timing and safety boundaries.

## Methodology

The statistical implementations follow the published methods described by:

- David H. Bailey and Marcos López de Prado,
  [The Deflated Sharpe Ratio](https://doi.org/10.2139/ssrn.2460551).
- David H. Bailey, Jonathan M. Borwein, Marcos López de Prado, and Qiji Jim Zhu,
  [The Probability of Backtest Overfitting](https://escholarship.org/uc/item/4w1110bb).

The exact estimator choices and limitations are documented in
[methodology](docs/methodology.md). This implementation has not been independently
certified against every reference implementation.

## Safety boundary

The evidence gate returns eligible_for_further_research only. Its result types
hard-code these claims as false:

- alpha_validated
- authorizes_paper_trading
- authorizes_live_trading

Passing an illustrative threshold is not sufficient evidence of an investable
strategy. Real research additionally needs licensed point-in-time data, temporal
leakage controls, independent reproduction, out-of-domain testing, execution
model validation, and human review.

## Public provenance

This is a new, no-history, clean-room portfolio edition distilled from design
lessons in a larger private research system maintained by springtiming. No source
file, Git history, issue, pull request, workflow, credential, market dataset,
strategy configuration, research report, deployment asset, or visual asset was
copied from that private repository.

See [provenance](PROVENANCE.md) and
[public snapshot manifest](PUBLIC_SNAPSHOT_MANIFEST.json).

## Verification

    .venv/bin/python -m ruff check .
    .venv/bin/python -m mypy src
    .venv/bin/python -m pytest --cov
    .venv/bin/python scripts/check_public_boundary.py

CI repeats these checks on Python 3.11, 3.12, and 3.13 without secrets.

## Limitations

- The vector backtest consumes precomputed positions; it is not an event-driven
  exchange simulator.
- Prices are close-to-close synthetic observations. There is no order book,
  latency, market impact, funding, borrowing, or gap execution model.
- The classic Deflated Sharpe calculation assumes iid periodic returns and does
  not adjust for serial correlation.
- CSCV ranks trial Sharpe across contiguous partitions; the method diagnoses
  selection instability but cannot prove generalization.
- Default thresholds are illustrative and are not a production promotion policy.

## License

MIT. See [LICENSE](LICENSE). Third-party packages are installed dependencies and
are not vendored; see [third-party notices](THIRD_PARTY_NOTICES.md).
