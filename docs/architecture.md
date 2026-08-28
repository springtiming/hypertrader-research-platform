# Architecture and safety boundaries

## Objective

The project isolates four research concerns:

1. causal return simulation;
2. performance accounting;
3. multiple-testing diagnostics;
4. an explainable research-only decision boundary.

It intentionally omits every capability that could turn a research result into a
trading write.

## Components

### Synthetic inputs

src/hypertrader_research/synthetic.py creates positive prices from a documented
regime-plus-noise process with a fixed seed. The toy candidate family uses only
prices available at the start of each holding period.

The fixture is designed to test the pipeline. It is not calibrated to a real
asset, venue, or time period.

### Vector backtest

src/hypertrader_research/backtest.py consumes:

- one positive price vector of length T + 1;
- one target-position vector of length T;
- an explicit cost and exposure configuration.

Position i is chosen with information available at price i and is held from price
i to price i + 1. Turnover is charged when the target changes. When final close is
enabled, liquidation turnover is charged in the final period.

Inputs outside the exposure boundary are rejected rather than clipped. A
configuration that can lose all capital within one period is rejected.

### Performance accounting

src/hypertrader_research/metrics.py compounds simple net returns and calculates
sample-standard-deviation Sharpe, annualized volatility, annualized compound
return, and maximum drawdown over the complete equity path.

### Statistical diagnostics

src/hypertrader_research/statistics.py accepts a synchronized T by N matrix of
periodic simple net returns. It contains no candidate names, strategy parameters,
data loaders, persistence, evidence registry, or promotion state.

### Evidence gate

src/hypertrader_research/gates.py compares observations against caller-visible
thresholds. It can produce only a further-research screening result. It cannot
authorize paper trading, live trading, or an alpha claim.

## Dependency direction

    synthetic ----> backtest ----> metrics
                                \
    synchronized returns --------> statistics
                                     |
    metrics -------------------------+
                                     |
                                     v
                                   gates

There is no adapter layer for HTTP, databases, cloud storage, exchanges, wallets,
environment files, or secret stores.

## Public-release constraints

Every committed example must be synthetic. The repository boundary check rejects
selected high-confidence credential patterns, private-key markers, environment
files, databases, archives, images, and binaries. It is a narrow repository
guardrail, not a complete secret or personal-data scanner. The check reports only
violation categories and repository-relative paths, never matched content.

## Non-goals

- signal discovery;
- portfolio optimization;
- production backtesting;
- paper or live execution;
- exchange integration;
- performance marketing;
- investment advice.
