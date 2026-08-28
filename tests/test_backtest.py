from __future__ import annotations

import numpy as np
import pytest

from hypertrader_research.backtest import BacktestConfig, run_backtest


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial_equity": 0.0},
        {"periods_per_year": 0.0},
        {"fee_bps": -1.0},
        {"slippage_bps": -1.0},
        {"max_abs_position": 0.0},
        {"fee_bps": float("nan")},
    ],
)
def test_backtest_config_fails_closed(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        BacktestConfig(**kwargs)


def test_long_and_short_are_mirrors_without_costs() -> None:
    prices = [100.0, 110.0, 99.0]
    config = BacktestConfig(fee_bps=0.0, slippage_bps=0.0, close_final_position=False)
    long_result = run_backtest(prices, [1.0, 1.0], config=config)
    short_result = run_backtest(prices, [-1.0, -1.0], config=config)
    assert long_result.strategy_returns == pytest.approx(-short_result.strategy_returns)


def test_costs_include_entry_rebalance_and_final_close() -> None:
    config = BacktestConfig(
        fee_bps=5.0,
        slippage_bps=5.0,
        close_final_position=True,
    )
    result = run_backtest([100.0, 100.0, 100.0], [0.5, -0.5], config=config)
    assert result.turnover == pytest.approx([0.5, 1.5])
    assert result.costs == pytest.approx([0.0005, 0.0015])
    assert result.strategy_returns == pytest.approx([-0.0005, -0.0015])


def test_position_at_index_is_applied_to_the_next_price_change() -> None:
    result = run_backtest(
        [100.0, 110.0, 99.0],
        [1.0, 0.0],
        config=BacktestConfig(fee_bps=0.0, slippage_bps=0.0, close_final_position=False),
    )
    assert result.strategy_returns == pytest.approx([0.10, 0.0])


def test_result_arrays_are_read_only() -> None:
    result = run_backtest([100.0, 101.0, 102.0], [1.0, 1.0])
    with pytest.raises(ValueError):
        result.strategy_returns[0] = 0.0


@pytest.mark.parametrize(
    ("prices", "positions", "message"),
    [
        ([100.0, 101.0], [1.0], "at least 3"),
        ([100.0, 0.0, 101.0], [1.0, 1.0], "positive"),
        ([100.0, 101.0, 102.0], [1.0], "exactly"),
        ([100.0, 101.0, 102.0], [2.0, 0.0], "exceed"),
    ],
)
def test_backtest_rejects_invalid_market_inputs(
    prices: list[float],
    positions: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        run_backtest(prices, positions)


def test_backtest_rejects_period_loss_beyond_capital() -> None:
    with pytest.raises(ValueError, match="lose all capital"):
        run_backtest(
            [1.0, 3.0, 3.0],
            [-1.0, -1.0],
            config=BacktestConfig(
                fee_bps=0.0,
                slippage_bps=0.0,
                close_final_position=False,
            ),
        )


def test_backtest_rejects_derived_return_overflow() -> None:
    with np.errstate(over="ignore"):
        with pytest.raises(ValueError, match="asset returns"):
            run_backtest([5e-324, 1e308, 1e308], [0.0, 0.0])


def test_costs_reduce_terminal_equity() -> None:
    prices = np.linspace(100.0, 120.0, 20)
    positions = np.ones(19)
    free = run_backtest(
        prices,
        positions,
        config=BacktestConfig(fee_bps=0.0, slippage_bps=0.0),
    )
    costly = run_backtest(
        prices,
        positions,
        config=BacktestConfig(fee_bps=3.0, slippage_bps=2.0),
    )
    assert costly.equity_curve[-1] < free.equity_curve[-1]
