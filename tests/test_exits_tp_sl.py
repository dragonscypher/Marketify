from marketify.backtest.simulator import evaluate_exit_price


def test_take_profit_long():
    price, reason = evaluate_exit_price(
        side="buy",
        stop_loss=99.0,
        take_profit=101.0,
        row_open=100.0,
        row_high=102.0,
        row_low=99.5,
        bars_held=1,
        time_stop_bars=48,
    )
    assert price == 101.0
    assert reason == "take_profit"


def test_stop_loss_short():
    price, reason = evaluate_exit_price(
        side="sell",
        stop_loss=101.0,
        take_profit=99.0,
        row_open=100.0,
        row_high=101.5,
        row_low=99.5,
        bars_held=1,
        time_stop_bars=48,
    )
    assert price == 101.0
    assert reason == "stop_loss"


def test_time_stop_uses_broker_bars_count():
    price, reason = evaluate_exit_price(
        side="buy",
        stop_loss=99.0,
        take_profit=101.0,
        row_open=100.0,
        row_high=100.4,
        row_low=99.6,
        bars_held=48,
        time_stop_bars=48,
    )
    assert price == 100.0
    assert reason == "time_stop"
