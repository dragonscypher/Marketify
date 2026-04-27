from marketify.broker.mock_http import (MockBrokerHTTPClient,
                                        start_mock_broker_server)


def test_mock_broker_buy_fixture_account_identity():
    server = start_mock_broker_server()
    try:
        client = MockBrokerHTTPClient(server.base_url)
        assert client.get_account()["equity"] == 100000.0
        assert client.get_positions() == []
        assert client.get_orders() == []
        assert client.get_fills() == []

        client.submit_order({"symbol": "AAPL", "side": "buy", "qty": 10, "price": 100.0})
        account = client.get_account()
        positions = client.get_positions()
        assert account["cash"] == 99000.0
        assert account["equity"] == 100010.0
        assert positions[0]["symbol"] == "AAPL"
        assert positions[0]["qty"] == 10
        assert positions[0]["avg_entry"] == 100.0
        assert positions[0]["mark_price"] == 101.0
        assert positions[0]["unrealized_pnl"] == 10.0
    finally:
        server.shutdown()


def test_mock_broker_cancel_filled_order_is_blocked():
    server = start_mock_broker_server()
    try:
        client = MockBrokerHTTPClient(server.base_url)
        order = client.submit_order({"symbol": "MSFT", "side": "sell", "qty": 5, "price": 200.0})
        result = client.cancel_order(order["order_id"])
        assert result["status"] == "cannot_cancel_filled"
        assert client.get_positions()[0]["qty"] == -5
    finally:
        server.shutdown()
