"""
Alpaca クライアントを偽物に差し替えて、執行エンジンの安全ガードと発注フローを確かめる。
実口座・実APIには一切接続しない（.env も読み込ませない）。
"""
import json
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from alpaca.trading.enums import OrderSide, OrderStatus, QueryOrderStatus

import main
from src.execution import alpaca_broker as broker
from src.execution.alpaca_broker import ExecutionError, load_target_weights

PRICES = {"SPY": 500.0, "QQQ": 400.0, "IWM": 200.0, "TLT": 100.0, "GLD": 250.0, "BIL": 91.5}
TARGET = {"SPY": 0.6, "QQQ": 0.2, "IWM": 0.1, "TLT": 0.0, "GLD": 0.0, "BIL": 0.1}
# ガード2〜4に掛からない緩い上限（個別ガードのテストでは該当する上限だけを戻す）
LOOSE = {"max_notional": 1e9, "max_turnover": 10.0, "max_orders": 50}


class FakeTradingClient:
    def __init__(self, api_key, secret_key, paper=False):
        self.paper = paper
        self.equity = 100_000.0
        self.positions: dict[str, float] = {}
        self.open_orders: list = []
        self.submitted: list = []
        self.market_open = True
        self.fail_on_submit_no: int | None = None

    def get_account(self):
        return SimpleNamespace(equity=str(self.equity), status="ACTIVE")

    def get_all_positions(self):
        return [SimpleNamespace(symbol=s, qty=str(q)) for s, q in self.positions.items()]

    def get_orders(self, filter=None):
        return list(self.open_orders) if filter is not None and filter.status == QueryOrderStatus.OPEN else []

    def get_clock(self):
        return SimpleNamespace(is_open=self.market_open, next_open=datetime(2026, 10, 13, 13, 30, tzinfo=timezone.utc))

    def submit_order(self, order_data):
        n = len(self.submitted) + 1
        if self.fail_on_submit_no == n:
            raise RuntimeError("simulated API failure")
        self.submitted.append(order_data)
        return SimpleNamespace(id=f"order-{n}", status=OrderStatus.ACCEPTED)


class FakeDataClient:
    def __init__(self, api_key=None, secret_key=None):
        traded_at = datetime.now(timezone.utc) - timedelta(hours=1)
        self.trades = {s: SimpleNamespace(price=p, timestamp=traded_at) for s, p in PRICES.items()}

    def get_stock_latest_trade(self, request):
        syms = request.symbol_or_symbols
        syms = [syms] if isinstance(syms, str) else syms
        return {s: self.trades[s] for s in syms if s in self.trades}


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setattr(broker, "ENV_FILE", tmp_path / "no.env")
    monkeypatch.setattr(broker, "TradingClient", FakeTradingClient)
    monkeypatch.setattr(broker, "StockHistoricalDataClient", FakeDataClient)
    monkeypatch.setenv("APCA_API_KEY_ID", "test-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "test-secret")
    return broker.AlpacaExecutionEngine(log_dir=tmp_path / "logs")


@pytest.fixture
def weights_file(tmp_path):
    return write_weights(tmp_path / "target_weights.json", TARGET)


def write_weights(path, weights: dict, as_of: str | None = None):
    payload = {"strategy": "test", "as_of": as_of or date.today().isoformat(), "params": {}, "weights": weights}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def last_log(engine) -> dict:
    return json.loads(engine.trade_log_file.read_text(encoding="utf-8").splitlines()[-1])


def today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# --- ガード1: ドライラン既定・paper 固定 -------------------------------------------------

def test_client_is_always_paper(engine):
    assert engine.trading_client.paper is True


def test_dry_run_returns_plan(engine, weights_file):
    plan = engine.rebalance_from_json(weights_file, **LOOSE)
    assert [(o["symbol"], o["side"], o["qty"]) for o in plan] == [
        ("BIL", "buy", 109), ("IWM", "buy", 50), ("QQQ", "buy", 50), ("SPY", "buy", 120),
    ]


def test_dry_run_submits_nothing(engine, weights_file):
    engine.rebalance_from_json(weights_file, **LOOSE)
    assert engine.trading_client.submitted == []
    assert not engine.last_run_file.exists()
    assert last_log(engine)["mode"] == "dry-run"


# --- 発注数量の計算と管理対象 ---------------------------------------------------------------

def test_target_qty_is_floor_of_equity_times_weight(engine, weights_file):
    bil = next(o for o in engine.rebalance_from_json(weights_file, **LOOSE) if o["symbol"] == "BIL")
    assert bil["target_qty"] == 109  # 100,000 * 0.1 / 91.5 = 109.29


def test_zero_weight_symbol_is_sold(engine, weights_file):
    engine.trading_client.positions = {"TLT": 30.0}
    tlt = next(o for o in engine.rebalance_from_json(weights_file, **LOOSE) if o["symbol"] == "TLT")
    assert (tlt["side"], tlt["qty"], tlt["target_qty"]) == ("sell", 30, 0)


def test_unmanaged_holding_is_untouched(engine, weights_file, capsys):
    engine.trading_client.positions = {"AAPL": 10.0}
    engine.rebalance_from_json(weights_file, submit=True, **LOOSE)
    assert "AAPL" not in {o.symbol for o in engine.trading_client.submitted}
    assert "管理対象外の保有銘柄（変更しません）: AAPL" in capsys.readouterr().out


def test_sells_are_sent_before_buys(engine, weights_file):
    engine.trading_client.positions = {"SPY": 200.0, "TLT": 30.0, "GLD": 10.0}
    engine.rebalance_from_json(weights_file, submit=True, **LOOSE)
    sent = [(o.side, o.symbol) for o in engine.trading_client.submitted]
    assert sent == [
        (OrderSide.SELL, "GLD"), (OrderSide.SELL, "SPY"), (OrderSide.SELL, "TLT"),
        (OrderSide.BUY, "BIL"), (OrderSide.BUY, "IWM"), (OrderSide.BUY, "QQQ"),
    ]


# --- 送信・ロック・ログ ---------------------------------------------------------------------

def test_submit_writes_lock(engine, weights_file):
    engine.rebalance_from_json(weights_file, submit=True, **LOOSE)
    lock = json.loads(engine.last_run_file.read_text(encoding="utf-8"))
    assert lock["date_utc"] == today_utc() and lock["n_orders"] == 4


def test_submit_is_logged(engine, weights_file):
    engine.rebalance_from_json(weights_file, submit=True, **LOOSE)
    rec = last_log(engine)
    assert rec["mode"] == "submit" and rec["complete"] is True
    assert [o["order_id"] for o in rec["submitted"]] == ["order-1", "order-2", "order-3", "order-4"]


# --- ガード2〜5 ----------------------------------------------------------------------------

def test_guard2_blocks_large_single_order(engine, weights_file):
    with pytest.raises(ExecutionError, match=r"\[Guard 2\]") as exc:
        engine.rebalance_from_json(weights_file, **{**LOOSE, "max_notional": 2000.0})
    assert "[Guard 3]" not in str(exc.value)


def test_guard3_blocks_high_turnover(engine, weights_file):
    with pytest.raises(ExecutionError, match=r"\[Guard 3\]"):
        engine.rebalance_from_json(weights_file, **{**LOOSE, "max_turnover": 0.10})


def test_guard4_blocks_too_many_orders(engine, weights_file):
    with pytest.raises(ExecutionError, match=r"\[Guard 4\] 注文数 4 が上限 3"):
        engine.rebalance_from_json(weights_file, **{**LOOSE, "max_orders": 3})


def test_guard5_skips_orders_below_min_notional(engine, weights_file):
    engine.trading_client.positions = {"SPY": 119.0, "QQQ": 50.0, "IWM": 50.0, "BIL": 109.0}
    assert [o["symbol"] for o in engine.rebalance_from_json(weights_file, min_notional=20.0, **LOOSE)] == ["SPY"]
    assert engine.rebalance_from_json(weights_file, min_notional=1000.0, **LOOSE) == []


def test_blocked_run_submits_nothing(engine, weights_file):
    with pytest.raises(ExecutionError):
        engine.rebalance_from_json(weights_file, submit=True)
    assert engine.trading_client.submitted == []
    assert not engine.last_run_file.exists()


def test_blocked_run_is_logged(engine, weights_file):
    with pytest.raises(ExecutionError):
        engine.rebalance_from_json(weights_file, submit=True)
    rec = last_log(engine)
    assert rec["blocked"] is True
    assert any(r.startswith("[Guard 2]") for r in rec["reasons"])


# --- ガード6: 冪等ロック ---------------------------------------------------------------------

def test_guard6_blocks_second_submit_on_same_day(engine, weights_file):
    engine.last_run_file.write_text(json.dumps({"date_utc": today_utc()}), encoding="utf-8")
    with pytest.raises(ExecutionError, match=r"\[Guard 6\] 本日"):
        engine.rebalance_from_json(weights_file, submit=True, **LOOSE)


def test_daily_lock_submits_nothing(engine, weights_file):
    engine.last_run_file.write_text(json.dumps({"date_utc": today_utc()}), encoding="utf-8")
    with pytest.raises(ExecutionError):
        engine.rebalance_from_json(weights_file, submit=True, **LOOSE)
    assert engine.trading_client.submitted == []


def test_guard6_blocks_while_orders_are_pending(engine, weights_file):
    engine.trading_client.open_orders = [SimpleNamespace(side=OrderSide.BUY, qty="5", symbol="SPY")]
    with pytest.raises(ExecutionError, match="未約定の注文が残っています"):
        engine.rebalance_from_json(weights_file, submit=True, **LOOSE)


def test_pending_orders_submit_nothing(engine, weights_file):
    engine.trading_client.open_orders = [SimpleNamespace(side=OrderSide.BUY, qty="5", symbol="SPY")]
    with pytest.raises(ExecutionError):
        engine.rebalance_from_json(weights_file, submit=True, **LOOSE)
    assert engine.trading_client.submitted == []


def test_corrupt_lock_fails_closed(engine, weights_file):
    engine.last_run_file.write_text("{broken", encoding="utf-8")
    with pytest.raises(ExecutionError, match="読み込めません"):
        engine.rebalance_from_json(weights_file, submit=True, **LOOSE)
    assert engine.trading_client.submitted == []


# --- 途中失敗 -----------------------------------------------------------------------------

def test_partial_failure_propagates(engine, weights_file):
    engine.trading_client.fail_on_submit_no = 3
    with pytest.raises(RuntimeError, match="simulated API failure"):
        engine.rebalance_from_json(weights_file, submit=True, **LOOSE)
    assert len(engine.trading_client.submitted) == 2


def test_partial_failure_still_writes_lock(engine, weights_file):
    engine.trading_client.fail_on_submit_no = 3
    with pytest.raises(RuntimeError):
        engine.rebalance_from_json(weights_file, submit=True, **LOOSE)
    lock = json.loads(engine.last_run_file.read_text(encoding="utf-8"))
    assert lock["date_utc"] == today_utc() and lock["n_orders"] == 2


def test_partial_failure_is_logged_as_incomplete(engine, weights_file):
    engine.trading_client.fail_on_submit_no = 3
    with pytest.raises(RuntimeError):
        engine.rebalance_from_json(weights_file, submit=True, **LOOSE)
    rec = last_log(engine)
    assert rec["complete"] is False and len(rec["submitted"]) == 2


# --- 入力ファイルと認証情報 -------------------------------------------------------------------

def test_negative_weight_is_rejected(tmp_path):
    path = write_weights(tmp_path / "w.json", {**TARGET, "SPY": -0.1})
    with pytest.raises(ExecutionError, match="不正なウェイト"):
        load_target_weights(path)


def test_non_numeric_weight_is_rejected(tmp_path):
    path = write_weights(tmp_path / "w.json", {**TARGET, "SPY": "abc"})
    with pytest.raises(ExecutionError, match="数値でないウェイト"):
        load_target_weights(path)


def test_missing_weights_file_points_to_backtest(tmp_path):
    with pytest.raises(ExecutionError, match="python main.py backtest"):
        load_target_weights(tmp_path / "missing.json")


def test_missing_credentials_raise_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(broker, "ENV_FILE", tmp_path / "no.env")
    monkeypatch.setattr(broker, "TradingClient", FakeTradingClient)
    monkeypatch.setattr(broker, "StockHistoricalDataClient", FakeDataClient)
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with pytest.raises(ExecutionError, match="APCA_API_KEY_ID"):
        broker.AlpacaExecutionEngine(log_dir=tmp_path / "logs")


# --- 取引時間 (--require-market-open) ------------------------------------------------------

def test_closed_market_submits_with_notice_by_default(engine, weights_file, capsys):
    engine.trading_client.market_open = False
    engine.rebalance_from_json(weights_file, submit=True, **LOOSE)
    assert len(engine.trading_client.submitted) == 4
    assert "市場はクローズ中です" in capsys.readouterr().out


def test_require_market_open_blocks_submit_while_closed(engine, weights_file):
    engine.trading_client.market_open = False
    with pytest.raises(ExecutionError, match=r"\[Market Hours\]"):
        engine.rebalance_from_json(weights_file, submit=True, require_market_open=True, **LOOSE)
    assert engine.trading_client.submitted == []
    assert last_log(engine)["blocked"] is True


def test_require_market_open_allows_submit_while_open(engine, weights_file):
    engine.rebalance_from_json(weights_file, submit=True, require_market_open=True, **LOOSE)
    assert len(engine.trading_client.submitted) == 4


def test_require_market_open_only_warns_in_dry_run(engine, weights_file, capsys):
    engine.trading_client.market_open = False
    assert len(engine.rebalance_from_json(weights_file, require_market_open=True, **LOOSE)) == 4
    assert "--require-market-open が指定されているため" in capsys.readouterr().out


# --- 最新約定の鮮度 -------------------------------------------------------------------------

@pytest.mark.parametrize("days", [5, 9])
def test_stale_trade_blocks_even_dry_run(engine, weights_file, days):
    engine.data_client.trades["SPY"].timestamp = datetime.now(timezone.utc) - timedelta(days=days)
    with pytest.raises(ExecutionError, match=r"\[Stale Price\].*SPY"):
        engine.rebalance_from_json(weights_file, **LOOSE)


def test_stale_trade_blocks_submit(engine, weights_file):
    engine.data_client.trades["BIL"].timestamp = datetime.now(timezone.utc) - timedelta(days=6)
    with pytest.raises(ExecutionError, match=r"\[Stale Price\]"):
        engine.rebalance_from_json(weights_file, submit=True, **LOOSE)
    assert engine.trading_client.submitted == []


def test_four_day_old_trade_is_accepted(engine, weights_file):
    engine.data_client.trades["SPY"].timestamp = datetime.now(timezone.utc) - timedelta(days=4)
    assert len(engine.rebalance_from_json(weights_file, **LOOSE)) == 4


def test_missing_trade_timestamp_is_treated_as_stale(engine, weights_file):
    engine.data_client.trades["QQQ"].timestamp = None
    with pytest.raises(ExecutionError, match="QQQ: 時刻不明"):
        engine.rebalance_from_json(weights_file, **LOOSE)


def test_naive_trade_timestamp_is_read_as_utc(engine, weights_file):
    engine.data_client.trades["SPY"].timestamp = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1)
    assert len(engine.rebalance_from_json(weights_file, **LOOSE)) == 4


# --- CLI の配線 -----------------------------------------------------------------------------

def test_cli_passes_require_market_open(monkeypatch, tmp_path):
    calls: list[dict] = []

    class RecordingEngine:
        def __init__(self, log_dir):
            pass

        def rebalance_from_json(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(broker, "AlpacaExecutionEngine", RecordingEngine)
    monkeypatch.setattr(main, "LOG_DIR", tmp_path / "logs")
    main.main(["rebalance"])
    main.main(["rebalance", "--require-market-open", "--submit"])
    assert [(c["require_market_open"], c["submit"]) for c in calls] == [(False, False), (True, True)]
