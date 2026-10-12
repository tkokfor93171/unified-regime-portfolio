# src/execution/alpaca_broker.py
from __future__ import annotations

import json
import math
import os
from datetime import date, datetime, timezone
from pathlib import Path

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest
from dotenv import load_dotenv

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
STALE_WEIGHTS_DAYS = 7


class ExecutionError(RuntimeError):
    """設定不備や安全ガード違反など、発注を止めるべき状態。"""


def _value(x) -> str:
    return str(getattr(x, "value", x))


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_target_weights(path: str | Path) -> dict:
    """target_weights.json を読み込み、0以上の有限なウェイトだけを受け付ける。"""
    path = Path(path)
    if not path.exists():
        raise ExecutionError(f"{path} がありません。先に python main.py backtest を実行してください。")
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("weights")
    if not isinstance(raw, dict) or not raw:
        raise ExecutionError(f"{path} に weights がありません。")

    weights: dict[str, float] = {}
    for sym, w in raw.items():
        try:
            w = float(w)
        except (TypeError, ValueError):
            raise ExecutionError(f"数値でないウェイトがあります: {sym}={w!r}") from None
        if not math.isfinite(w) or w < 0:
            raise ExecutionError(f"不正なウェイトです: {sym}={w}（ロングオンリーのため0以上の有限値のみ）")
        weights[str(sym).upper()] = w
    payload["weights"] = weights
    return payload


class AlpacaExecutionEngine:
    def __init__(self, log_dir: str | Path = "logs"):
        load_dotenv(ENV_FILE)
        api_key = os.getenv("APCA_API_KEY_ID")
        secret_key = os.getenv("APCA_API_SECRET_KEY")
        if not api_key or not secret_key:
            raise ExecutionError(
                "APCA_API_KEY_ID または APCA_API_SECRET_KEY が .env に設定されていません（.env.example を参照）。"
            )

        # 安全のため常に paper=True で初期化
        self.trading_client = TradingClient(api_key, secret_key, paper=True)
        self.data_client = StockHistoricalDataClient(api_key, secret_key)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.last_run_file = self.log_dir / "last_run.json"
        self.trade_log_file = self.log_dir / "trade_log.jsonl"

    def audit_status(self, order_id: str | None = None, limit: int = 10) -> None:
        """whoami + positions + check_orders (+ 注文ID指定時は get_order_by_id) を一括表示する監査コマンド"""
        acct = self.trading_client.get_account()
        print(
            f"=== Account Status: {_value(acct.status)} | Equity: ${_num(acct.equity):,.2f} | "
            f"Cash: ${_num(acct.cash):,.2f} | Buying Power: ${_num(acct.buying_power):,.2f} ==="
        )

        positions = self.trading_client.get_all_positions()
        print(f"\n[Current Positions: {len(positions)}]")
        for p in positions:
            print(
                f"  {p.symbol:5s}: {str(p.qty):>8s} shares | AvgEntry: ${_num(p.avg_entry_price):>9.2f} | "
                f"MktValue: ${_num(p.market_value):>12,.2f} | Unrealized PnL: ${_num(p.unrealized_pl):>10,.2f}"
            )

        orders = self.trading_client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit))
        print(f"\n[Recent {limit} Orders]")
        for o in orders:
            print(
                f"  {str(o.created_at)[:19]} | {o.id} | {o.symbol:5s} | {_value(o.side):4s} | "
                f"Qty:{o.qty} | Filled:{o.filled_qty} | Status:{_value(o.status)}"
            )

        if order_id:
            o = self.trading_client.get_order_by_id(order_id)
            print(f"\n[Order {o.id}]")
            for field in ("symbol", "side", "status", "qty", "filled_qty", "filled_avg_price", "created_at", "filled_at"):
                print(f"  {field}: {_value(getattr(o, field, None))}")

    def rebalance_from_json(
        self,
        weights_path: str | Path = "outputs/target_weights.json",
        submit: bool = False,
        max_notional: float = 2000.0,
        max_turnover: float = 0.10,
        max_orders: int = 5,
        min_notional: float = 20.0,
        allow_multi_run_today: bool = False,
    ) -> list[dict]:
        """
        target_weights.json を読み込み、6つの安全ガードを適用してリバランスを実行する。

        ガード1: 既定はドライラン。--submit を明示したときだけ発注する（口座は常に paper）
        ガード2: 1注文あたりの想定元本の上限 (max_notional)
        ガード3: 総売買代金 / 口座評価額 の上限 (max_turnover)
        ガード4: 1回あたりの注文数の上限 (max_orders)
        ガード5: 最小発注金額未満の差分は発注しない (min_notional)
        ガード6: 冪等ロック。UTC日付で1日1回に制限し、未約定の注文が残っていれば送信しない

        ウェイトファイルに載っている銘柄だけを管理対象とし、それ以外の保有銘柄には触れない。
        """
        payload = load_target_weights(weights_path)
        target_weights: dict[str, float] = payload["weights"]
        symbols = sorted(target_weights)
        gross = sum(target_weights.values())
        held = ", ".join(f"{s}={w:.2%}" for s, w in target_weights.items() if w > 0) or "なし（全ポジション解消）"
        print(
            f"Loaded target weights (Strategy: {payload.get('strategy')}, AsOf: {payload.get('as_of')}, "
            f"Gross: {gross:.2f}): {held}"
        )
        self._warn_if_stale(payload.get("as_of"))
        if gross > 1.0 + 1e-9:
            print(f"  ※ グロス {gross:.0%} のため信用の買付余力を使います。paper 口座の buying power を確認してください。")

        now = datetime.now(timezone.utc)
        today_utc = now.strftime("%Y-%m-%d")

        acct = self.trading_client.get_account()
        equity = float(acct.equity)
        if equity <= 0:
            raise ExecutionError(f"口座評価額が0以下です (equity={equity})。")
        positions = {p.symbol: float(p.qty) for p in self.trading_client.get_all_positions()}
        unmanaged = sorted(set(positions) - set(symbols))

        latest = self.data_client.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbols))
        missing = [s for s in symbols if s not in latest]
        if missing:
            raise ExecutionError(f"最新価格を取得できない銘柄があります: {missing}")

        planned: list[dict] = []
        for sym in symbols:
            price = float(latest[sym].price)
            w = target_weights[sym]
            target_qty = math.floor(equity * w / price) if w > 0 else 0
            curr_qty = positions.get(sym, 0.0)
            delta_qty = int(target_qty - curr_qty)
            if delta_qty == 0:
                continue

            notional = abs(delta_qty) * price
            # ガード5: 最小発注金額未満はスキップ
            if notional < min_notional:
                continue
            planned.append({
                "symbol": sym,
                "side": "buy" if delta_qty > 0 else "sell",
                "qty": abs(delta_qty),
                "price_est": round(price, 4),
                "notional": round(notional, 2),
                "current_qty": curr_qty,
                "target_qty": target_qty,
            })

        # 売りを先に送って買付余力を確保する
        planned.sort(key=lambda o: (o["side"] != "sell", o["symbol"]))
        total_traded = sum(o["notional"] for o in planned)
        turnover_ratio = total_traded / equity

        reasons: list[str] = []
        # ガード2: 1注文あたりの最大想定元本チェック
        for o in planned:
            if o["notional"] > max_notional:
                reasons.append(
                    f"[Guard 2] {o['symbol']} の注文金額 ${o['notional']:,.2f} が上限 ${max_notional:,.2f} を超えています"
                )
        # ガード3: 総売買回転率の上限チェック
        if turnover_ratio > max_turnover:
            reasons.append(f"[Guard 3] 総回転率 {turnover_ratio:.1%} が上限 {max_turnover:.1%} を超えています")
        # ガード4: 最大注文数のチェック
        if len(planned) > max_orders:
            reasons.append(f"[Guard 4] 注文数 {len(planned)} が上限 {max_orders} を超えています")
        # ガード6: 冪等ロック（送信時のみ判定）
        open_orders = self.trading_client.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=symbols, limit=500)
        )
        if submit:
            if not allow_multi_run_today and self._last_submit_date() == today_utc:
                reasons.append(
                    f"[Guard 6] 本日 ({today_utc} UTC) は既に発注済みです。"
                    "再実行するには --allow-multi-run-today を指定してください"
                )
            if open_orders:
                pending = ", ".join(f"{_value(o.side)} {o.qty} {o.symbol}" for o in open_orders)
                reasons.append(
                    f"[Guard 6] 未約定の注文が残っています ({pending})。約定かキャンセルを確認してから再実行してください"
                )

        clock = self.trading_client.get_clock()
        print(f"\n[Rebalance Plan | Mode: {'SUBMIT (PAPER)' if submit else 'DRY-RUN'} | Equity: ${equity:,.2f}]")
        if not clock.is_open:
            print(f"  ※ 市場はクローズ中です。成行注文は次の寄り付き ({clock.next_open}) で執行されます。")
        for o in planned:
            print(
                f"  -> {o['side'].upper():4s} {o['qty']:5d} {o['symbol']:5s} @ ~${o['price_est']:,.2f} "
                f"(${o['notional']:,.2f})  保有 {o['current_qty']:g} -> 目標 {o['target_qty']}"
            )
        if not planned:
            print("  目標との差分はありません。")
        print(f"  合計 ${total_traded:,.2f} (評価額の {turnover_ratio:.2%}) / {len(planned)} 注文")
        if unmanaged:
            print(f"  管理対象外の保有銘柄（変更しません）: {', '.join(unmanaged)}")

        record = {
            "ts_utc": now.isoformat(),
            "mode": "submit" if submit else "dry-run",
            "strategy": payload.get("strategy"),
            "as_of": payload.get("as_of"),
            "equity": equity,
            "target_weights": target_weights,
            "positions": positions,
            "plans": planned,
            "turnover_ratio": round(turnover_ratio, 6),
        }

        if reasons:
            self._append_log({**record, "blocked": True, "reasons": reasons})
            raise ExecutionError("安全ガードにより停止しました。注文は送信していません。\n  - " + "\n  - ".join(reasons))

        if not submit:
            self._append_log({**record, "blocked": False})
            if open_orders:
                print("※ 未約定の注文が残っているため、このまま --submit するとガード6で停止します。")
            print("\n※ ドライラン完了。実際に発注するには --submit を付与してください。")
            return planned

        if not planned:
            self._append_log({**record, "blocked": False, "submitted": []})
            print("\n発注は不要です。")
            return []

        submitted: list[dict] = []
        try:
            for o in planned:
                req = MarketOrderRequest(
                    symbol=o["symbol"],
                    qty=o["qty"],
                    side=OrderSide.BUY if o["side"] == "buy" else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
                resp = self.trading_client.submit_order(order_data=req)
                submitted.append({**o, "order_id": str(resp.id), "status": _value(resp.status)})
                print(
                    f"  submitted: {o['side'].upper():4s} {o['qty']:5d} {o['symbol']:5s} "
                    f"| id: {resp.id} | status: {_value(resp.status)}"
                )
        finally:
            # 途中で失敗しても、送信済みの注文があれば同日の再送をロックする
            if submitted:
                lock = {
                    "date_utc": today_utc,
                    "ts_utc": now.isoformat(),
                    "strategy": payload.get("strategy"),
                    "n_orders": len(submitted),
                }
                self.last_run_file.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
            self._append_log({**record, "blocked": False, "submitted": submitted, "complete": len(submitted) == len(planned)})

        print(f"\n発注完了: {len(submitted)} 件の注文を送信し、{self.trade_log_file} に記録しました。")
        return submitted

    def _last_submit_date(self) -> str | None:
        if not self.last_run_file.exists():
            return None
        try:
            return json.loads(self.last_run_file.read_text(encoding="utf-8")).get("date_utc")
        except (OSError, json.JSONDecodeError) as e:
            raise ExecutionError(f"{self.last_run_file} を読み込めません。内容を確認してください: {e}") from None

    def _append_log(self, record: dict) -> None:
        with self.trade_log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    @staticmethod
    def _warn_if_stale(as_of: str | None) -> None:
        try:
            age = (date.today() - date.fromisoformat(str(as_of))).days
        except ValueError:
            print(f"  ※ as_of ({as_of}) を日付として解釈できません。")
            return
        if age > STALE_WEIGHTS_DAYS:
            print(f"  ※ ウェイトの基準日が {age} 日前です。python main.py backtest で更新してから発注してください。")
