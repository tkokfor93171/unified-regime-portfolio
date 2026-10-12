from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.allocators.base import BaseAllocator
from src.allocators.core_overlay import CoreOverlayAllocator, CoreOverlayParams
from src.allocators.defensive_multi import DefensiveMultiAllocator
from src.backtest.engine import compute_metrics, evaluate_by_crowding_regime, run_backtest
from src.backtest.optimize import random_search_core_overlay
from src.backtest.plot import plot_performance, plot_regime_performance
from src.data.loader import load_market_data

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
LOG_DIR = ROOT / "logs"
DEFAULT_WEIGHTS_FILE = OUTPUT_DIR / "target_weights.json"


def split_metrics(bt: pd.DataFrame, split_date: pd.Timestamp) -> pd.DataFrame:
    rows = {}
    for label, mask in (("TRAIN", bt.index < split_date), ("TEST", bt.index >= split_date)):
        part = bt.loc[mask]
        rows[f"{label}_Strategy"] = compute_metrics(part["port_ret"], part["spy_ret"], part["turnover"])
        rows[f"{label}_SPY"] = compute_metrics(part["spy_ret"], part["spy_ret"])
    return pd.DataFrame.from_dict(rows, orient="index")


def export_target_weights(strategy: str, weights: pd.DataFrame, params: dict, path: Path) -> dict:
    payload = {
        "strategy": strategy,
        "as_of": str(weights.index[-1].date()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "params": params,
        # 0 の銘柄も残し、執行側が管理対象として売却できるようにする
        "weights": {sym: round(float(w), 4) for sym, w in weights.iloc[-1].items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def cmd_backtest(args: argparse.Namespace) -> None:
    prices, dollar_vol = load_market_data(start_date=args.start, end_date=args.end)
    print(f"Loaded {list(prices.columns)}: {prices.index[0].date()} -> {prices.index[-1].date()} ({len(prices)} days)")

    split_date = None
    if args.strategy == "core_overlay":
        params = CoreOverlayParams()
        if args.optimize:
            result = random_search_core_overlay(
                prices,
                dollar_vol,
                cost_bps=args.cost_bps,
                n_candidates=args.n_candidates,
                patience=args.patience,
                test_frac=args.test_frac,
                seed=args.seed,
            )
            params, split_date = result.params, result.split_date
            print("\n=== Best Params (selected on TRAIN only) ===")
            for k, v in asdict(params).items():
                print(f"  {k}: {v}")
        allocator: BaseAllocator = CoreOverlayAllocator(params)
    else:
        allocator = DefensiveMultiAllocator()

    weights, crowd_series = allocator.generate_daily_weights(prices, dollar_vol)
    bt = run_backtest(prices, weights, cost_bps=args.cost_bps)

    overall = compute_metrics(bt["port_ret"], bt["spy_ret"], bt["turnover"])
    spy_bench = compute_metrics(bt["spy_ret"], bt["spy_ret"])
    summary = pd.DataFrame.from_dict({"Strategy": overall, "SPY_BuyHold": spy_bench}, orient="index")
    regime_df = evaluate_by_crowding_regime(bt, crowd_series)
    tables = [summary]

    print(f"\n=== Strategy Backtest Summary: [{args.strategy}] (Cost: {args.cost_bps} bps) ===")
    print(summary.to_string())
    if split_date is not None:
        split_df = split_metrics(bt, split_date)
        tables.append(split_df)
        print(f"\n=== Selection (TRAIN) vs Holdout (TEST: {split_date.date()} ->) ===")
        print(split_df.to_string())
    print("\n=== Conditional Performance by Crowding Regime ===")
    print(regime_df.to_string())
    tables.append(regime_df)

    # 成績CSVとチャートPNGを戦略名つきで保存（戦略同士を並べて比較できるようにする）
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily = bt.assign(
        equity=(1.0 + bt["port_ret"]).cumprod(),
        spy_equity=(1.0 + bt["spy_ret"]).cumprod(),
        regime_indicator=crowd_series,
    )
    daily.to_csv(OUTPUT_DIR / f"{args.strategy}_daily.csv")
    weights.to_csv(OUTPUT_DIR / f"{args.strategy}_weights.csv")
    pd.concat(tables).to_csv(OUTPUT_DIR / f"{args.strategy}_metrics.csv")
    title = f"{args.strategy} ({prices.index[0].date()} - {prices.index[-1].date()}, {args.cost_bps:g} bps)"
    plot_performance(bt, title, OUTPUT_DIR / f"{args.strategy}_performance.png")
    plot_regime_performance(regime_df, title, OUTPUT_DIR / f"{args.strategy}_regime.png")
    print(f"\n[Saved] outputs/{args.strategy}_{{daily,weights,metrics}}.csv, outputs/{args.strategy}_{{performance,regime}}.png")

    # 最新日のターゲットウェイトを執行エンジン向けにJSON出力
    payload = export_target_weights(args.strategy, weights, allocator.params_dict(), DEFAULT_WEIGHTS_FILE)
    held = {k: v for k, v in payload["weights"].items() if v > 0}
    print(
        f"[Exported] 最新ターゲットウェイト ({payload['as_of']}) を "
        f"{DEFAULT_WEIGHTS_FILE.relative_to(ROOT)} に出力しました: {held}"
    )


def cmd_execute(args: argparse.Namespace) -> None:
    from src.execution.alpaca_broker import AlpacaExecutionEngine, ExecutionError

    try:
        engine = AlpacaExecutionEngine(log_dir=LOG_DIR)
        engine.rebalance_from_json(
            weights_path=args.weights_file,
            submit=args.submit,
            max_notional=args.max_notional,
            max_turnover=args.max_turnover,
            max_orders=args.max_orders,
            min_notional=args.min_notional,
            allow_multi_run_today=args.allow_multi_run_today,
        )
    except ExecutionError as e:
        sys.exit(f"\n[STOPPED] {e}")


def cmd_audit(args: argparse.Namespace) -> None:
    from src.execution.alpaca_broker import AlpacaExecutionEngine, ExecutionError

    try:
        AlpacaExecutionEngine(log_dir=LOG_DIR).audit_status(order_id=args.order_id)
    except ExecutionError as e:
        sys.exit(f"[STOPPED] {e}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified Regime & Market-Structure Portfolio CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. Backtest & Export Weights
    p_bt = sub.add_parser("backtest", help="Run backtest and export latest target_weights.json")
    p_bt.add_argument("--strategy", choices=["core_overlay", "defensive_multi"], default="core_overlay")
    p_bt.add_argument("--start", default="2010-01-01")
    p_bt.add_argument("--end", default=None, help="End date (exclusive). Default: latest available")
    p_bt.add_argument("--cost-bps", type=float, default=10.0)
    p_bt.add_argument("--optimize", action="store_true", help="core_overlay only: capped random search on TRAIN, report TEST")
    p_bt.add_argument("--n-candidates", type=int, default=200, help="Random search: max candidates")
    p_bt.add_argument("--patience", type=int, default=60, help="Random search: early stop after N non-improving candidates")
    p_bt.add_argument("--test-frac", type=float, default=0.20, help="Random search: holdout fraction at the end")
    p_bt.add_argument("--seed", type=int, default=42)
    p_bt.set_defaults(func=cmd_backtest)

    # 2. Rebalance Execution (Alpaca Paper)
    p_ex = sub.add_parser("rebalance", aliases=["execute"], help="Rebalance Alpaca paper account from target_weights.json")
    p_ex.add_argument("--weights-file", default=str(DEFAULT_WEIGHTS_FILE))
    p_ex.add_argument("--submit", action="store_true", help="Send actual paper orders (Default: dry-run)")
    p_ex.add_argument("--max-notional", type=float, default=2000.0, help="Max notional per order (USD)")
    p_ex.add_argument("--max-turnover", type=float, default=0.10, help="Max total traded / equity")
    p_ex.add_argument("--max-orders", type=int, default=5)
    p_ex.add_argument("--min-notional", type=float, default=20.0, help="Skip orders smaller than this (USD)")
    p_ex.add_argument("--allow-multi-run-today", action="store_true")
    p_ex.set_defaults(func=cmd_execute)

    # 3. Account & Order Audit
    p_au = sub.add_parser("audit", help="Inspect Alpaca paper account, holdings, and recent orders")
    p_au.add_argument("--order-id", default=None, help="Also show details of this order")
    p_au.set_defaults(func=cmd_audit)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "backtest":
        if args.optimize and args.strategy != "core_overlay":
            parser.error("--optimize は --strategy core_overlay のときだけ使えます")
        if not 0.0 < args.test_frac < 1.0:
            parser.error("--test-frac は 0 と 1 の間で指定してください")
    args.func(args)


if __name__ == "__main__":
    main()
