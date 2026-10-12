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
from src.allocators.ensemble import ENSEMBLE_MODES, EnsembleAllocator
from src.backtest.engine import compute_metrics, evaluate_by_crowding_regime, run_backtest
from src.backtest.optimize import random_search_core_overlay
from src.backtest.plot import plot_performance, plot_regime_performance
from src.data.loader import load_market_data

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
LOG_DIR = ROOT / "logs"
DEFAULT_WEIGHTS_FILE = OUTPUT_DIR / "target_weights.json"
# defensive_multi の12ヶ月リターンと36ヶ月Zスコア（計48ヶ月 ≒ 1,461暦日）が評価開始日までにそろう長さ
WARMUP_CALENDAR_DAYS = 1500


def run_label(args: argparse.Namespace) -> str:
    """出力ファイル名と表示に使う名前。ensemble は設定違いの結果を並べて残せるよう比率か窓幅を含める。"""
    if args.strategy == "ensemble":
        if args.ensemble_mode == "inv_vol":
            return f"ensemble_invvol{args.vol_window}"
        return f"ensemble_core{round(args.core_ratio * 100)}"
    return args.strategy


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


def warn_short_history(prices: pd.DataFrame, load_start: pd.Timestamp, first_day: pd.Timestamp) -> None:
    """ウォームアップ用のデータが足りない場合や、評価期間の途中から価格がある銘柄がある場合に注意を表示する。"""
    if prices.index[0] > load_start + pd.Timedelta(days=7):
        print(
            f"  ※ データは {prices.index[0].date()} からしかなく、ウォームアップが {WARMUP_CALENDAR_DAYS} 暦日に足りません。"
            "評価期間の初めはシグナルが未成熟です。"
        )
    late = {sym: d for sym, d in prices.apply(pd.Series.first_valid_index).items() if d > first_day}
    if late:
        listed = ", ".join(f"{sym} ({d.date()}~)" for sym, d in late.items())
        print(f"  ※ 評価開始日より後に価格が始まる銘柄があります: {listed}。それまでのリターンは0として扱います。")


def cmd_backtest(args: argparse.Namespace) -> None:
    # --start より前のデータはシグナルのウォームアップにだけ使い、損益と成績は --start 以降で計上する
    eval_start = pd.Timestamp(args.start)
    load_start = eval_start - pd.Timedelta(days=WARMUP_CALENDAR_DAYS)
    prices, dollar_vol = load_market_data(start_date=load_start.strftime("%Y-%m-%d"), end_date=args.end)
    window = prices.index[prices.index >= eval_start]
    if window.empty:
        sys.exit(f"[STOPPED] --start {args.start} 以降の価格データがありません。")
    print(f"Loaded {list(prices.columns)}: {prices.index[0].date()} -> {prices.index[-1].date()} ({len(prices)} days)")
    print(
        f"Backtest window: {window[0].date()} -> {window[-1].date()} ({len(window)} days), "
        f"signals warmed up over the preceding {len(prices) - len(window)} days"
    )
    warn_short_history(prices, load_start, window[0])

    holdout = None
    params_note = ""
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
                eval_start=window[0],
            )
            holdout = result.holdout
            print("\n=== Best Params (selected on TRAIN only) ===")
            for k, v in asdict(result.params).items():
                print(f"  {k}: {v}")
            print(f"\n=== Holdout Check (TEST: {result.split_date.date()} ->) ===")
            print(holdout.to_string())

            alpha = result.holdout_alpha
            if result.accepted:
                params, params_note = result.params, "optimized params"
                print(f"[ADOPTED] 検証期間のアルファが {alpha:+.2f}% で正のため、最適化パラメータを採用します。")
            elif args.force_optimized:
                params, params_note = result.params, "optimized params (forced)"
                print(
                    f"[FORCED] 検証期間のアルファは {alpha:+.2f}% で正ではありませんが、"
                    "--force-optimized により最適化パラメータを採用します。"
                )
            else:
                params_note = "default params"
                print(
                    f"[REJECTED] 検証期間のアルファが {alpha:+.2f}% で正ではないため、"
                    "過学習と判定して最適化パラメータを棄却しました。\n"
                    "           target_weights.json には既定パラメータのウェイトを書き出します（強制採用は --force-optimized）。"
                )
        allocator: BaseAllocator = CoreOverlayAllocator(params)
    elif args.strategy == "ensemble":
        if args.ensemble_mode == "inv_vol":
            allocator = EnsembleAllocator(mode="inv_vol", vol_window=args.vol_window)
        else:
            allocator = EnsembleAllocator(core_ratio=args.core_ratio)
    else:
        allocator = DefensiveMultiAllocator()
    label = run_label(args)

    weights, crowd_series = allocator.generate_daily_weights(prices, dollar_vol)
    alpha_t = allocator.alpha_series_ if isinstance(allocator, EnsembleAllocator) else None

    prices, weights, crowd_series = prices.loc[window[0]:], weights.loc[window[0]:], crowd_series.loc[window[0]:]
    if alpha_t is not None:
        alpha_t = alpha_t.loc[window[0]:]
    bt = run_backtest(prices, weights, cost_bps=args.cost_bps)

    overall = compute_metrics(bt["port_ret"], bt["spy_ret"], bt["turnover"])
    spy_bench = compute_metrics(bt["spy_ret"], bt["spy_ret"])
    summary = pd.DataFrame.from_dict({"Strategy": overall, "SPY_BuyHold": spy_bench}, orient="index")
    regime_df = evaluate_by_crowding_regime(bt, crowd_series)
    tables = [summary] if holdout is None else [summary, holdout]

    header = f"{label}, {params_note}" if params_note else label
    print(f"\n=== Strategy Backtest Summary: [{header}] (Cost: {args.cost_bps} bps) ===")
    print(summary.to_string())
    if alpha_t is not None and args.ensemble_mode == "inv_vol":
        print(
            f"α_t (core_overlay への配分比率): 平均 {alpha_t.mean():.3f} / 最小 {alpha_t.min():.3f} / "
            f"最大 {alpha_t.max():.3f} / 最新 {alpha_t.iloc[-1]:.3f}"
        )
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
    if alpha_t is not None:
        daily["core_ratio"] = alpha_t
    daily.to_csv(OUTPUT_DIR / f"{label}_daily.csv")
    weights.to_csv(OUTPUT_DIR / f"{label}_weights.csv")
    pd.concat(tables).to_csv(OUTPUT_DIR / f"{label}_metrics.csv")
    title = f"{label} ({prices.index[0].date()} - {prices.index[-1].date()}, {args.cost_bps:g} bps)"
    plot_performance(bt, title, OUTPUT_DIR / f"{label}_performance.png")
    plot_regime_performance(regime_df, title, OUTPUT_DIR / f"{label}_regime.png")
    print(f"\n[Saved] outputs/{label}_{{daily,weights,metrics}}.csv, outputs/{label}_{{performance,regime}}.png")

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
            require_market_open=args.require_market_open,
        )
    except ExecutionError as e:
        sys.exit(f"\n[STOPPED] {e}")


def cmd_audit(args: argparse.Namespace) -> None:
    from src.execution.alpaca_broker import AlpacaExecutionEngine, ExecutionError

    try:
        AlpacaExecutionEngine(log_dir=LOG_DIR).audit_status(order_id=args.order_id)
    except ExecutionError as e:
        sys.exit(f"[STOPPED] {e}")


def parse_core_ratio(value: str) -> float | str:
    """--core-ratio の値。auto / inv_vol は逆ボラ比率モードの指定として受け付ける。"""
    if value.strip().lower() in ("auto", "inv_vol"):
        return "inv_vol"
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"0.0〜1.0 の数値か auto / inv_vol を指定してください: {value!r}") from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified Regime & Market-Structure Portfolio CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. Backtest & Export Weights
    p_bt = sub.add_parser("backtest", help="Run backtest and export latest target_weights.json")
    p_bt.add_argument("--strategy", choices=["core_overlay", "defensive_multi", "ensemble"], default="core_overlay")
    p_bt.add_argument(
        "--core-ratio",
        "--blend-ratio",
        dest="core_ratio",
        type=parse_core_ratio,
        default=None,
        help="ensemble only: capital share of core_overlay in [0, 1], the rest goes to defensive_multi (Default: 0.5). "
        "'auto' or 'inv_vol' selects --ensemble-mode inv_vol",
    )
    p_bt.add_argument(
        "--ensemble-mode",
        choices=ENSEMBLE_MODES,
        default=None,
        help="ensemble only: fixed = constant --core-ratio, inv_vol = inverse-volatility ratio of the two strategies "
        "updated every 5 days (Default: fixed)",
    )
    p_bt.add_argument(
        "--vol-window",
        type=int,
        default=None,
        help="ensemble inv_vol only: lookback days for each strategy's volatility (Default: 63)",
    )
    p_bt.add_argument(
        "--start",
        default="2010-01-01",
        help=f"Backtest start date. Data from {WARMUP_CALENDAR_DAYS} calendar days earlier is used only to warm up signals",
    )
    p_bt.add_argument("--end", default=None, help="End date (exclusive). Default: latest available")
    p_bt.add_argument("--cost-bps", type=float, default=10.0)
    p_bt.add_argument("--optimize", action="store_true", help="core_overlay only: capped random search on TRAIN, report TEST")
    p_bt.add_argument(
        "--force-optimized",
        action="store_true",
        help="With --optimize: adopt the optimized params even if the holdout (TEST) alpha is not positive",
    )
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
    p_ex.add_argument(
        "--require-market-open",
        action="store_true",
        help="Block --submit while the market is closed (Default: warn and queue orders for the next open)",
    )
    p_ex.set_defaults(func=cmd_execute)

    # 3. Account & Order Audit
    p_au = sub.add_parser("audit", help="Inspect Alpaca paper account, holdings, and recent orders")
    p_au.add_argument("--order-id", default=None, help="Also show details of this order")
    p_au.set_defaults(func=cmd_audit)
    return parser


def resolve_ensemble_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """ensemble の比率指定を ensemble_mode / core_ratio / vol_window の組み合わせに正規化する。"""
    if args.strategy != "ensemble":
        for flag, value in (
            ("--core-ratio", args.core_ratio),
            ("--ensemble-mode", args.ensemble_mode),
            ("--vol-window", args.vol_window),
        ):
            if value is not None:
                parser.error(f"{flag} は --strategy ensemble のときだけ使えます")
        return

    if args.core_ratio == "inv_vol":
        if args.ensemble_mode == "fixed":
            parser.error("--core-ratio auto と --ensemble-mode fixed は同時に指定できません")
        args.ensemble_mode, args.core_ratio = "inv_vol", None
    args.ensemble_mode = args.ensemble_mode or "fixed"

    if args.ensemble_mode == "fixed":
        if args.vol_window is not None:
            parser.error("--vol-window は --ensemble-mode inv_vol のときだけ使えます")
        args.core_ratio = 0.5 if args.core_ratio is None else args.core_ratio
        if not 0.0 <= args.core_ratio <= 1.0:
            parser.error("--core-ratio は 0.0〜1.0 で指定してください")
    else:
        if args.core_ratio is not None:
            parser.error("--ensemble-mode inv_vol は比率を自動で決めるため --core-ratio とは併用できません")
        args.vol_window = 63 if args.vol_window is None else args.vol_window
        if args.vol_window < 10:
            parser.error("--vol-window は10以上で指定してください")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "backtest":
        try:
            pd.Timestamp(args.start)
        except ValueError:
            parser.error(f"--start を日付として解釈できません: {args.start}")
        if args.optimize and args.strategy != "core_overlay":
            parser.error("--optimize は --strategy core_overlay のときだけ使えます")
        if args.force_optimized and not args.optimize:
            parser.error("--force-optimized は --optimize と一緒に指定してください")
        resolve_ensemble_args(parser, args)
        if not 0.0 < args.test_frac < 1.0:
            parser.error("--test-frac は 0 と 1 の間で指定してください")
    args.func(args)


if __name__ == "__main__":
    main()
