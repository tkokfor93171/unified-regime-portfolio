import json

import numpy as np
import pandas as pd
import pytest

import main
from src.allocators.core_overlay import CoreOverlayAllocator
from src.allocators.defensive_multi import DefensiveMultiAllocator
from src.allocators.ensemble import EnsembleAllocator
from src.backtest.engine import run_backtest


@pytest.fixture(scope="module")
def singles(market):
    prices, dollar_vol = market
    return {
        "core": CoreOverlayAllocator().generate_daily_weights(prices, dollar_vol),
        "def": DefensiveMultiAllocator().generate_daily_weights(prices, dollar_vol),
    }


@pytest.fixture(scope="module")
def inv_vol_run(market):
    allocator = EnsembleAllocator(mode="inv_vol")
    weights, regime = allocator.generate_daily_weights(*market)
    return allocator, weights, regime


# --- 固定比率 -------------------------------------------------------------------------------

def test_core_ratio_one_equals_core_overlay(market, singles):
    prices, dollar_vol = market
    w, regime = EnsembleAllocator(core_ratio=1.0).generate_daily_weights(prices, dollar_vol)
    w_core, crowd_core = singles["core"]

    pd.testing.assert_frame_equal(w, w_core, check_exact=True)
    pd.testing.assert_series_equal(regime, crowd_core, check_exact=True)
    pd.testing.assert_frame_equal(run_backtest(prices, w), run_backtest(prices, w_core), check_exact=True)


def test_core_ratio_zero_equals_defensive_multi(market, singles):
    prices, dollar_vol = market
    w, regime = EnsembleAllocator(core_ratio=0.0).generate_daily_weights(prices, dollar_vol)
    w_def, _ = singles["def"]

    pd.testing.assert_frame_equal(w, w_def, check_exact=True)
    pd.testing.assert_frame_equal(run_backtest(prices, w), run_backtest(prices, w_def), check_exact=True)
    # レジーム別評価は比率によらず core_overlay の日次 Crowding スコアで行う仕様
    pd.testing.assert_series_equal(regime, singles["core"][1], check_exact=True)


@pytest.mark.parametrize("ratio", [0.3, 0.5, 0.7])
def test_blend_is_linear_combination(market, singles, ratio):
    allocator = EnsembleAllocator(core_ratio=ratio)
    w, _ = allocator.generate_daily_weights(*market)
    expected = ratio * singles["core"][0] + (1.0 - ratio) * singles["def"][0]
    pd.testing.assert_frame_equal(w, expected, check_exact=True)
    assert (allocator.alpha_series_ == ratio).all() and allocator.latest_alpha_ == ratio


def test_fixed_params_dict_records_both_strategies():
    params = EnsembleAllocator(core_ratio=0.7).params_dict()
    assert params == {
        "ensemble_mode": "fixed",
        "core_ratio": 0.7,
        "effective_core_ratio": 0.7,
        "core_overlay": CoreOverlayAllocator().params_dict(),
        "defensive_multi": DefensiveMultiAllocator().params_dict(),
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"core_ratio": -0.1},
        {"core_ratio": 1.1},
        {"core_ratio": float("nan")},
        {"mode": "risk_parity"},
        {"mode": "inv_vol", "vol_window": 9},
        {"mode": "inv_vol", "rebal_freq": 0},
        {"mode": "inv_vol", "min_vol": 0.0},
    ],
)
def test_invalid_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        EnsembleAllocator(**kwargs)


# --- 逆ボラ比率 (inv_vol) --------------------------------------------------------------------

def test_inv_vol_alpha_follows_spec(market, singles, inv_vol_run):
    """前日ウェイト × 当日リターンの直近63日年率ボラ（下限5%）から α_t = vol_def / (vol_core + vol_def) を手計算と照合する。"""
    prices, _ = market
    allocator, _, _ = inv_vol_run
    ret = prices.pct_change().fillna(0.0)
    r_core = (singles["core"][0].shift(1).fillna(0.0) * ret).sum(axis=1).to_numpy()
    r_def = (singles["def"][0].shift(1).fillna(0.0) * ret).sum(axis=1).to_numpy()
    alpha = allocator.alpha_series_.to_numpy()

    def ann_vol(r: np.ndarray, i: int) -> float:
        return max(float(np.std(r[max(0, i - 62) : i + 1], ddof=1)) * np.sqrt(252), 0.05)

    # 最小観測数 max(10, 63 // 2) = 31日に届くまでは 0.5
    assert (alpha[:30] == 0.5).all()
    for i in range(30, len(alpha), 5):
        v_core, v_def = ann_vol(r_core, i), ann_vol(r_def, i)
        assert alpha[i] == pytest.approx(v_def / (v_core + v_def), rel=1e-9)
        # 次の更新日までは据え置き
        assert (alpha[i : i + 5] == alpha[i]).all()


def test_inv_vol_alpha_changes_only_on_update_days(inv_vol_run):
    alpha = inv_vol_run[0].alpha_series_.to_numpy()
    changed = np.flatnonzero(alpha[1:] != alpha[:-1]) + 1
    assert len(changed) > 100
    assert (changed % 5 == 0).all()


def test_volatility_floor_keeps_alpha_finite():
    allocator = EnsembleAllocator(mode="inv_vol", vol_window=20, rebal_freq=1)
    r_core = pd.Series(np.random.default_rng(1).normal(0.0, 0.01, 200))
    flat = pd.Series(0.0, index=r_core.index)  # defensive_multi が BIL 100% で値動きがない状態

    alpha = allocator.inverse_vol_alpha(r_core, flat)
    vol_core = (r_core.rolling(20, min_periods=10).std() * np.sqrt(252)).clip(lower=0.05)
    pd.testing.assert_series_equal(alpha, (0.05 / (vol_core + 0.05)).fillna(0.5), check_names=False)
    assert np.isfinite(alpha).all() and alpha.between(0.0, 0.5).all()
    # 両方とも下限に張り付けば等分
    assert (allocator.inverse_vol_alpha(flat, flat) == 0.5).all()


def test_inv_vol_blend_uses_daily_ratio(singles, inv_vol_run):
    allocator, weights, regime = inv_vol_run
    alpha = allocator.alpha_series_
    expected = singles["core"][0].mul(alpha, axis=0) + singles["def"][0].mul(1.0 - alpha, axis=0)
    pd.testing.assert_frame_equal(weights, expected, check_exact=True)
    pd.testing.assert_series_equal(regime, singles["core"][1], check_exact=True)


def test_inv_vol_records_effective_ratio(inv_vol_run):
    allocator = inv_vol_run[0]
    assert allocator.latest_alpha_ == allocator.alpha_series_.iloc[-1]
    params = allocator.params_dict()
    assert params["ensemble_mode"] == "inv_vol"
    assert (params["vol_window"], params["rebal_freq"], params["min_vol"]) == (63, 5, 0.05)
    assert params["effective_core_ratio"] == round(allocator.latest_alpha_, 4)
    assert "core_ratio" not in params


# --- CLI ------------------------------------------------------------------------------------

def run_ensemble_cli(out_dir, *args: str) -> dict:
    main.main(["backtest", "--strategy", "ensemble", *args])
    return json.loads((out_dir / "target_weights.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "args",
    [["--ensemble-mode", "inv_vol"], ["--core-ratio", "auto"], ["--core-ratio", "inv_vol"], ["--blend-ratio", "AUTO"]],
)
def test_inv_vol_run_is_saved_under_window_label(backtest_cli, market, capsys, args):
    payload = run_ensemble_cli(backtest_cli, *args)
    expected = EnsembleAllocator(mode="inv_vol", vol_window=63)
    weights, _ = expected.generate_daily_weights(*market)

    assert payload["params"]["ensemble_mode"] == "inv_vol"
    assert payload["params"]["vol_window"] == 63
    assert payload["params"]["effective_core_ratio"] == round(expected.latest_alpha_, 4)
    assert payload["weights"] == {sym: round(float(w), 4) for sym, w in weights.iloc[-1].items()}
    assert (backtest_cli / "ensemble_invvol63_weights.csv").exists()
    assert not list(backtest_cli.glob("ensemble_core*"))
    assert "α_t (core_overlay への配分比率)" in capsys.readouterr().out


def test_vol_window_is_part_of_the_label(backtest_cli):
    payload = run_ensemble_cli(backtest_cli, "--ensemble-mode", "inv_vol", "--vol-window", "21")
    assert payload["params"]["vol_window"] == 21
    daily = pd.read_csv(backtest_cli / "ensemble_invvol21_daily.csv", index_col=0)
    assert daily["core_ratio"].between(0.0, 1.0).all()
    assert daily["core_ratio"].iloc[-1] == pytest.approx(payload["params"]["effective_core_ratio"], abs=5e-5)


def test_fixed_mode_keeps_ratio_label(backtest_cli):
    payload = run_ensemble_cli(backtest_cli, "--core-ratio", "0.3")
    assert payload["params"]["ensemble_mode"] == "fixed"
    assert payload["params"]["core_ratio"] == payload["params"]["effective_core_ratio"] == 0.3
    assert "vol_window" not in payload["params"]
    daily = pd.read_csv(backtest_cli / "ensemble_core30_daily.csv", index_col=0)
    assert (daily["core_ratio"] == 0.3).all()


@pytest.mark.parametrize(
    "args",
    [
        ["--strategy", "ensemble", "--ensemble-mode", "inv_vol", "--core-ratio", "0.5"],
        ["--strategy", "ensemble", "--ensemble-mode", "fixed", "--core-ratio", "auto"],
        ["--strategy", "ensemble", "--vol-window", "63"],
        ["--strategy", "ensemble", "--ensemble-mode", "inv_vol", "--vol-window", "5"],
        ["--strategy", "ensemble", "--core-ratio", "1.5"],
        ["--strategy", "ensemble", "--core-ratio", "abc"],
        ["--strategy", "core_overlay", "--ensemble-mode", "inv_vol"],
        ["--strategy", "defensive_multi", "--core-ratio", "auto"],
    ],
)
def test_conflicting_ensemble_options_are_rejected(backtest_cli, args):
    with pytest.raises(SystemExit) as exc:
        main.main(["backtest", *args])
    assert exc.value.code == 2
