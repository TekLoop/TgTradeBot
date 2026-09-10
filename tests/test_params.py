"""Параметры на таймфрейм: разрешение, валидация, границы (Задачи 1 и 3)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import AppConfig, ParamsConfig, build_params
from app.core.timeframes import Timeframe


def test_acceptance_config_is_valid():
    """Эталонный конфиг приёмки должен проходить валидацию."""
    params = ParamsConfig(
        anchor_ttl_bars=40,
        max_bars_between_points=40,
        min_bars_between_points=0,
        chain_max_points=4,
    )
    assert params.max_bars_between_points == 40


def test_zero_is_accepted_for_all_four():
    params = ParamsConfig(
        anchor_ttl_bars=0,
        max_bars_between_points=0,
        min_bars_between_points=0,
        chain_max_points=0,
    )
    assert params.chain_max_points == 0


def test_chain_max_points_one_is_rejected():
    with pytest.raises(ValidationError) as exc:
        ParamsConfig(chain_max_points=1)
    assert "chain_max_points" in str(exc.value)


# --- четыре правила валидации Задачи 3 --------------------------------------


def test_min_distance_must_clear_fractal_width():
    ParamsConfig(fractal_n=2, min_bars_between_points=3, max_bars_between_points=40)
    with pytest.raises(ValidationError):
        ParamsConfig(fractal_n=2, min_bars_between_points=2)


def test_max_equal_to_ttl_passes_and_greater_is_rejected():
    """Граница: max == ttl проходит (TTL убивает при СТРОГОМ превышении)."""
    ParamsConfig(anchor_ttl_bars=40, max_bars_between_points=40)
    with pytest.raises(ValidationError):
        ParamsConfig(anchor_ttl_bars=40, max_bars_between_points=41)


def test_ttl_must_fit_into_window():
    ParamsConfig(lookback_bars=300, rsi_period=14, fractal_n=2, anchor_ttl_bars=281)
    with pytest.raises(ValidationError):
        ParamsConfig(lookback_bars=300, rsi_period=14, fractal_n=2, anchor_ttl_bars=282)


def test_min_must_be_below_max():
    with pytest.raises(ValidationError):
        ParamsConfig(
            fractal_n=2, min_bars_between_points=40, max_bars_between_points=40
        )


def test_ttl_checked_against_own_timeframe_lookback():
    """lookback_bars тоже переопределяем на ТФ — правило 3 должно считаться
    по СОБРАННОМУ конфигу, а не по глобальному значению."""
    base = ParamsConfig(lookback_bars=300, anchor_ttl_bars=200)
    with pytest.raises(ValidationError):
        build_params(base, {"lookback_bars": 100})


# --- порядок наложения слоёв ------------------------------------------------


def test_layers_apply_in_order():
    base = ParamsConfig(oversold=30.0)
    resolved = build_params(
        base,
        {"oversold": 28.0},   # YAML params_by_timeframe
        {"oversold": 26.0},   # БД, глобально
        {"oversold": 24.0},   # БД, на таймфрейм
    )
    assert resolved.oversold == 24.0

    assert build_params(base, {"oversold": 28.0}).oversold == 28.0
    assert build_params(base, None, {"oversold": 26.0}).oversold == 26.0


def test_unknown_keys_are_ignored_not_fatal():
    """Мусор, переживший миграцию, не должен валить опрос."""
    resolved = build_params(ParamsConfig(), {"max_bars_between": 60})
    assert resolved.max_bars_between_points == 40


def test_broken_layer_raises_so_caller_can_roll_back():
    with pytest.raises(ValidationError):
        build_params(ParamsConfig(anchor_ttl_bars=40), {"max_bars_between_points": 99})


# --- секция params_by_timeframe в AppConfig ---------------------------------


def _app(raw_by_tf: dict) -> AppConfig:
    return AppConfig.model_validate({"params_by_timeframe": raw_by_tf})


def test_params_by_timeframe_parses_keys():
    config = _app({"1D": {"anchor_ttl_bars": 60, "max_bars_between_points": 60}})
    assert Timeframe.D1 in config.params_by_timeframe


def test_broken_section_refuses_startup_with_timeframe_and_field():
    with pytest.raises(ValidationError) as exc:
        _app({"4H": {"max_bars_between_points": 999}})
    message = str(exc.value)
    assert "4H" in message
    assert "max_bars_between_points" in message


def test_unknown_key_in_section_refuses_startup():
    with pytest.raises(ValidationError) as exc:
        _app({"1D": {"no_such_param": 1}})
    assert "no_such_param" in str(exc.value)


def test_params_for_applies_yaml_overlay():
    config = _app({"1D": {"anchor_ttl_bars": 60, "max_bars_between_points": 60}})
    assert config.params_for(Timeframe.D1).anchor_ttl_bars == 60
    assert config.params_for(Timeframe.H4).anchor_ttl_bars == 40
    assert config.params_for(None).anchor_ttl_bars == 40


def test_to_signal_params_carries_new_fields():
    params = ParamsConfig(
        anchor_ttl_bars=50,
        max_bars_between_points=45,
        min_bars_between_points=6,
        extreme_alerts_enabled=True,
        wick_alerts_enabled=True,
        wick_threshold_pct=7.5,
    )
    signal_params = params.to_signal_params()
    assert signal_params.anchor_ttl_bars == 50
    assert signal_params.max_bars_between_points == 45
    assert signal_params.min_bars_between_points == 6
    assert signal_params.extreme_alerts_enabled is True
    assert signal_params.wick_threshold_pct == 7.5
