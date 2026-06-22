import math

from test_predictor_rank6 import EnsemblePredictor


def test_sanitize_rate_falls_back_for_nan_long_anchor():
    predictor = EnsemblePredictor.__new__(EnsemblePredictor)

    assert predictor._finite_rate(float("nan"), 8.5) == 8.5
    assert predictor._finite_rate(None, 8.5) == 8.5
    assert predictor._finite_rate(9.25, 8.5) == 9.25


def test_sanitize_rate_uses_first_finite_fallback():
    predictor = EnsemblePredictor.__new__(EnsemblePredictor)

    value = predictor._finite_rate(float("nan"), float("nan"), 7.25, 6.0)

    assert math.isclose(value, 7.25)
