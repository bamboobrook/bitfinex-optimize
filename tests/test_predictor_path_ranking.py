import unittest

from test_predictor_rank6 import (
    EnsemblePredictor,
    _make_prediction,
)


class PredictorPathRankingTest(unittest.TestCase):
    def _make_predictor(self):
        predictor = EnsemblePredictor.__new__(EnsemblePredictor)
        predictor.policy = {}
        predictor.policy_version = "test-policy"
        predictor.db_path = ":memory:"
        predictor.order_manager = None
        predictor._funding_book_cache = {}
        predictor._stale_issues = []
        predictor._get_pending_order_pressure = lambda currency, period: 0.0
        predictor._estimate_rank6_reference_rate = (
            lambda preds: next(
                (
                    float(pred.get("predicted_rate", 0.0) or 0.0)
                    for pred in preds
                    if pred.get("currency") == "fUSD" and int(pred.get("period", 0) or 0) == 2
                ),
                5.0,
            )
        )
        return predictor

    def test_fusd_120d_with_real_yield_beats_fake_high_premium_fusd_20d(self):
        predictor = self._make_predictor()
        preds = [
            _make_prediction("fUSD", 120, 12.8, exec_prob=0.72),
            _make_prediction("fUSD", 20, 14.5, exec_prob=0.21),
            _make_prediction("fUSD", 2, 5.0, exec_prob=0.55),
        ]
        preds[0]["current_rate"] = 11.9
        preds[0]["execution_rate_7d"] = 0.68
        preds[1]["current_rate"] = 4.0
        preds[1]["execution_rate_7d"] = 0.14
        preds[1]["market_follow_error"] = 10.5
        preds[1]["avg_rate_gap_failed"] = 7.5

        ranked = predictor._apply_path_ranking(
            preds,
            market_liquidity={
                "fUSD": {
                    "score": 62.0,
                    "fast_score": 0.58,
                    "fillability_signal": 0.56,
                    "volume_ratio_24h": 0.82,
                }
            },
            fusd_2d_pred=None,
        )

        self.assertEqual((ranked[0]["currency"], ranked[0]["period"]), ("fUSD", 120))
        self.assertGreater(ranked[0]["final_rank_score"], ranked[1]["final_rank_score"])
        self.assertGreater(ranked[0]["path_value_score"], ranked[1]["path_value_score"])
        self.assertLess(ranked[1]["safety_multiplier"], ranked[0]["safety_multiplier"])

    def test_fust_needs_burst_and_strong_liquidity_to_beat_fusd_by_default(self):
        predictor = self._make_predictor()
        fusd = _make_prediction("fUSD", 30, 9.2, exec_prob=0.58)
        fusd["current_rate"] = 8.7
        fusd["execution_rate_7d"] = 0.56

        fust = _make_prediction("fUST", 30, 10.6, exec_prob=0.64)
        fust["current_rate"] = 8.2
        fust["execution_rate_7d"] = 0.49

        default_ranked = predictor._apply_path_ranking(
            [fusd.copy(), fust.copy()],
            market_liquidity={
                "fUSD": {
                    "score": 64.0,
                    "fast_score": 0.62,
                    "fillability_signal": 0.60,
                    "volume_ratio_24h": 0.88,
                },
                "fUST": {
                    "score": 45.0,
                    "fast_score": 0.43,
                    "fillability_signal": 0.41,
                    "volume_ratio_24h": 0.52,
                },
            },
            fusd_2d_pred=None,
        )

        self.assertEqual((default_ranked[0]["currency"], default_ranked[0]["period"]), ("fUSD", 30))
        self.assertLess(default_ranked[1]["currency_regime_multiplier"], 1.0)

        burst_fust = fust.copy()
        burst_fust["predicted_rate"] = 11.4
        burst_fust["execution_probability"] = 0.81
        burst_fust["calibrated_execution_prob"] = 0.81
        burst_fust["execution_rate_7d"] = 0.79

        burst_ranked = predictor._apply_path_ranking(
            [fusd.copy(), burst_fust],
            market_liquidity={
                "fUSD": {
                    "score": 64.0,
                    "fast_score": 0.62,
                    "fillability_signal": 0.60,
                    "volume_ratio_24h": 0.88,
                },
                "fUST": {
                    "score": 78.0,
                    "fast_score": 0.86,
                    "fillability_signal": 0.84,
                    "volume_ratio_24h": 1.35,
                },
            },
            fusd_2d_pred=None,
        )

        self.assertEqual((burst_ranked[0]["currency"], burst_ranked[0]["period"]), ("fUST", 30))
        self.assertGreaterEqual(burst_ranked[0]["currency_regime_multiplier"], 1.0)

    def test_fust_guarded_high_frr_fallback_still_does_not_beat_fusd_default(self):
        predictor = self._make_predictor()
        predictor._estimate_frr_proxy_rate = lambda currency, current_rate: 12.0 if currency == "fUST" else 7.3

        fusd = _make_prediction("fUSD", 60, 9.7908, exec_prob=0.66)
        fusd["current_rate"] = 17.113
        fusd["execution_rate_7d"] = 0.28
        fusd["avg_rate_gap_failed"] = 7.7

        fust = _make_prediction("fUST", 14, 9.6810, exec_prob=0.69)
        fust["current_rate"] = 9.49
        fust["execution_rate_7d"] = 0.50
        fust["avg_rate_gap_failed"] = 2.8

        ranked = predictor._apply_path_ranking(
            [fusd, fust],
            market_liquidity={
                "fUSD": {
                    "score": 56.8,
                    "fast_score": 0.59,
                    "fillability_signal": 0.96,
                    "volume_ratio_24h": 0.41,
                },
                "fUST": {
                    "score": 55.5,
                    "fast_score": 0.53,
                    "fillability_signal": 1.00,
                    "volume_ratio_24h": 2.44,
                },
            },
            fusd_2d_pred={"currency": "fUSD", "period": 2, "predicted_rate": 3.2574},
        )

        self.assertEqual((ranked[0]["currency"], ranked[0]["period"]), ("fUSD", 60))
        self.assertEqual(ranked[1]["currency_regime_state"], "fust_guarded")
        self.assertLess(ranked[1]["currency_regime_multiplier"], 0.90)

    def test_live_like_fust_cluster_does_not_fill_rank2_to_rank4_by_default(self):
        predictor = self._make_predictor()
        predictor._estimate_frr_proxy_rate = lambda currency, current_rate: 11.972 if currency == "fUST" else 7.3

        live_like_preds = [
            _make_prediction("fUSD", 60, 9.7908, exec_prob=0.6577),
            _make_prediction("fUST", 14, 9.6810, exec_prob=0.6936),
            _make_prediction("fUST", 15, 9.5628, exec_prob=0.6081),
            _make_prediction("fUST", 30, 9.2152, exec_prob=0.6594),
            _make_prediction("fUSD", 30, 7.2708, exec_prob=0.5834),
            _make_prediction("fUSD", 120, 7.6048, exec_prob=0.7275),
            _make_prediction("fUSD", 2, 3.2574, exec_prob=0.6655),
        ]
        overrides = {
            ("fUSD", 60): {"current_rate": 17.1130, "execution_rate_7d": 0.2911, "avg_rate_gap_failed": 7.7617},
            ("fUST", 14): {"current_rate": 9.4900, "execution_rate_7d": 0.5000, "avg_rate_gap_failed": 2.8025},
            ("fUST", 15): {"current_rate": 9.9645, "execution_rate_7d": 0.3750, "avg_rate_gap_failed": 3.8325},
            ("fUST", 30): {"current_rate": 8.7885, "execution_rate_7d": 0.3659, "avg_rate_gap_failed": 4.4454},
            ("fUSD", 30): {"current_rate": 5.5564, "execution_rate_7d": 0.5000, "avg_rate_gap_failed": 4.2070},
            ("fUSD", 120): {"current_rate": 7.3000, "execution_rate_7d": 0.4375, "avg_rate_gap_failed": 7.8342},
            ("fUSD", 2): {"current_rate": 4.9640, "execution_rate_7d": 0.3431, "avg_rate_gap_failed": 0.0},
        }
        for pred in live_like_preds:
            pred.update(overrides[(pred["currency"], pred["period"])])

        ranked = predictor._apply_path_ranking(
            live_like_preds,
            market_liquidity={
                "fUSD": {
                    "score": 56.6,
                    "fast_score": 0.579,
                    "fillability_signal": 0.967,
                    "volume_ratio_24h": 0.414,
                },
                "fUST": {
                    "score": 55.4,
                    "fast_score": 0.534,
                    "fillability_signal": 1.0,
                    "volume_ratio_24h": 2.395,
                },
            },
            fusd_2d_pred={"currency": "fUSD", "period": 2, "predicted_rate": 3.2574},
        )

        top4 = [(pred["currency"], pred["period"]) for pred in ranked[:4]]
        top4_fusd_count = sum(1 for currency, _ in top4 if currency == "fUSD")

        self.assertEqual(top4[0], ("fUSD", 60))
        self.assertGreaterEqual(top4_fusd_count, 2)


if __name__ == "__main__":
    unittest.main()
