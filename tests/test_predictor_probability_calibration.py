import sys
import threading
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_engine.predictor import EnsemblePredictor


def _predictor_with_samples(scores, labels):
    predictor = EnsemblePredictor.__new__(EnsemblePredictor)
    predictor._v2_calibrator_lock = threading.Lock()
    predictor._v2_calibrator_cache = None
    predictor._load_v2_calibration_samples = lambda as_of=None: (
        np.asarray(scores, dtype=float),
        np.asarray(labels, dtype=int),
        {
            "sample_count": len(labels),
            "positive_count": int(sum(labels)),
            "negative_count": int(len(labels) - sum(labels)),
            "latest_validated_at": "2026-08-11 00:00:00",
        },
    )
    return predictor


def test_v2_platt_activates_only_after_time_split_brier_improves():
    scores = []
    labels = []
    patterns = {
        0.60: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        0.70: [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        0.80: [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        0.90: [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    }
    for _ in range(10):
        for score, block_labels in patterns.items():
            scores.extend([score] * len(block_labels))
            labels.extend(block_labels)

    predictor = _predictor_with_samples(scores, labels)
    calibrator = predictor._build_v2_calibrator()

    assert calibrator["active"] is True
    assert calibrator["method"] == "v2_global_platt"
    assert calibrator["holdout_calibrated_brier"] < calibrator["holdout_raw_brier"]
    probability, meta = predictor._calibrate_v2_probability(0.80)
    assert probability is not None
    assert 0.0 < probability < 0.80
    assert meta["sample_count"] == 400


def test_v2_platt_falls_back_when_mature_labels_are_insufficient():
    predictor = _predictor_with_samples([0.8] * 40, [1, 0] * 20)

    probability, calibrator = predictor._calibrate_v2_probability(0.80)

    assert probability is None
    assert calibrator["active"] is False
    assert calibrator["reason"] == "insufficient_mature_labels"
