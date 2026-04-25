"""
Unit tests for pure functions in conformal_prediction_pipeline.py
and the image pre-processing helper in services/calibration_service.py.
No real files, models, or Azure calls are used — all data is synthetic numpy/cv2.
"""

import io

import cv2
import numpy as np
import pytest

from conformal_prediction_pipeline import compute_lamhat, false_negative_rate
from services.calibration_service import preprocess_image_bytes


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_cal_data(n: int = 100, n_classes: int = 3, seed: int = 0):
    """Return (probs, labels, pos_mask) with every row guaranteed positive."""
    rng = np.random.default_rng(seed)
    probs = rng.uniform(0.1, 0.9, size=(n, n_classes)).astype(np.float32)
    labels = (rng.uniform(size=(n, n_classes)) > 0.6).astype(np.float32)
    for i in range(n):
        if labels[i].sum() == 0:
            labels[i, rng.integers(n_classes)] = 1.0
    pos_mask = np.ones(n, dtype=bool)
    return probs, labels, pos_mask


def _encode_png(img: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


_PREPROC = {"width": 32, "height": 32, "pixel_mean": 128.0, "pixel_std": 64.0}


# ── false_negative_rate ───────────────────────────────────────────────────

def test_fnr_perfect_predictions():
    labels = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.float32)
    pred_set = labels.copy()
    pos_mask = np.array([True, True])
    assert false_negative_rate(pred_set, labels, pos_mask) == 0.0


def test_fnr_no_predictions():
    labels = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
    pred_set = np.zeros_like(labels)
    pos_mask = np.array([True, True])
    assert false_negative_rate(pred_set, labels, pos_mask) == 1.0


def test_fnr_partial_predictions():
    # Row 0: 2 positives, 1 predicted  → recall = 0.5
    # Row 1: 1 positive,  1 predicted  → recall = 1.0
    # Mean recall = 0.75  →  FNR = 0.25
    labels   = np.array([[1, 1, 0], [0, 1, 0]], dtype=np.float32)
    pred_set = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    pos_mask = np.array([True, True])
    assert abs(false_negative_rate(pred_set, labels, pos_mask) - 0.25) < 1e-9


# ── compute_lamhat ────────────────────────────────────────────────────────

def test_compute_lamhat_returns_float():
    probs, labels, pos_mask = _make_cal_data()
    result = compute_lamhat(probs, labels, pos_mask, alpha=0.1)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_compute_lamhat_low_alpha():
    """
    Stricter alpha (0.01) must produce a lower or equal threshold than a looser one (0.2).
    Lower threshold → more predictions → lower FNR, which is necessary to meet the tighter target.
    """
    probs, labels, pos_mask = _make_cal_data(n=150)
    lam_strict = compute_lamhat(probs, labels, pos_mask, alpha=0.01)
    lam_loose = compute_lamhat(probs, labels, pos_mask, alpha=0.2)
    assert lam_strict <= lam_loose


def test_compute_lamhat_high_alpha():
    """With alpha=0.99 the empirical FNR at the returned lambda must be <= the corrected target."""
    probs, labels, pos_mask = _make_cal_data(n=200)
    alpha = 0.99
    lam = compute_lamhat(probs, labels, pos_mask, alpha=alpha)

    n_pos = int(pos_mask.sum())
    target = ((n_pos + 1) / n_pos) * alpha - (1 / n_pos)

    pred_sets = probs >= lam
    fnr = false_negative_rate(pred_sets, labels, pos_mask)
    assert fnr <= target + 1e-6


# ── preprocess_image_bytes ────────────────────────────────────────────────

def test_preprocess_grayscale_image():
    """A grayscale PNG should come out as a (3, H, W) float32 array."""
    gray = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    out = preprocess_image_bytes(_encode_png(gray), _PREPROC)
    assert out.shape == (3, 32, 32)
    assert out.dtype == np.float32


def test_preprocess_colour_image():
    """An RGB PNG should come out as (3, H, W) with ImageNet-normalised values."""
    colour = np.zeros((64, 64, 3), dtype=np.uint8)
    colour[:, :, 0] = 50    # B channel distinct
    colour[:, :, 1] = 130   # G channel distinct
    colour[:, :, 2] = 200   # R channel distinct
    out = preprocess_image_bytes(_encode_png(colour), _PREPROC)
    assert out.shape == (3, 32, 32)
    assert out.dtype == np.float32
    # ImageNet normalisation centres values around 0
    assert out.min() < 0 and out.max() > 0
