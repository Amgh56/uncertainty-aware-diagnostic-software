"""
Edge-case tests for calibration pipeline functions, zip extraction,
model loading, and image preprocessing.
All data is synthetic — no real files, models, or Azure calls.
"""

import io

import numpy as np
import pytest
import torch

from services.calibration_service import compute_lamhat, false_negative_rate
from services.calibration_service import (
    extract_dataset_from_zip_bytes,
    load_model_from_bytes,
    preprocess_image_bytes,
)
from tests.helpers import make_dataset_zip


# ── compute_lamhat edge cases ─────────────────────────────────────────────

def _single_positive_data(n_classes: int = 3):
    """Calibration set with exactly 1 positive image."""
    probs = np.array([[0.8, 0.3, 0.6]], dtype=np.float32)
    labels = np.array([[1, 0, 1]], dtype=np.float32)
    pos_mask = np.array([True])
    return probs, labels, pos_mask


def test_lamhat_single_positive_example():
    """A single-row calibration set must not crash and must return a valid float."""
    probs, labels, pos_mask = _single_positive_data()
    result = compute_lamhat(probs, labels, pos_mask, alpha=0.1)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_lamhat_all_zeros_probs():
    """All-zero probability matrix should not crash."""
    n, c = 20, 3
    probs = np.zeros((n, c), dtype=np.float32)
    labels = np.ones((n, c), dtype=np.float32)
    pos_mask = np.ones(n, dtype=bool)
    result = compute_lamhat(probs, labels, pos_mask, alpha=0.1)
    assert isinstance(result, float)


def test_lamhat_all_ones_probs():
    """All-ones probability matrix should not crash."""
    n, c = 20, 3
    probs = np.ones((n, c), dtype=np.float32)
    labels = np.ones((n, c), dtype=np.float32)
    pos_mask = np.ones(n, dtype=bool)
    result = compute_lamhat(probs, labels, pos_mask, alpha=0.1)
    assert isinstance(result, float)


# ── false_negative_rate edge cases ────────────────────────────────────────

def test_fnr_raises_on_zero_positive_row():
    """
    pos_mask marks a row as positive but the label row is all zeros.
    false_negative_rate must raise ValueError because the denominator is zero.
    """
    labels = np.array([[0, 0, 0]], dtype=np.float32)
    pred_set = np.array([[1, 1, 1]], dtype=np.float32)
    pos_mask = np.array([True])
    with pytest.raises(ValueError):
        false_negative_rate(pred_set, labels, pos_mask)


# ── extract_dataset_from_zip_bytes edge cases ─────────────────────────────

def test_extract_zip_unsafe_path():
    """A zip containing '../' in any member path must raise ValueError."""
    bad_zip = make_dataset_zip(n_images=60, unsafe_path=True)
    with pytest.raises(ValueError, match="Unsafe path"):
        extract_dataset_from_zip_bytes(bad_zip)


def test_extract_zip_too_few_images():
    """A zip with fewer than 50 images must raise ValueError."""
    small_zip = make_dataset_zip(n_images=10)
    with pytest.raises(ValueError, match="50"):
        extract_dataset_from_zip_bytes(small_zip)


def test_extract_zip_missing_labels_csv():
    """A zip without a labels.csv must raise ValueError."""
    no_csv = make_dataset_zip(n_images=60, include_labels_csv=False)
    with pytest.raises(ValueError, match="labels.csv"):
        extract_dataset_from_zip_bytes(no_csv)


# ── load_model_from_bytes edge cases ──────────────────────────────────────

def test_load_model_invalid_file():
    """Random bytes (not a valid PyTorch file) must raise ValueError."""
    garbage = b"\x00\x01\x02\x03" * 256
    with pytest.raises(ValueError):
        load_model_from_bytes(garbage)


def test_load_model_state_dict_rejected():
    """
    A state dict (OrderedDict) is not a Module and must raise ValueError.
    Developers must use torch.jit.save or torch.save(model, ...) instead.
    """
    import collections
    sd = collections.OrderedDict([("weight", torch.ones(3, 3))])
    buf = io.BytesIO()
    torch.save(sd, buf)
    with pytest.raises(ValueError):
        load_model_from_bytes(buf.getvalue())


# ── preprocess_image_bytes edge cases ─────────────────────────────────────

def test_preprocess_unreadable_image():
    """Random bytes that cannot be decoded as an image must raise ValueError."""
    garbage = b"\xff\xd8" + b"\x00" * 512  # fake JPEG header, corrupt body
    preproc = {"width": 32, "height": 32, "pixel_mean": 128.0, "pixel_std": 64.0}
    with pytest.raises(ValueError, match="[Cc]ould not decode"):
        preprocess_image_bytes(garbage, preproc)
