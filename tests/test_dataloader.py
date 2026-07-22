"""Tests for utils/dataloader.py — dataset discovery, per-volume
normalization, and the train/val/test split logic. A bug here (e.g. an
off-by-one in the split, or a patient silently missing a modality file) is
exactly the kind of thing that wastes hours: training runs to completion on
the wrong data, or crashes on batch N when a bad file finally gets sampled.
"""
import numpy as np
import pytest

pytest.importorskip("monai")

from utils.dataloader import load_datalist, apply_clahe_to_volume, zscore_normalize, BratsDataset


# ---------------------------------------------------------------------------
# load_datalist
# ---------------------------------------------------------------------------

def test_load_datalist_finds_complete_patients(make_brats_root):
    root = make_brats_root(num_patients=5)
    datalist = load_datalist(root)
    assert len(datalist) == 5
    for entry in datalist:
        assert len(entry["image"]) == 4
        assert entry["label"].endswith("_seg.nii")


def test_load_datalist_skips_patient_missing_a_modality(make_nii_patient, tmp_path):
    import os
    root = tmp_path / "root"
    os.makedirs(str(root), exist_ok=True)
    complete_dir = make_nii_patient(patient_id="Complete_1", root=str(root))
    incomplete_dir = make_nii_patient(patient_id="Incomplete_1", root=str(root))
    # remove one modality file to simulate an incomplete/corrupted download
    os.remove(os.path.join(incomplete_dir, "Incomplete_1_t2.nii"))

    datalist = load_datalist(str(root))

    patient_ids = {os.path.basename(e["label"]).replace("_seg.nii", "") for e in datalist}
    assert patient_ids == {"Complete_1"}


def test_load_datalist_ignores_non_directory_entries(make_brats_root, tmp_path):
    root = make_brats_root(num_patients=2)
    with open(f"{root}/stray_file.txt", "w") as f:
        f.write("not a patient folder")
    datalist = load_datalist(root)
    assert len(datalist) == 2


def test_load_datalist_empty_dir_returns_empty_list(tmp_path):
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    assert load_datalist(str(empty_root)) == []


# ---------------------------------------------------------------------------
# apply_clahe_to_volume
# ---------------------------------------------------------------------------

def test_apply_clahe_preserves_shape_and_dtype():
    volume = np.random.rand(6, 20, 20).astype(np.float32) * 500
    out = apply_clahe_to_volume(volume)
    assert out.shape == volume.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_apply_clahe_keeps_background_exactly_zero():
    volume = np.zeros((4, 16, 16), dtype=np.float32)
    volume[:, 4:12, 4:12] = np.random.rand(4, 8, 8).astype(np.float32) * 300 + 50
    out = apply_clahe_to_volume(volume)
    background_mask = volume <= 1e-5
    assert np.all(out[background_mask] == 0.0)


def test_apply_clahe_constant_volume_returns_as_is():
    volume = np.full((3, 10, 10), 42.0, dtype=np.float32)
    out = apply_clahe_to_volume(volume)
    np.testing.assert_array_equal(out, volume.astype(np.float32))


def test_apply_clahe_mostly_empty_slice_skips_clahe_but_still_normalizes():
    # brain_mask covers 2% of the slice -> under the 5% per-slice CLAHE
    # threshold, so CLAHE itself is skipped. Note the volume-wide p1/p99
    # clip + min-max normalization runs BEFORE that per-slice check, so the
    # "skip" path preserves the *normalized* value (0 or 1 here), not the
    # raw input value (100.0) — 8/400=2% is enough to survive the p99 clip
    # (which would otherwise crush a rarer outlier back to ~0).
    volume = np.zeros((1, 20, 20), dtype=np.float32)
    volume.flat[:8] = 100.0
    out = apply_clahe_to_volume(volume)
    assert out.flat[0] == pytest.approx(1.0)
    assert out.flat[8] == pytest.approx(0.0)
    assert np.all(out[0][1:, 1:] == 0.0)


def test_apply_clahe_all_nonfinite_returns_zeros_like():
    volume = np.full((2, 5, 5), np.nan, dtype=np.float32)
    out = apply_clahe_to_volume(volume)
    assert out.shape == volume.shape


# ---------------------------------------------------------------------------
# zscore_normalize
# ---------------------------------------------------------------------------

def test_zscore_normalize_background_stays_zero():
    volume = np.zeros((10, 10), dtype=np.float32)
    volume[3:7, 3:7] = 10.0
    out = zscore_normalize(volume)
    assert np.all(out[volume == 0] == 0.0)


def test_zscore_normalize_foreground_has_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    volume = np.zeros((50, 50), dtype=np.float32)
    volume[10:40, 10:40] = rng.normal(loc=100, scale=15, size=(30, 30)).astype(np.float32)
    out = zscore_normalize(volume)
    fg = out[volume > 1e-5]
    assert fg.mean() == pytest.approx(0.0, abs=0.1)
    assert fg.std() == pytest.approx(1.0, abs=0.2)


def test_zscore_normalize_clamps_to_plus_minus_5():
    volume = np.zeros((10, 10), dtype=np.float32)
    volume[0, 0] = 1.0
    volume[1, 1] = 1e6  # extreme outlier
    out = zscore_normalize(volume)
    assert out.max() <= 5.0
    assert out.min() >= -5.0


def test_zscore_normalize_all_background_returns_zeros():
    volume = np.zeros((5, 5), dtype=np.float32)
    out = zscore_normalize(volume)
    assert np.all(out == 0.0)


def test_zscore_normalize_handles_nan_and_inf_input():
    volume = np.array([[np.nan, np.inf, -np.inf, 5.0]], dtype=np.float32)
    out = zscore_normalize(volume)
    assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# BratsDataset split logic
# ---------------------------------------------------------------------------

def test_brats_dataset_raises_on_missing_root_dir(tmp_path):
    with pytest.raises(RuntimeError):
        BratsDataset(root_dir=str(tmp_path / "does_not_exist"), section="training", cache_rate=0.0)


def test_brats_dataset_splits_are_disjoint_and_cover_all_patients(make_brats_root):
    root = make_brats_root(num_patients=20)
    common = dict(root_dir=root, val_frac=0.15, test_frac=0.10, seed=0, cache_rate=0.0)

    train_ds = BratsDataset(section="training", **common)
    val_ds = BratsDataset(section="validation", **common)
    test_ds = BratsDataset(section="test", **common)

    train_idx, val_idx, test_idx = (set(train_ds.get_indices().tolist()),
                                     set(val_ds.get_indices().tolist()),
                                     set(test_ds.get_indices().tolist()))

    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)
    assert train_idx | val_idx | test_idx == set(range(20))


def test_brats_dataset_split_sizes_match_configured_fractions(make_brats_root):
    root = make_brats_root(num_patients=20)
    common = dict(root_dir=root, val_frac=0.15, test_frac=0.10, seed=0, cache_rate=0.0)

    val_ds = BratsDataset(section="validation", **common)
    test_ds = BratsDataset(section="test", **common)
    train_ds = BratsDataset(section="training", **common)

    assert len(val_ds) == int(20 * 0.15)
    assert len(test_ds) == int(20 * 0.10)
    assert len(train_ds) == 20 - len(val_ds) - len(test_ds)


def test_brats_dataset_split_is_deterministic_given_same_seed(make_brats_root):
    root = make_brats_root(num_patients=20)
    common = dict(root_dir=root, val_frac=0.15, test_frac=0.10, seed=42, cache_rate=0.0)

    train_a = BratsDataset(section="training", **common).get_indices()
    train_b = BratsDataset(section="training", **common).get_indices()

    np.testing.assert_array_equal(train_a, train_b)


def test_brats_dataset_split_changes_with_different_seed(make_brats_root):
    root = make_brats_root(num_patients=20)
    idx_seed0 = BratsDataset(root_dir=root, section="training", val_frac=0.15, test_frac=0.10,
                              seed=0, cache_rate=0.0).get_indices()
    idx_seed1 = BratsDataset(root_dir=root, section="training", val_frac=0.15, test_frac=0.10,
                              seed=1, cache_rate=0.0).get_indices()
    assert not np.array_equal(idx_seed0, idx_seed1)
