"""Shared fixtures for the test suite.

Run the whole suite with (from the repo root, inside the env that has
torch/monai/nibabel/etc. installed — e.g. `conda activate pytorch2`):

    pytest

These tests are meant to be fast and dependency-light (synthetic tensors and
tiny synthetic NIfTI volumes, not real patient data) so they can run in
seconds and be used as a pre-flight check before kicking off real training.
"""
import os

import numpy as np
import pytest
import torch

pytest.importorskip("easydict", reason="config.py needs easydict")


@pytest.fixture
def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def has_cuda():
    return torch.cuda.is_available()


@pytest.fixture
def tiny_unetr_kwargs(monkeypatch):
    """Monkeypatches every cfg.unetr field down to tiny values, on the real
    (global, singleton) cfg object, and returns a kwargs dict mirroring
    those same values.

    Two things read these dimensions differently, so both need to agree:
      - engine.build_model()/UNETR.__init__'s *default* args (e.g.
        img_shape=cfg.unetr.img_shape) are read fresh at call time for
        build_model (it does `UNETR(img_shape=cfg.unetr.img_shape, ...)`
        explicitly) — but UNETR's bare *default* parameter values are bound
        once at class-definition/import time, so constructing UNETR()
        without arguments after this fixture runs would NOT pick up the
        patched cfg. That's why callers should pass this fixture's dict
        into UNETR(**kwargs) explicitly rather than relying on defaults.
      - num_layers, extract_layers, mlp_dim aren't constructor parameters at
        all — TransformerBlock/UNETR read cfg.unetr.num_layers etc. directly
        inside __init__, fresh on every call, so patching cfg here is both
        necessary and sufficient for those three.
    """
    from config import cfg

    small = dict(
        img_shape=(32, 32, 16),
        input_dim=4,
        output_dim=3,
        embed_dim=32,
        # NOTE: patch_size looks like a free hyperparameter but isn't — the
        # decoder's skip-connection levels only line up (z3_out ends up at
        # the same resolution as z0) when patch_size == 16, because the
        # decoder3/6/9 branches plus their *_upsampler's extra deconv always
        # add up to exactly 16x total upsampling from the patch grid back to
        # img_shape, regardless of img_shape itself. Changing this breaks
        # UNETR.forward with a shape-mismatch RuntimeError at decoder0_header.
        patch_size=16,
        num_heads=4,
        dropout=0.0,
    )
    monkeypatch.setattr(cfg.unetr, "num_layers", 4, raising=True)
    monkeypatch.setattr(cfg.unetr, "extract_layers", [1, 2, 3, 4], raising=True)
    monkeypatch.setattr(cfg.unetr, "mlp_dim", 64, raising=True)
    for key, value in small.items():
        monkeypatch.setattr(cfg.unetr, key, value, raising=True)

    return small


@pytest.fixture
def tiny_unetr(tiny_unetr_kwargs):
    from models.unetr import UNETR
    return UNETR(**tiny_unetr_kwargs)


@pytest.fixture
def make_nii_patient(tmp_path):
    """Factory fixture: writes one synthetic BraTS-style patient folder
    (flair/t1/t1ce/t2 + seg, all tiny .nii volumes) and returns its directory
    path, matching the exact layout utils/dataloader.py:load_datalist expects.
    """
    nib = pytest.importorskip("nibabel")

    def _make(patient_id="TestPatient_1", shape=(8, 9, 6), seed=0, root=None):
        rng = np.random.default_rng(seed)
        base = tmp_path if root is None else root
        patient_dir = os.path.join(str(base), patient_id)
        os.makedirs(patient_dir, exist_ok=True)
        affine = np.eye(4)

        for modality in ("flair", "t1", "t1ce", "t2"):
            vol = (rng.random(shape) * 500).astype(np.float32)
            nib.save(nib.Nifti1Image(vol, affine),
                      os.path.join(patient_dir, f"{patient_id}_{modality}.nii"))

        seg = rng.choice([0, 1, 2, 4], size=shape,
                          p=[0.85, 0.05, 0.07, 0.03]).astype(np.float32)
        nib.save(nib.Nifti1Image(seg, affine),
                  os.path.join(patient_dir, f"{patient_id}_seg.nii"))

        return patient_dir

    return _make


@pytest.fixture
def make_brats_root(make_nii_patient, tmp_path):
    """Factory fixture: builds a root_dir containing N synthetic patients,
    for exercising BratsDataset's train/val/test split logic end to end."""

    def _make(num_patients=10, shape=(8, 9, 6)):
        root = tmp_path / "brats_root"
        os.makedirs(str(root), exist_ok=True)
        for i in range(num_patients):
            make_nii_patient(patient_id=f"Patient_{i}", shape=shape, seed=i, root=str(root))
        return str(root)

    return _make
