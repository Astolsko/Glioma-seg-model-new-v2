import os
import sys
from typing import List, Dict, Union, Sequence, Callable

import numpy as np
from monai.data import CacheDataset, DataLoader, list_data_collate
from monai.transforms import (
    MapTransform,
    Randomizable,
)


def load_datalist(
    root_dir: str
) -> List[Dict]:
    """
    Load image/label paths of dataset
    """
    
    datalist = []
    for data in os.listdir(root_dir):
        data_dir_path = os.path.join(root_dir, data)
        if os.path.isdir(data_dir_path):
            model_scans = ["flair", "t1", "t1ce", "t2"]
            image_paths = [os.path.join(data_dir_path, f"{data}_{model}.nii") for model in model_scans]
            label_path = os.path.join(data_dir_path, f"{data}_seg.nii")

            if (all(os.path.exists(path) for path in [*image_paths, label_path])):
                datalist.append({
                    "image": image_paths,
                    "label": label_path
                })

    return datalist


def zscore_normalize(volume: np.ndarray) -> np.ndarray:
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)

    # brain mask — BraTS background is stored as exactly 0.0, so a small
    # positive threshold isolates the brain and normalizes over it alone.
    mask = volume > 1e-5

    if not mask.any():
        return np.zeros_like(volume, dtype=np.float32)

    mean = volume[mask].mean()
    std = volume[mask].std()

    # always start with a zero array so background stays 0
    result = np.zeros_like(volume, dtype=np.float32)

    if std < 1e-3:
        result[mask] = (volume[mask] - mean).astype(np.float32)
    else:
        result[mask] = ((volume[mask] - mean) / std).astype(np.float32)

    # clamp to [-5, 5] to prevent transformer overflow
    result = np.clip(result, -5.0, 5.0)
    
    # force background back to exactly 0 after clamping
    result[~mask] = 0.0

    return result

class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = np.asarray(d[key])
            # squeeze any leading channel dim that may have been added upstream
            if label.ndim == 4 and label.shape[0] == 1:
                label = label.squeeze(0)
            result = []
            result.append(np.logical_or(label == 1, label == 4))
            result.append(np.logical_or(np.logical_or(label == 1, label == 4), label == 2))
            result.append(label == 4)
            d[key] = np.stack(result, axis=0).astype("float32")
        return d


class ZScoreNormalized(MapTransform):
    """Per-channel z-score normalization over the brain mask.

    CLAHE was removed: it injected banding/over-sharpening artifacts on exactly
    the fine T1ce enhancement texture ET depends on (see plan.md §3, §10 #1), so
    the pipeline is per-channel z-score only, the standard BraTS normalization.
    The former `apply_clahe_to_volume` had already been dead code; this transform
    only ever called `zscore_normalize`, so dropping CLAHE changes no behavior.
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = np.asarray(d[key])
            if img.ndim != 4:
                d[key] = img
                continue
            modalities = img.copy()
            for c in range(modalities.shape[0]):
                modalities[c] = zscore_normalize(modalities[c])
            d[key] = modalities.astype(np.float32)
        return d
    
    
class BratsDataset(Randomizable, CacheDataset):
    """
    Generate items for training, validation or test.
    """

    def __init__(
        self,
        root_dir: str,
        section: str,
        transform: Union[Sequence[Callable], Callable] = (),
        val_frac: float = 0.15,
        test_frac: float = 0.10,
        seed: int = 0,
        cache_num: int = sys.maxsize,
        cache_rate: float = 1.0,
        num_workers: int = 0,
    ) -> None:
        if not os.path.isdir(root_dir) or not os.path.exists(root_dir):
            raise RuntimeError(
                f"Cannot find dataset directory: {root_dir}."
            )

        self.section = section
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.set_random_state(seed=seed)
        self.indices: np.ndarray = np.array([])
        
        data = self._generate_data_list(root_dir)
        CacheDataset.__init__(
            self, data, transform, cache_num=cache_num, cache_rate=cache_rate, num_workers=num_workers
        )

    def get_indices(self) -> np.ndarray:
        """
        Get the indices of datalist used in this dataset.
        """
        return self.indices

    def randomize(self, data: List[int]) -> None:
        self.R.shuffle(data)

    def _generate_data_list(self, root_dir: str) -> List[Dict]:
        datalist = load_datalist(root_dir)
        return self._split_datalist(datalist)

    def _split_datalist(self, datalist: List[Dict]) -> List[Dict]:
        length = len(datalist)
        indices = np.arange(length)
        self.randomize(indices)

        val_length = int(length * self.val_frac)
        test_length = int(length * self.test_frac)
        if self.section == "training":
            self.indices = indices[val_length+test_length:]
        elif self.section == "validation":
            self.indices = indices[test_length:val_length+test_length]
        else:
            self.indices = indices[:test_length]

        return [datalist[i] for i in self.indices]


def build_dataloaders(cfg):
    """Build train/val/test BratsDatasets + DataLoaders from cfg.

    Validation and test share the exact same (crop-based) transform so the
    model always sees consistently preprocessed input across splits.
    """
    from utils.transforms import build_train_transform, build_val_transform

    train_transform = build_train_transform(cfg)
    val_transform = build_val_transform(cfg)
    print(train_transform)

    train_ds = BratsDataset(
        root_dir=cfg.paths.root_dir,
        section="training",
        transform=train_transform,
        val_frac=cfg.data.val_frac,
        test_frac=cfg.data.test_frac,
        seed=cfg.seed,
        cache_rate=cfg.data.cache_rate,
        num_workers=cfg.data.num_workers_train,
    )
    # RandCropByPosNegLabeld yields cfg.crop.num_samples crops per volume as a
    # LIST, so the train loader needs list_data_collate to flatten those into
    # the batch (each crop becomes its own training example). Passed explicitly
    # rather than relying on monai.data.DataLoader's default so it can't
    # silently regress. Val/test draw one whole volume each, no list to flatten.
    train_loader = DataLoader(train_ds, batch_size=cfg.data.batch_size_train,
                               shuffle=True, num_workers=cfg.data.num_workers_train,
                               collate_fn=list_data_collate)

    val_ds = BratsDataset(
        root_dir=cfg.paths.root_dir,
        section="validation",
        transform=val_transform,
        val_frac=cfg.data.val_frac,
        test_frac=cfg.data.test_frac,
        seed=cfg.seed,
        cache_rate=cfg.data.cache_rate,
        num_workers=cfg.data.num_workers_val,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.data.batch_size_val,
                             shuffle=False, num_workers=cfg.data.num_workers_val)

    test_ds = BratsDataset(
        root_dir=cfg.paths.root_dir,
        section="test",
        transform=val_transform,
        val_frac=cfg.data.val_frac,
        test_frac=cfg.data.test_frac,
        seed=cfg.seed,
        cache_rate=cfg.data.cache_rate,
        num_workers=cfg.data.num_workers_test,
    )
    test_loader = DataLoader(test_ds, batch_size=cfg.data.batch_size_test,
                              shuffle=False, num_workers=cfg.data.num_workers_test)

    # quick shape verification. RandCropByPosNegLabeld makes the train
    # transform return a LIST of cfg.crop.num_samples crops per volume, so
    # train_ds[0] is a list; inspect its first crop.
    sample = train_ds[0]
    if isinstance(sample, list):
        sample = sample[0]
    print("image:", sample["image"].shape)
    print("label:", sample["label"].shape)

    return {
        "train_ds": train_ds, "train_loader": train_loader,
        "val_ds": val_ds, "val_loader": val_loader,
        "test_ds": test_ds, "test_loader": test_loader,
    }