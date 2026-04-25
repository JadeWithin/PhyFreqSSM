from __future__ import annotations

from __future__ import annotations

import csv
import json
import os
import random
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        import torch.backends.cudnn as cudnn

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = deterministic
        cudnn.benchmark = not deterministic
    except ImportError:
        pass


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_output_dir(config: dict[str, Any], seed: int) -> Path:
    root = Path(config["experiment"]["output_root"])
    dataset = config["dataset"]["name"]
    split_mode = config["dataset"]["split_mode"]
    exp_name = config["experiment"]["name"]
    return ensure_dir(root / dataset / split_mode / exp_name / str(seed))


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def save_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_text(text: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=False)


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= truth < num_classes and 0 <= pred < num_classes:
            matrix[truth, pred] += 1
    return matrix


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict[str, Any]:
    matrix = confusion_matrix_np(y_true, y_pred, num_classes)
    total = matrix.sum()
    correct = np.trace(matrix)
    oa = float(correct / total) if total else 0.0

    per_class_total = matrix.sum(axis=1)
    per_class_correct = np.diag(matrix)
    per_class_acc = np.divide(
        per_class_correct,
        np.maximum(per_class_total, 1),
        out=np.zeros_like(per_class_correct, dtype=np.float64),
        where=per_class_total > 0,
    )
    aa = float(per_class_acc.mean()) if len(per_class_acc) else 0.0

    pred_total = matrix.sum(axis=0)
    precision = np.divide(
        per_class_correct,
        np.maximum(pred_total, 1),
        out=np.zeros_like(per_class_correct, dtype=np.float64),
        where=pred_total > 0,
    )
    recall = per_class_acc
    f1 = np.divide(
        2 * precision * recall,
        np.maximum(precision + recall, 1e-12),
        out=np.zeros_like(precision, dtype=np.float64),
        where=(precision + recall) > 0,
    )
    macro_f1 = float(f1.mean()) if len(f1) else 0.0

    union = per_class_total + pred_total - per_class_correct
    iou = np.divide(
        per_class_correct,
        np.maximum(union, 1),
        out=np.zeros_like(per_class_correct, dtype=np.float64),
        where=union > 0,
    )
    miou = float(iou.mean()) if len(iou) else 0.0

    pe = float((per_class_total * pred_total).sum() / max(total * total, 1))
    kappa = float((oa - pe) / max(1 - pe, 1e-12))

    return {
        "OA": oa,
        "AA": aa,
        "Kappa": kappa,
        "Macro-F1": macro_f1,
        "MIoU": miou,
        "per_class_accuracy": per_class_acc.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_iou": iou.tolist(),
        "confusion_matrix": matrix.tolist(),
    }


def parameter_count(model: Any) -> int:
    return int(sum(p.numel() for p in model.parameters()))



from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler



SUPPORTED_DATASETS = {
    "IndianPines": ("IP", "indian_pines"),
    "PaviaUniversity": ("PU", "pavia_university", "university_of_pavia"),
    "Salinas": ("SA", "salinas_scene"),
    "Houston2018": ("houston", "houston2018", "houston18"),
    "WHU-Hi-LongKou": ("WHU", "LongKou", "whu_hi_longkou"),
    "WHU-Hi-HanChuan": ("HanChuan", "WHU_Hi_HanChuan", "whu_hi_hanchuan", "hanchuan"),
}

DATASET_FILE_PATTERNS = {
    "IndianPines": {
        "cube": ["Indian_pines_corrected.mat", "Indian_pines.mat"],
        "label": ["Indian_pines_gt.mat"],
    },
    "PaviaUniversity": {
        "cube": ["PaviaU.mat", "PaviaU_corrected.mat", "Pavia_University.mat"],
        "label": ["PaviaU_gt.mat", "Pavia_University_gt.mat"],
    },
    "Salinas": {
        "cube": ["Salinas_corrected.mat", "Salinas.mat"],
        "label": ["Salinas_gt.mat"],
    },
    "Houston2018": {
        "cube": ["Houston2018.mat", "Houston13.mat", "Houston.mat"],
        "label": ["Houston2018_gt.mat", "Houston13_7gt.mat", "Houston_gt.mat"],
    },
    "WHU-Hi-LongKou": {
        "cube": [
            "WHU_Hi_LongKou.mat",
            "WHU-Hi-LongKou.mat",
            "LongKou.mat",
            "longkou.mat",
        ],
        "label": [
            "WHU_Hi_LongKou_gt.mat",
            "WHU-Hi-LongKou_gt.mat",
            "LongKou_gt.mat",
            "longkou_gt.mat",
        ],
    },
    "WHU-Hi-HanChuan": {
        "cube": [
            "WHU_Hi_HanChuan.mat",
            "WHU-Hi-HanChuan.mat",
            "HanChuan.mat",
            "hanchuan.mat",
        ],
        "label": [
            "WHU_Hi_HanChuan_gt.mat",
            "WHU-Hi-HanChuan_gt.mat",
            "HanChuan_gt.mat",
            "hanchuan_gt.mat",
        ],
    },
}


def canonical_dataset_name(name: str) -> str:
    normalized = name.lower().replace(" ", "").replace("_", "").replace("-", "")
    for canonical, aliases in SUPPORTED_DATASETS.items():
        for candidate in (canonical,) + aliases:
            key = candidate.lower().replace(" ", "").replace("_", "").replace("-", "")
            if normalized == key:
                return canonical
    raise KeyError(f"Unsupported dataset: {name}")


def spectral_smoothing(cube: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    if kernel_size <= 1:
        return cube
    pad = kernel_size // 2
    padded = np.pad(cube, ((0, 0), (0, 0), (pad, pad)), mode="reflect")
    windows = [padded[:, :, i : i + cube.shape[2]] for i in range(kernel_size)]
    return np.mean(np.stack(windows, axis=0), axis=0)


def fit_standardize(cube: np.ndarray, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pixels = cube[coords[:, 0], coords[:, 1]]
    mean = pixels.mean(axis=0)
    std = pixels.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def apply_standardize(cube: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (cube - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)


def fit_pca(cube: np.ndarray, coords: np.ndarray, out_dim: int) -> tuple[np.ndarray, np.ndarray]:
    pixels = cube[coords[:, 0], coords[:, 1]].astype(np.float64)
    mean = pixels.mean(axis=0, keepdims=True)
    centered = pixels - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:out_dim]
    return components.astype(np.float32), mean.squeeze(0).astype(np.float32)


def apply_pca(cube: np.ndarray, components: np.ndarray, mean: np.ndarray) -> np.ndarray:
    h, w, c = cube.shape
    flat = cube.reshape(-1, c).astype(np.float32)
    transformed = (flat - mean.reshape(1, -1)) @ components.T
    return transformed.reshape(h, w, components.shape[0])


def prepare_cube(
    cube: np.ndarray,
    train_coords: np.ndarray,
    standardize: bool = True,
    pca_bands: int | None = None,
    use_smoothing: bool = False,
    smoothing_kernel: int = 5,
) -> np.ndarray:
    cube_proc = cube.astype(np.float32)
    if use_smoothing:
        cube_proc = spectral_smoothing(cube_proc, smoothing_kernel)
    if standardize:
        mean, std = fit_standardize(cube_proc, train_coords)
        cube_proc = apply_standardize(cube_proc, mean, std)
    if pca_bands is not None and pca_bands < cube_proc.shape[-1]:
        components, mean = fit_pca(cube_proc, train_coords, pca_bands)
        cube_proc = apply_pca(cube_proc, components, mean)
    return cube_proc


@dataclass
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    dropped_by_guard_band: np.ndarray | None = None
    guard_band: int = 0


def _stack_coords(parts: list[np.ndarray]) -> np.ndarray:
    valid_parts = []
    for part in parts:
        arr = np.asarray(part, dtype=np.int64)
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        valid_parts.append(arr)
    if not valid_parts:
        return np.empty((0, 2), dtype=np.int64)
    return np.concatenate(valid_parts, axis=0).astype(np.int64, copy=False)


def _group_coords_by_class(labels: np.ndarray) -> dict[int, np.ndarray]:
    grouped: dict[int, list[list[int]]] = {}
    for row, col in np.argwhere(labels > 0):
        grouped.setdefault(int(labels[row, col]), []).append([int(row), int(col)])
    return {cls: np.asarray(coords, dtype=np.int64) for cls, coords in grouped.items()}


def random_ratio_split(labels: np.ndarray, train_ratio: float, val_ratio: float, seed: int) -> SplitIndices:
    rng = np.random.default_rng(seed)
    grouped = _group_coords_by_class(labels)
    train_parts, val_parts, test_parts = [], [], []
    for coords in grouped.values():
        order = rng.permutation(len(coords))
        coords = coords[order]
        n_train = max(1, int(round(len(coords) * train_ratio)))
        n_val = max(1, int(round(len(coords) * val_ratio)))
        n_val = min(n_val, max(len(coords) - n_train - 1, 0))
        train_parts.append(coords[:n_train])
        val_parts.append(coords[n_train : n_train + n_val])
        test_parts.append(coords[n_train + n_val :])
    return SplitIndices(_stack_coords(train_parts), _stack_coords(val_parts), _stack_coords(test_parts))


def fixed_shot_split(labels: np.ndarray, shots_per_class: int, val_shots_per_class: int, seed: int) -> SplitIndices:
    rng = np.random.default_rng(seed)
    grouped = _group_coords_by_class(labels)
    train_parts, val_parts, test_parts = [], [], []
    for coords in grouped.values():
        order = rng.permutation(len(coords))
        coords = coords[order]
        n_train = min(shots_per_class, len(coords))
        n_val = min(val_shots_per_class, max(len(coords) - n_train, 0))
        train_parts.append(coords[:n_train])
        val_parts.append(coords[n_train : n_train + n_val])
        test_parts.append(coords[n_train + n_val :])
    return SplitIndices(_stack_coords(train_parts), _stack_coords(val_parts), _stack_coords(test_parts))


def _coord_near_shared_border(row: int, col: int, block_size: int, delta_row: int, delta_col: int, guard_band: int) -> bool:
    row_offset = row % block_size
    col_offset = col % block_size
    if delta_row < 0:
        near_row = row_offset < guard_band
    elif delta_row > 0:
        near_row = (block_size - 1 - row_offset) < guard_band
    else:
        near_row = True
    if delta_col < 0:
        near_col = col_offset < guard_band
    elif delta_col > 0:
        near_col = (block_size - 1 - col_offset) < guard_band
    else:
        near_col = True
    return near_row and near_col


def spatial_block_split(labels: np.ndarray, train_ratio: float, val_ratio: float, block_size: int, seed: int, guard_band: int = 0) -> SplitIndices:
    rng = np.random.default_rng(seed)
    valid_coords = np.argwhere(labels > 0)
    block_coords = np.stack((valid_coords[:, 0] // block_size, valid_coords[:, 1] // block_size), axis=1)
    unique_blocks = np.unique(block_coords, axis=0)
    unique_blocks = unique_blocks[rng.permutation(len(unique_blocks))]
    n_train_blocks = max(1, int(round(len(unique_blocks) * train_ratio)))
    max_val_blocks = max(len(unique_blocks) - n_train_blocks - 1, 0)
    requested_val_blocks = max(1, int(round(len(unique_blocks) * val_ratio)))
    n_val_blocks = min(requested_val_blocks, max_val_blocks) if len(unique_blocks) > n_train_blocks else 0
    train_blocks = {tuple(map(int, item)) for item in unique_blocks[:n_train_blocks].tolist()}
    val_blocks = {tuple(map(int, item)) for item in unique_blocks[n_train_blocks : n_train_blocks + n_val_blocks].tolist()}
    block_to_split: dict[tuple[int, int], str] = {}
    for block in train_blocks:
        block_to_split[block] = "train"
    for block in val_blocks:
        block_to_split[block] = "val"
    for block in map(tuple, unique_blocks.tolist()):
        block = (int(block[0]), int(block[1]))
        block_to_split.setdefault(block, "test")

    train_coords, val_coords, test_coords, dropped_coords = [], [], [], []
    for coord, block_coord in zip(valid_coords, block_coords):
        row, col = int(coord[0]), int(coord[1])
        block_key = (int(block_coord[0]), int(block_coord[1]))
        split_name = block_to_split[block_key]
        if guard_band > 0:
            should_drop = False
            for delta_row in (-1, 0, 1):
                for delta_col in (-1, 0, 1):
                    if delta_row == 0 and delta_col == 0:
                        continue
                    neighbor_key = (block_key[0] + delta_row, block_key[1] + delta_col)
                    neighbor_split = block_to_split.get(neighbor_key)
                    if neighbor_split is None or neighbor_split == split_name:
                        continue
                    if _coord_near_shared_border(row, col, block_size, delta_row, delta_col, guard_band):
                        should_drop = True
                        break
                if should_drop:
                    break
            if should_drop:
                dropped_coords.append(coord.astype(np.int64))
                continue
        item = coord.astype(np.int64)
        if split_name == "train":
            train_coords.append(item)
        elif split_name == "val":
            val_coords.append(item)
        else:
            test_coords.append(item)
    return SplitIndices(
        _stack_coords(train_coords),
        _stack_coords(val_coords),
        _stack_coords(test_coords),
        dropped_by_guard_band=_stack_coords(dropped_coords),
        guard_band=int(guard_band),
    )


def _select_cube(mat_dict: dict[str, Any]) -> np.ndarray:
    cube = None
    for key, value in mat_dict.items():
        if key.startswith("__") or not isinstance(value, np.ndarray):
            continue
        if value.ndim == 3:
            cube = value
            break
    if cube is None:
        raise ValueError("Unable to infer HSI cube from .mat file.")
    return cube.astype(np.float32)


def _select_labels(mat_dict: dict[str, Any]) -> np.ndarray:
    labels = None
    for key, value in mat_dict.items():
        if key.startswith("__") or not isinstance(value, np.ndarray):
            continue
        if value.ndim == 2 and np.issubdtype(value.dtype, np.integer):
            labels = value
            break
    if labels is None:
        raise ValueError("Unable to infer HSI label map from .mat file.")
    return labels.astype(np.int64)


def _select_cube_and_labels(mat_dict: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return _select_cube(mat_dict), _select_labels(mat_dict)


def _find_dataset_file(root: Path, candidates: list[str]) -> Path | None:
    for candidate in candidates:
        direct = root / candidate
        if direct.exists():
            return direct
    for candidate in candidates:
        matches = list(root.rglob(candidate))
        if matches:
            return matches[0]
    return None


def _load_hsi_scene_from_patterns(dataset_name: str, root: Path) -> tuple[np.ndarray, np.ndarray] | None:
    patterns = DATASET_FILE_PATTERNS.get(dataset_name)
    if patterns is None:
        return None

    cube_file = _find_dataset_file(root, patterns["cube"])
    label_file = _find_dataset_file(root, patterns["label"])
    if cube_file is None or label_file is None:
        return None

    cube_dict = loadmat(cube_file)
    label_dict = loadmat(label_file)
    cube = _select_cube(cube_dict)
    labels = _select_labels(label_dict)
    return cube, labels


def _resolve_data_root(root: str | Path) -> Path:
    root = Path(root)
    if root.is_absolute() and root.exists():
        return root

    repo_root = Path(__file__).resolve().parents[1]
    package_root = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / root,
        repo_root / root,
        package_root / root,
    ]
    if str(root).replace("\\", "/") == "data":
        candidates.extend(
            [
                repo_root / "data",
                package_root / "data",
            ]
        )

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return candidates[0]


def load_hsi_scene(root: str | Path, dataset_name: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    root = _resolve_data_root(root)
    if dataset_name is not None:
        resolved = _load_hsi_scene_from_patterns(dataset_name, root)
        if resolved is not None:
            return resolved

    mat_files = sorted(root.glob("*.mat"))
    if not mat_files:
        mat_files = sorted(root.rglob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found under resolved root: {root}")
    cube = None
    labels = None
    for mat_file in mat_files:
        data = loadmat(mat_file)
        try:
            file_cube, file_labels = _select_cube_and_labels(data)
        except ValueError:
            continue
        if cube is None:
            cube = file_cube
        if labels is None:
            labels = file_labels
        if cube is not None and labels is not None:
            break
    if cube is None or labels is None:
        raise ValueError(f"Could not load scene arrays from {root}")
    return cube, labels


class HSIPatchDataset(Dataset):
    def __init__(
        self,
        cube: np.ndarray,
        labels: np.ndarray,
        coords: np.ndarray,
        patch_size: int,
        train: bool = False,
        augmentation: dict[str, bool] | None = None,
        seed: int = 0,
    ):
        super().__init__()
        self.cube = cube
        self.labels = labels
        self.coords = coords.astype(np.int64)
        self.pad = patch_size // 2
        self.train = train
        self.augmentation = augmentation or {}
        self.base_seed = int(seed)
        self.epoch = 0
        self.padded = np.pad(cube, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode="reflect")

    def __len__(self) -> int:
        return len(self.coords)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng_for_index(self, index: int) -> np.random.Generator:
        mixed = (
            (np.uint64(self.base_seed + 1) * np.uint64(1000003))
            + (np.uint64(self.epoch + 1) * np.uint64(9176))
            + (np.uint64(index + 1) * np.uint64(1315423911))
        ) % np.uint64(2**32 - 1)
        return np.random.default_rng(int(mixed))

    def _augment_patch(self, patch: np.ndarray, index: int) -> np.ndarray:
        if not self.train or not self.augmentation.get("enabled", False):
            return patch
        rng = self._rng_for_index(index)
        if self.augmentation.get("rot90", False):
            patch = np.rot90(patch, k=int(rng.integers(0, 4)), axes=(1, 2))
        if self.augmentation.get("hflip", False) and float(rng.random()) < 0.5:
            patch = np.flip(patch, axis=2)
        if self.augmentation.get("vflip", False) and float(rng.random()) < 0.5:
            patch = np.flip(patch, axis=1)
        return np.ascontiguousarray(patch, dtype=np.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row, col = self.coords[index]
        row_pad, col_pad = row + self.pad, col + self.pad
        patch = self.padded[
            row_pad - self.pad : row_pad + self.pad + 1,
            col_pad - self.pad : col_pad + self.pad + 1,
        ]
        patch = np.transpose(patch, (2, 0, 1)).astype(np.float32)
        patch = self._augment_patch(patch, index)
        return {
            "patch": patch,
            "label": int(self.labels[row, col]) - 1,
            "spectrum": self.cube[row, col].astype(np.float32),
            "coord": np.asarray([row, col], dtype=np.int64),
        }


@dataclass
class DataBundle:
    cube: np.ndarray
    labels: np.ndarray
    split: SplitIndices
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    all_loader: DataLoader
    num_classes: int
    in_channels: int


def _build_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    classes, counts = np.unique(labels, return_counts=True)
    weights = {int(cls): 1.0 / count for cls, count in zip(classes, counts)}
    sample_weights = np.asarray([weights[int(label)] for label in labels], dtype=np.float64)
    return WeightedRandomSampler(weights=sample_weights.tolist(), num_samples=len(sample_weights), replacement=True)


def _resolve_split(config: dict[str, Any], labels: np.ndarray, seed: int) -> SplitIndices:
    split_mode = config["dataset"]["split_mode"]
    if split_mode == "random_ratio":
        return random_ratio_split(labels, config["dataset"]["train_ratio"], config["dataset"]["val_ratio"], seed)
    if split_mode == "fixed_shot":
        shots = int(config["dataset"]["shots_per_class"])
        return fixed_shot_split(labels, shots, max(1, shots // 2), seed)
    if split_mode == "spatial_block":
        guard_band = int(config["dataset"].get("spatial_guard_band", 0))
        return spatial_block_split(
            labels,
            config["dataset"]["train_ratio"],
            config["dataset"]["val_ratio"],
            config["dataset"]["block_size"],
            seed,
            guard_band=guard_band,
        )
    raise ValueError(f"Unsupported split_mode: {split_mode}")


def _make_loader(dataset: HSIPatchDataset, batch_size: int, num_workers: int, shuffle: bool = False, sampler=None) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle if sampler is None else False, sampler=sampler, num_workers=num_workers, pin_memory=True, drop_last=False)


def build_datamodule(config: dict[str, Any], seed: int, output_dir: str | Path | None = None) -> DataBundle:
    config["dataset"]["name"] = canonical_dataset_name(config["dataset"]["name"])
    cube, labels = load_hsi_scene(config["dataset"]["root"], config["dataset"]["name"])
    split = _resolve_split(config, labels, seed)
    cube = prepare_cube(cube, split.train, config["preprocessing"]["standardize"], config["preprocessing"]["pca_bands"], config["preprocessing"]["spectral_smoothing"], config["preprocessing"]["smoothing_kernel"])

    patch_size = int(config["preprocessing"]["patch_size"])
    aug_config = {
        "enabled": bool(config["preprocessing"].get("train_augmentation", False)),
        "rot90": bool(config["preprocessing"].get("aug_rot90", False)),
        "hflip": bool(config["preprocessing"].get("aug_hflip", False)),
        "vflip": bool(config["preprocessing"].get("aug_vflip", False)),
    }
    train_set = HSIPatchDataset(cube, labels, split.train, patch_size, train=True, augmentation=aug_config, seed=seed)
    val_set = HSIPatchDataset(cube, labels, split.val, patch_size, train=False, augmentation=None, seed=seed)
    test_set = HSIPatchDataset(cube, labels, split.test, patch_size, train=False, augmentation=None, seed=seed)
    all_set = HSIPatchDataset(cube, labels, np.argwhere(labels > 0), patch_size, train=False, augmentation=None, seed=seed)

    sampler = None
    if config["preprocessing"]["class_balanced_sampler"]:
        train_labels = np.asarray([labels[row, col] - 1 for row, col in split.train], dtype=np.int64)
        sampler = _build_sampler(train_labels)

    batch_size = int(config["train"]["batch_size"])
    workers = int(config["dataset"]["num_workers"])
    bundle = DataBundle(
        cube=cube,
        labels=labels,
        split=split,
        train_loader=_make_loader(train_set, batch_size, workers, shuffle=not config["preprocessing"]["class_balanced_sampler"], sampler=sampler),
        val_loader=_make_loader(val_set, batch_size, workers),
        test_loader=_make_loader(test_set, batch_size, workers),
        all_loader=_make_loader(all_set, batch_size, workers),
        num_classes=int(labels.max()),
        in_channels=int(cube.shape[-1]),
    )

    split_message = (
        f"[Data] split={config['dataset']['split_mode']} "
        f"train={len(split.train)} val={len(split.val)} test={len(split.test)}"
    )
    if config["dataset"]["split_mode"] == "spatial_block":
        dropped_count = 0 if split.dropped_by_guard_band is None else int(len(split.dropped_by_guard_band))
        split_message += f" guard_band={int(split.guard_band)} dropped={dropped_count}"
    print(split_message, flush=True)

    if output_dir is not None and config["experiment"]["save_train_indices"]:
        save_json(
            {
                "train": split.train.tolist(),
                "val": split.val.tolist(),
                "test": split.test.tolist(),
                "guard_band": int(split.guard_band),
                "dropped_by_guard_band": []
                if split.dropped_by_guard_band is None
                else split.dropped_by_guard_band.tolist(),
                "dropped_by_guard_band_count": 0 if split.dropped_by_guard_band is None else int(len(split.dropped_by_guard_band)),
            },
            Path(output_dir) / "split_indices.json",
        )
    return bundle



from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from mamba_ssm.modules.mamba2 import Mamba2 as ExternalMamba2
except ImportError:
    ExternalMamba2 = None


class FallbackMamba2(nn.Module):
    def __init__(self, d_model: int, d_state: int = 64, expand: int = 2):
        super().__init__()
        hidden = max(d_model, d_state)
        self.net = nn.Sequential(nn.Linear(d_model, hidden * expand), nn.GELU(), nn.Linear(hidden * expand, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SafeMamba2(nn.Module):
    def __init__(self, d_model: int, d_state: int = 64, expand: int = 2, allow_fallback: bool = False):
        super().__init__()
        self.allow_fallback = allow_fallback
        self.primary = ExternalMamba2(d_model=d_model, d_state=d_state, expand=expand) if ExternalMamba2 is not None else None
        self.fallback = FallbackMamba2(d_model=d_model, d_state=d_state, expand=expand)
        self.use_fallback = self.primary is None and allow_fallback
        self.warned = False

    @staticmethod
    def _runtime_error_message(exc: Exception) -> str:
        details = f"{type(exc).__name__}: {exc}"
        extra = (
            "This often means the Mamba2 runtime extension loaded partially, but its Triton / "
            "causal-conv backend is missing or incompatible."
        )
        if isinstance(exc, TypeError) and "NoneType" in str(exc):
            extra = (
                "This usually means the Mamba2 package is present, but the low-level "
                "`causal-conv1d` or Triton kernel did not load correctly."
            )
        return (
            "Official Mamba2 backend failed at runtime. "
            f"{extra} Original error: {details}. "
            "Either install a complete `mamba_ssm` + `causal-conv1d` + Triton stack, "
            "or set `model.allow_mamba_fallback=true` to use the PyTorch fallback."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fallback:
            return self.fallback(x)
        if self.primary is None:
            raise RuntimeError(
                "mamba_ssm is not available, but this model requires the official Mamba2 backend. "
                "Install the full mamba_ssm stack, or set model.allow_mamba_fallback=true only for debugging."
            )
        try:
            return self.primary(x)
        except Exception as exc:
            if not self.allow_fallback:
                raise RuntimeError(self._runtime_error_message(exc)) from exc
            if not self.warned:
                warnings.warn(
                    f"Mamba2 runtime backend failed ({type(exc).__name__}: {exc}). Falling back to PyTorch implementation.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.warned = True
            self.use_fallback = True
            return self.fallback(x)


def make_mamba2(d_model: int, d_state: int, allow_fallback: bool = False) -> nn.Module:
    if ExternalMamba2 is not None:
        return SafeMamba2(d_model=d_model, d_state=d_state, expand=2, allow_fallback=allow_fallback)
    if allow_fallback:
        return FallbackMamba2(d_model=d_model, d_state=d_state, expand=2)
    raise RuntimeError(
        "mamba_ssm is not installed, but this model requires the official Mamba2 backend. "
        "Install mamba_ssm and its runtime dependencies, or set model.allow_mamba_fallback=true only for debugging."
    )


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.pointwise(self.depthwise(x))))


@dataclass
class TokenizerOutput:
    tokens: torch.Tensor
    reconstructed_map: torch.Tensor
    token_stats: dict[str, float]


class DynamicSpatialSpectralTokenizer(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, compression_ratio: float = 0.4, version: str = "full", entropy_guided_sampling: bool = True):
        super().__init__()
        self.version = version
        self.compression_ratio = compression_ratio
        self.entropy_guided_sampling = entropy_guided_sampling
        self.stem = DepthwiseSeparableConv2d(in_channels, embed_dim, kernel_size=3)
        self.context = DepthwiseSeparableConv2d(embed_dim, embed_dim, kernel_size=5)
        self.offset_head = nn.Conv2d(embed_dim, 2, kernel_size=3, padding=1)
        self.mask_head = nn.Conv2d(embed_dim, 1, kernel_size=3, padding=1)
        self.score_head = nn.Conv2d(embed_dim, 1, kernel_size=1)

    @staticmethod
    def _base_grid(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=-1)

    def _sample_features(self, features: torch.Tensor) -> torch.Tensor:
        if self.version == "no_deformable":
            return features
        b, _, h, w = features.shape
        base_grid = self._base_grid(h, w, features.device, features.dtype).unsqueeze(0).repeat(b, 1, 1, 1)
        offsets = torch.tanh(self.offset_head(features)).permute(0, 2, 3, 1) / max(h, w)
        sampled = F.grid_sample(features, base_grid + offsets, mode="bilinear", padding_mode="border", align_corners=True)
        if self.version == "full":
            sampled = sampled * torch.sigmoid(self.mask_head(features))
        return sampled

    def _select_indices(self, token_map: torch.Tensor, score_map: torch.Tensor) -> torch.Tensor:
        b, _, h, w = token_map.shape
        num_tokens = h * w
        keep_tokens = max(1, int(round(num_tokens * (1.0 - self.compression_ratio))))
        if self.version == "uniform_sampling":
            base = torch.linspace(0, num_tokens - 1, keep_tokens, device=token_map.device).long()
            return base.unsqueeze(0).repeat(b, 1)
        scores = score_map.flatten(1)
        if self.entropy_guided_sampling:
            probs = torch.softmax(token_map.flatten(2), dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-6))).sum(dim=1)
            scores = scores + entropy
        return torch.topk(scores, k=keep_tokens, dim=1).indices

    def forward(self, x: torch.Tensor) -> TokenizerOutput:
        features = self.context(self.stem(x))
        sampled = self._sample_features(features)
        scores = self.score_head(sampled)
        full_tokens = sampled.flatten(2).transpose(1, 2)
        indices = self._select_indices(sampled, scores)
        gather_index = indices.unsqueeze(-1).expand(-1, -1, full_tokens.size(-1))
        tokens = torch.gather(full_tokens, dim=1, index=gather_index)
        return TokenizerOutput(tokens=tokens, reconstructed_map=sampled, token_stats={"raw_tokens": float(full_tokens.size(1)), "compressed_tokens": float(tokens.size(1)), "compression_ratio": float(1.0 - tokens.size(1) / max(full_tokens.size(1), 1))})


class PlainTokenizer(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        self.stem = DepthwiseSeparableConv2d(in_channels, embed_dim, kernel_size=3)

    def forward(self, x: torch.Tensor) -> TokenizerOutput:
        features = self.stem(x)
        tokens = features.flatten(2).transpose(1, 2)
        return TokenizerOutput(tokens=tokens, reconstructed_map=features, token_stats={"raw_tokens": float(tokens.size(1)), "compressed_tokens": float(tokens.size(1)), "compression_ratio": 0.0})


class SpatialFrequencyStem(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.lowpass = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        low = self.lowpass(x)
        high = x - low
        fused = self.fuse(torch.cat([x, low, high], dim=1))
        stats = {
            "low_energy": float(low.detach().abs().mean().cpu()),
            "high_energy": float(high.detach().abs().mean().cpu()),
        }
        return fused, stats


class SpatialRasterTokenizer(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.stem = DepthwiseSeparableConv2d(in_channels, embed_dim, kernel_size=3)
        self.context = DepthwiseSeparableConv2d(embed_dim, embed_dim, kernel_size=5)
        self.pos_embed = nn.Parameter(torch.zeros(1, patch_size * patch_size, embed_dim))

    def forward(self, x: torch.Tensor) -> TokenizerOutput:
        features = self.context(self.stem(x))
        tokens = features.flatten(2).transpose(1, 2)
        if self.pos_embed.size(1) == tokens.size(1):
            tokens = tokens + self.pos_embed
        return TokenizerOutput(
            tokens=tokens,
            reconstructed_map=features,
            token_stats={
                "raw_tokens": float(tokens.size(1)),
                "compressed_tokens": float(tokens.size(1)),
                "compression_ratio": 0.0,
            },
        )


class SimpleSelectiveScan(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.skip = nn.Parameter(torch.ones(embed_dim))

    def forward(self, x: torch.Tensor, delta: torch.Tensor, b_term: torch.Tensor, c_term: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        state = torch.zeros(batch, dim, device=x.device, dtype=x.dtype)
        outputs = []
        for step in range(length):
            alpha = torch.sigmoid(delta[:, step])
            state = alpha * state + torch.tanh(b_term[:, step]) * x[:, step]
            outputs.append(torch.tanh(c_term[:, step]) * state + self.skip * x[:, step])
        return torch.stack(outputs, dim=1)


class FrequencyGuidedSSDBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        d_state: int = 64,
        dropout: float = 0.1,
        variant: str = "full",
        scan_mode: str = "single",
        use_frequency_conditioning: bool = True,
        use_lowfreq_for_delta_b: bool = True,
        use_highfreq_for_c: bool = True,
        use_noncausal_single_pass: bool = True,
        use_multiscan_baseline: bool = False,
        use_mamba_branch: bool = True,
        allow_mamba_fallback: bool = False,
    ):
        super().__init__()
        self.variant = variant
        self.scan_mode = "four" if use_multiscan_baseline else scan_mode
        self.use_frequency_conditioning = use_frequency_conditioning
        self.use_lowfreq_for_delta_b = use_lowfreq_for_delta_b
        self.use_highfreq_for_c = use_highfreq_for_c
        self.use_noncausal_single_pass = use_noncausal_single_pass
        self.use_mamba_branch = use_mamba_branch
        self.norm = nn.LayerNorm(embed_dim)
        self.freq_router = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Sigmoid())
        self.delta_proj = nn.Linear(embed_dim, embed_dim)
        self.b_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        self.scan = SimpleSelectiveScan(embed_dim)
        self.mamba = make_mamba2(embed_dim, d_state, allow_fallback=allow_mamba_fallback) if use_mamba_branch else None
        self.gate = nn.Linear(embed_dim * 2, embed_dim)
        self.ffn = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(embed_dim * 4, embed_dim))

    def _freq_split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.variant == "no_freq_split":
            return x, x
        low_gate = self.freq_router(x)
        low = x * low_gate
        high = x * (1.0 - low_gate)
        return low, high

    def _scan_once(self, x: torch.Tensor, delta: torch.Tensor, b_term: torch.Tensor, c_term: torch.Tensor) -> torch.Tensor:
        return self.scan(x, delta, b_term, c_term)

    def _run_scan(self, x: torch.Tensor, delta: torch.Tensor, b_term: torch.Tensor, c_term: torch.Tensor) -> torch.Tensor:
        if self.scan_mode == "single":
            return self._scan_once(x, delta, b_term, c_term)
        if self.scan_mode == "bi":
            forward = self._scan_once(x, delta, b_term, c_term)
            backward = torch.flip(self._scan_once(torch.flip(x, dims=[1]), torch.flip(delta, dims=[1]), torch.flip(b_term, dims=[1]), torch.flip(c_term, dims=[1])), dims=[1])
            return 0.5 * (forward + backward)
        forward = self._scan_once(x, delta, b_term, c_term)
        backward = torch.flip(self._scan_once(torch.flip(x, dims=[1]), torch.flip(delta, dims=[1]), torch.flip(b_term, dims=[1]), torch.flip(c_term, dims=[1])), dims=[1])
        shifted = torch.roll(self._scan_once(torch.roll(x, shifts=1, dims=1), delta, b_term, c_term), shifts=-1, dims=1)
        shifted_back = torch.roll(torch.flip(self._scan_once(torch.flip(torch.roll(x, shifts=1, dims=1), dims=[1]), torch.flip(delta, dims=[1]), torch.flip(b_term, dims=[1]), torch.flip(c_term, dims=[1])), dims=[1]), shifts=-1, dims=1)
        return 0.25 * (forward + backward + shifted + shifted_back)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        low, high = self._freq_split(x)
        if self.variant == "plain_ssd" or not self.use_frequency_conditioning:
            cond_delta_b = x
            cond_c = x
        elif self.variant == "reverse_assign":
            cond_delta_b = high if self.use_highfreq_for_c else low
            cond_c = low if self.use_lowfreq_for_delta_b else high
        else:
            cond_delta_b = low if self.use_lowfreq_for_delta_b else high
            cond_c = high if self.use_highfreq_for_c else low

        ssd_out = self._run_scan(x, self.delta_proj(cond_delta_b), self.b_proj(cond_delta_b), self.c_proj(cond_c))
        if self.mamba is None:
            mixed = ssd_out
        else:
            mamba_out = self.mamba(low if self.use_frequency_conditioning else x)
            fused = torch.cat([ssd_out, mamba_out], dim=-1)
            gate = torch.sigmoid(self.gate(fused))
            mixed = gate * ssd_out + (1.0 - gate) * mamba_out
        x = residual + mixed
        return x + self.ffn(x)


class PlainSSDBlock(FrequencyGuidedSSDBlock):
    def __init__(self, embed_dim: int, d_state: int = 64, dropout: float = 0.1, scan_mode: str = "single"):
        super().__init__(embed_dim=embed_dim, d_state=d_state, dropout=dropout, variant="plain_ssd", scan_mode=scan_mode, use_frequency_conditioning=False, use_lowfreq_for_delta_b=True, use_highfreq_for_c=True, use_noncausal_single_pass=True, use_multiscan_baseline=False)


class PhysicsConsistencyHead(nn.Module):
    def __init__(self, embed_dim: int, out_channels: int):
        super().__init__()
        self.decoder = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.Linear(embed_dim * 2, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class ClassifierHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.norm(x))


class LocalFrequencyBranch(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        third = max(1, embed_dim // 3)
        self.branch3 = DepthwiseSeparableConv2d(in_channels, third, 3)
        self.branch5 = DepthwiseSeparableConv2d(in_channels, third, 5)
        self.branch7 = DepthwiseSeparableConv2d(in_channels, embed_dim - 2 * third, 7)
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.se = SqueezeExcitation(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([self.branch3(x), self.branch5(x), self.branch7(x)], dim=1)
        return self.se(self.fuse(feat))


class PhyFreqVSSD(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        model_cfg = config["model"]
        in_channels = int(model_cfg["in_channels"])
        num_classes = int(model_cfg["num_classes"])
        embed_dim = int(model_cfg["embed_dim"])
        depth = int(model_cfg["depth"])
        d_state = int(model_cfg["d_state"])
        dropout = float(model_cfg["dropout"])
        patch_size = int(config["preprocessing"]["patch_size"])
        allow_mamba_fallback = bool(model_cfg.get("allow_mamba_fallback", False))
        self.use_dsst = bool(model_cfg["use_dsst"])
        self.use_spatial_freq_stem = bool(model_cfg.get("use_spatial_freq_stem", self.use_dsst))
        self.use_raster_tokenizer = bool(model_cfg.get("use_raster_tokenizer", self.use_dsst))
        self.use_local_branch = bool(model_cfg.get("use_local_branch", True))
        self.freq_stem = SpatialFrequencyStem(in_channels) if self.use_spatial_freq_stem else None
        self.tokenizer = SpatialRasterTokenizer(in_channels, embed_dim, patch_size) if self.use_raster_tokenizer else PlainTokenizer(in_channels, embed_dim)
        self.local_branch = LocalFrequencyBranch(in_channels, embed_dim) if self.use_local_branch else None
        if model_cfg["use_fg_ssd"]:
            self.blocks = nn.ModuleList([
                FrequencyGuidedSSDBlock(
                    embed_dim=embed_dim,
                    d_state=d_state,
                    dropout=dropout,
                    variant=model_cfg["fgssd_variant"],
                    scan_mode=model_cfg["scan_mode"],
                    use_frequency_conditioning=model_cfg["use_frequency_conditioning"],
                    use_lowfreq_for_delta_b=model_cfg["use_lowfreq_for_delta_b"],
                    use_highfreq_for_c=model_cfg["use_highfreq_for_c"],
                    use_noncausal_single_pass=model_cfg["use_noncausal_single_pass"],
                    use_multiscan_baseline=model_cfg["use_multiscan_baseline"],
                    use_mamba_branch=bool(model_cfg.get("use_mamba_branch", True)),
                    allow_mamba_fallback=allow_mamba_fallback,
                )
                for _ in range(depth)
            ])
        else:
            self.blocks = nn.ModuleList([PlainSSDBlock(embed_dim=embed_dim, d_state=d_state, dropout=dropout, scan_mode=model_cfg["scan_mode"]) for _ in range(depth)])
        self.fusion_gate = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.Sigmoid())
        self.fusion = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = ClassifierHead(embed_dim, num_classes)
        self.reconstruction_head = PhysicsConsistencyHead(embed_dim, in_channels) if (model_cfg["use_pch"] or model_cfg["use_plain_reconstruction_head"]) else None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | dict[str, float]]:
        model_input = x
        stem_stats: dict[str, float] = {}
        if self.freq_stem is not None:
            model_input, stem_stats = self.freq_stem(x)
        token_output = self.tokenizer(model_input)
        tokens = token_output.tokens
        for block in self.blocks:
            tokens = block(tokens)
        seq_features = tokens.mean(dim=1)
        if self.local_branch is None:
            features = seq_features
        else:
            local_map = self.local_branch(model_input)
            local_features = local_map.mean(dim=(2, 3))
            gated_seq = self.fusion_gate(local_features) * seq_features
            features = self.fusion(torch.cat([local_features, gated_seq], dim=-1))
        token_stats = dict(token_output.token_stats)
        token_stats.update(stem_stats)
        return {
            "logits": self.classifier(features),
            "reconstructed": self.reconstruction_head(features) if self.reconstruction_head is not None else None,
            "features": features,
            "token_stats": token_stats,
        }


def build_model(config: dict[str, Any]):
    model_cfg = config["model"]
    name = model_cfg["name"].lower()
    if name in {"phyfreq_vssd", "phyfreqssm"}:
        return PhyFreqVSSD(config)
    raise ValueError(f"Unsupported model name for the open-source release: {model_cfg['name']}")



import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast


class PhysicsConsistencyLoss(nn.Module):
    def forward(self, reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        reconstructed = F.normalize(reconstructed, dim=-1)
        target = F.normalize(target, dim=-1)
        sam = torch.acos(torch.clamp((reconstructed * target).sum(dim=-1), -1.0 + 1e-6, 1.0 - 1e-6)).mean()
        grad_recon = reconstructed[:, 1:] - reconstructed[:, :-1]
        grad_target = target[:, 1:] - target[:, :-1]
        smooth = F.l1_loss(grad_recon, grad_target)
        return sam + smooth


@dataclass
class LossBreakdown:
    total: torch.Tensor
    ce: torch.Tensor
    recon: torch.Tensor
    physics: torch.Tensor


class CompositeLoss(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        self.ce = nn.CrossEntropyLoss(label_smoothing=float(config["loss"]["label_smoothing"]))
        self.recon = nn.MSELoss() if config["loss"]["reconstruction_type"] == "mse" else nn.SmoothL1Loss()
        self.physics = PhysicsConsistencyLoss()

    def stage_weights(self, epoch: int) -> tuple[float, float]:
        if epoch <= int(self.config["loss"]["stage_a_epochs"]):
            return float(self.config["loss"]["stage_a_recon_weight"]), 0.0
        return float(self.config["loss"]["lambda_recon"]), float(self.config["loss"]["lambda_physics"])

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], epoch: int) -> LossBreakdown:
        ce_loss = self.ce(outputs["logits"], batch["label"])
        recon_loss = outputs["logits"].new_zeros(())
        physics_loss = outputs["logits"].new_zeros(())
        recon_weight, physics_weight = self.stage_weights(epoch)
        if outputs.get("reconstructed") is not None:
            recon_loss = self.recon(outputs["reconstructed"], batch["spectrum"])
            physics_loss = self.physics(outputs["reconstructed"], batch["spectrum"])
        total = ce_loss + recon_weight * recon_loss + physics_weight * physics_loss
        return LossBreakdown(total=total, ce=ce_loss, recon=recon_loss, physics=physics_loss)


@torch.no_grad()
def benchmark_model(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    config: dict[str, Any],
    include_flops: bool = True,
thop=None) -> dict[str, Any]:
    model.eval()
    batch = next(iter(loader))
    sample = torch.as_tensor(batch["patch"], device=device)
    warmup = int(config["benchmark"]["num_warmup"])
    iters = int(config["benchmark"]["num_iters"])

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(warmup):
        _ = model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _ = model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    latency_per_batch = elapsed / max(iters, 1)
    throughput = sample.size(0) / max(latency_per_batch, 1e-12)
    report = {
        "Params": parameter_count(model),
        "latency_per_batch_sec": latency_per_batch,
        "latency_per_sample_ms": latency_per_batch * 1000.0 / max(sample.size(0), 1),
        "throughput_samples_per_sec": throughput,
        "benchmark_batch_size": int(sample.size(0)),
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else 0.0,
        "FLOPs": None,
    }

    if include_flops:
        try:
            from thop import profile

            class _LogitsWrapper(nn.Module):
                def __init__(self, inner_model: torch.nn.Module):
                    super().__init__()
                    self.inner_model = inner_model

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self.inner_model(x)["logits"]

            macs, params = profile(_LogitsWrapper(model), inputs=(sample,), verbose=False)
            report["FLOPs"] = float(macs * 2)
            report["Params_from_thop"] = float(params)
        except Exception:
            report["FLOPs"] = None
    return report


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
    full_shape: tuple[int, int] | None = None,
    output_dir: str | Path | None = None,
    split_name: str = "test",
    dataset_name: str | None = None,
    map_loader=None,
    legacy_map_style: bool = False,
) -> dict[str, Any]:
    model.eval()
    all_logits, all_labels, all_coords = [], [], []
    for batch in loader:
        outputs = model(torch.as_tensor(batch["patch"], device=device))
        all_logits.append(outputs["logits"].detach().cpu())
        all_labels.append(torch.as_tensor(batch["label"]).cpu())
        all_coords.append(torch.as_tensor(batch["coord"]).cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy()
    preds = logits.argmax(dim=1).numpy()
    coords = torch.cat(all_coords, dim=0).numpy()
    metrics = classification_metrics(labels, preds, num_classes)
    metrics["num_samples"] = int(len(labels))

    if output_dir is not None:
        output_dir = Path(output_dir)
        save_json(metrics, output_dir / f"{split_name}_metrics.json")
    return metrics


@dataclass
class TrainerArtifacts:
    best_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    best_epoch: int
    output_dir: Path


def _build_optimizer(config: dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    lr = float(config["train"]["lr"])
    wd = float(config["train"]["weight_decay"])
    if config["train"]["optimizer"].lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


def _build_scheduler(config: dict[str, Any], optimizer: torch.optim.Optimizer):
    epochs = int(config["train"]["epochs"])
    warmup_epochs = int(config["train"]["warmup_epochs"])

    def schedule_fn(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_fn)


class Trainer:
    def __init__(self, config: dict[str, Any], model: torch.nn.Module, data_bundle, output_dir: str | Path):
        self.config = config
        self.model = model
        self.data_bundle = data_bundle
        self.output_dir = Path(output_dir)
        device_name = config["train"]["device"]
        if device_name == "cuda" and not torch.cuda.is_available():
            device_name = "cpu"
        self.device = torch.device(device_name)
        self.model.to(self.device)
        self.optimizer = _build_optimizer(config, model)
        self.scheduler = _build_scheduler(config, self.optimizer)
        self.criterion = CompositeLoss(config)
        self.scaler = GradScaler("cuda", enabled=bool(config["train"]["amp"]) and self.device.type == "cuda")
        self.history: list[dict[str, Any]] = []
        self.print_freq = max(1, int(config.get("logging", {}).get("print_freq", 20)))
        self.metric_mode = self._resolve_metric_mode()
        weights = config["train"].get("metric_weights", {})
        self.metric_weights = {
            "oa": float(weights.get("oa", 1.0)),
            "macro_f1": float(weights.get("macro_f1", 0.0)),
        }

    def _log(self, message: str) -> None:
        print(message, flush=True)

    def _resolve_metric_mode(self) -> str:
        metric_mode = str(self.config["train"].get("metric_for_best", "oa")).lower()
        if metric_mode == "auto":
            split_mode = str(self.config["dataset"].get("split_mode", "random_ratio")).lower()
            if split_mode in {"fixed_shot", "spatial_block"}:
                return "hybrid"
            return "oa"
        if metric_mode not in {"oa", "macro_f1", "hybrid"}:
            raise ValueError(f"Unsupported train.metric_for_best: {metric_mode}")
        return metric_mode

    def _selection_score(self, metrics: dict[str, Any]) -> float:
        oa = float(metrics["OA"])
        macro_f1 = float(metrics["Macro-F1"])
        if self.metric_mode == "oa":
            return oa
        if self.metric_mode == "macro_f1":
            return macro_f1
        return self.metric_weights["oa"] * oa + self.metric_weights["macro_f1"] * macro_f1

    def _metric_mode_label(self) -> str:
        if self.metric_mode != "hybrid":
            return self.metric_mode
        return (
            f"hybrid(oa={self.metric_weights['oa']:.2f},"
            f"macro_f1={self.metric_weights['macro_f1']:.2f})"
        )

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {
            "patch": torch.as_tensor(batch["patch"], device=self.device),
            "label": torch.as_tensor(batch["label"], device=self.device, dtype=torch.long),
            "spectrum": torch.as_tensor(batch["spectrum"], device=self.device, dtype=torch.float32),
            "coord": torch.as_tensor(batch["coord"], device=self.device),
        }

    def _run_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        meters = {"loss": 0.0, "ce": 0.0, "recon": 0.0, "physics": 0.0}
        count = 0
        start = time.perf_counter()
        total_batches = len(self.data_bundle.train_loader)
        train_dataset = getattr(self.data_bundle.train_loader, "dataset", None)
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        self._log(f"[Train] Epoch {epoch} started on {self.device.type} with {total_batches} mini-batches")
        for batch_idx, batch in enumerate(self.data_bundle.train_loader, start=1):
            batch = self._move_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                outputs = self.model(batch["patch"])
                losses = self.criterion(outputs, batch, epoch)
            self.scaler.scale(losses.total).backward()
            if self.config["train"]["grad_clip"] is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config["train"]["grad_clip"]))
            self.scaler.step(self.optimizer)
            self.scaler.update()
            count += 1
            meters["loss"] += float(losses.total.detach().cpu())
            meters["ce"] += float(losses.ce.detach().cpu())
            meters["recon"] += float(losses.recon.detach().cpu())
            meters["physics"] += float(losses.physics.detach().cpu())
            should_print = batch_idx == 1 or batch_idx % self.print_freq == 0 or batch_idx == total_batches
            if should_print:
                elapsed = time.perf_counter() - start
                avg_loss = meters["loss"] / max(count, 1)
                self._log(
                    f"[Train] Epoch {epoch} batch {batch_idx}/{total_batches} "
                    f"loss={avg_loss:.4f} elapsed={elapsed:.1f}s"
                )
        self.scheduler.step()
        meters = {key: value / max(count, 1) for key, value in meters.items()}
        meters["epoch_time"] = time.perf_counter() - start
        return meters

    def fit(self) -> TrainerArtifacts:
        best_epoch = -1
        best_score = -float("inf")
        best_metrics: dict[str, Any] = {}
        best_state = None
        no_improve = 0
        total_start = time.perf_counter()
        self._log(
            f"[Run] Starting training: exp={self.config['experiment']['name']} "
            f"dataset={self.config['dataset']['name']} "
            f"device={self.device.type} epochs={int(self.config['train']['epochs'])} "
            f"best_metric={self._metric_mode_label()}"
        )

        for epoch in range(1, int(self.config["train"]["epochs"]) + 1):
            train_stats = self._run_epoch(epoch)
            val_metrics = evaluate_model(
                self.model,
                self.data_bundle.val_loader,
                self.device,
                self.data_bundle.num_classes,
                dataset_name=self.config["dataset"]["name"],
            )
            score = self._selection_score(val_metrics)
            combined = {
                "epoch": epoch,
                **train_stats,
                **{f"val_{k}": v for k, v in val_metrics.items() if not isinstance(v, list)},
                "val_selection_score": score,
                "metric_for_best": self.metric_mode,
            }
            self.history.append(combined)
            self._log(
                f"[Eval] Epoch {epoch} "
                f"train_loss={train_stats['loss']:.4f} "
                f"val_OA={val_metrics['OA']:.4f} "
                f"val_AA={val_metrics['AA']:.4f} "
                f"val_MacroF1={val_metrics['Macro-F1']:.4f} "
                f"select_score={score:.4f} mode={self._metric_mode_label()}"
            )
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_metrics = val_metrics
                no_improve = 0
                best_state = {key: value.detach().cpu() for key, value in self.model.state_dict().items()}
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": best_state,
                        "optimizer_state": self.optimizer.state_dict(),
                        "config": self.config,
                        "selection_score": score,
                        "metric_for_best": self.metric_mode,
                    },
                    self.output_dir / "best_checkpoint.pt",
                )
                self._log(
                    f"[Checkpoint] New best model at epoch {epoch} "
                    f"score={score:.4f} mode={self._metric_mode_label()}"
                )
            else:
                no_improve += 1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "config": self.config,
                    "selection_score": score,
                    "metric_for_best": self.metric_mode,
                },
                self.output_dir / "last_checkpoint.pt",
            )
            if no_improve >= int(self.config["train"]["early_stopping_patience"]):
                self._log(f"[EarlyStop] No improvement for {no_improve} epochs, stopping at epoch {epoch}")
                break

        total_train_time = time.perf_counter() - total_start
        if best_state is not None:
            self.model.load_state_dict(best_state)
        test_metrics = evaluate_model(
            self.model,
            self.data_bundle.test_loader,
            self.device,
            self.data_bundle.num_classes,
            output_dir=self.output_dir,
            split_name="test",
        )
        complexity_report = {
            "training_time_per_epoch": float(np.mean([row["epoch_time"] for row in self.history])) if self.history else 0.0,
            "total_training_time": total_train_time,
        }
        complexity_report.update(benchmark_model(self.model, self.data_bundle.test_loader, self.device, self.config))
        flops_report = {
            "dataset": self.config["dataset"]["name"],
            "exp_name": self.config["experiment"]["name"],
            "model_name": self.config["model"]["name"],
            "patch_size": int(self.config["preprocessing"]["patch_size"]),
            "pca_bands": self.config["preprocessing"].get("pca_bands"),
            "profile_mode": "post_train_benchmark",
            "device": self.device.type,
            "Params": complexity_report.get("Params"),
            "FLOPs": complexity_report.get("FLOPs"),
            "latency_per_sample_ms": complexity_report.get("latency_per_sample_ms"),
            "throughput_samples_per_sec": complexity_report.get("throughput_samples_per_sec"),
            "peak_gpu_memory_mb": complexity_report.get("peak_gpu_memory_mb"),
            "training_time_per_epoch": complexity_report.get("training_time_per_epoch"),
            "total_training_time": complexity_report.get("total_training_time"),
        }
        save_json(complexity_report, self.output_dir / "complexity_report.json")
        save_json(flops_report, self.output_dir / "flops_report.json")
        save_json(
            {
                **test_metrics,
                **complexity_report,
                "best_epoch": best_epoch,
                "best_selection_score": best_score,
                "metric_for_best": self.metric_mode,
                "metric_weights": self.metric_weights,
            },
            self.output_dir / "metrics.json",
        )
        save_csv([{k: v for k, v in row.items() if not isinstance(v, list)} for row in self.history], self.output_dir / "metrics.csv")
        save_text("\n".join(json.dumps(row, ensure_ascii=False) for row in self.history), self.output_dir / "train_log.txt")
        self._log(
            f"[Done] best_epoch={best_epoch} best_score={best_score:.4f} "
            f"test_OA={test_metrics['OA']:.4f} test_AA={test_metrics['AA']:.4f} "
            f"test_MacroF1={test_metrics['Macro-F1']:.4f}"
        )
        return TrainerArtifacts(best_metrics=best_metrics, test_metrics=test_metrics, best_epoch=best_epoch, output_dir=self.output_dir)
