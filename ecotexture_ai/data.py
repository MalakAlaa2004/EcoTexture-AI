from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
from scipy.cluster.vq import vq
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split

from .config import SEED, SIFT_K

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class Sample:
    path: Path
    label: str


def iter_class_folders(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()])


def collect_samples(root: Path) -> list[Sample]:
    samples: list[Sample] = []
    for class_dir in iter_class_folders(root):
        for file_path in sorted(class_dir.iterdir()):
            if file_path.suffix.lower() in IMG_EXTS:
                samples.append(Sample(file_path, class_dir.name))
    return samples


def split_samples(samples: list[Sample], val_ratio: float = 0.15, test_ratio: float = 0.15):
    labels = [sample.label for sample in samples]
    train_samples, temp_samples = train_test_split(
        samples,
        test_size=val_ratio + test_ratio,
        random_state=SEED,
        stratify=labels,
    )
    temp_labels = [sample.label for sample in temp_samples]
    rel_test = test_ratio / (val_ratio + test_ratio)
    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=rel_test,
        random_state=SEED,
        stratify=temp_labels,
    )
    return train_samples, val_samples, test_samples


def build_class_index(samples: Iterable[Sample]) -> dict[str, int]:
    classes = sorted({sample.label for sample in samples})
    return {label: idx for idx, label in enumerate(classes)}


def save_split_manifest(samples: list[Sample], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{"path": str(sample.path), "label": sample.label} for sample in samples]
    output_path.write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")


def load_split_manifest(path: Path) -> list[Sample]:
    payload = __import__("json").loads(path.read_text(encoding="utf-8"))
    return [Sample(Path(item["path"]), item["label"]) for item in payload]


def image_to_array(image_path: Path, img_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, img_size, interpolation=cv2.INTER_AREA)
    return image.astype(np.float32) / 255.0


def sift_histogram(image_path: Path, centers: np.ndarray, vocab_size: int = SIFT_K) -> np.ndarray:
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return np.zeros(vocab_size, dtype=np.float32)
    # Resize to fixed size for performance and consistency
    gray = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_AREA)
    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(gray, None)
    histogram = np.zeros(vocab_size, dtype=np.float32)
    if descriptors is None or len(descriptors) == 0:
        return histogram
    predictions, _ = vq(descriptors.astype(np.float32), centers.astype(np.float32))
    hist, _ = np.histogram(predictions, bins=np.arange(vocab_size + 1), density=True)
    return hist.astype(np.float32)


def fit_sift_vocabulary(samples: list[Sample], vocab_size: int = SIFT_K, max_images: int = 150) -> np.ndarray:
    sift = cv2.SIFT_create()
    descriptors: list[np.ndarray] = []
    per_class_counter: dict[str, int] = defaultdict(int)

    for sample in samples:
        if per_class_counter[sample.label] >= max_images:
            continue
        gray = cv2.imread(str(sample.path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        _, des = sift.detectAndCompute(gray, None)
        if des is not None and len(des) > 0:
            descriptors.append(des)
            per_class_counter[sample.label] += 1

    if not descriptors:
        raise RuntimeError("No SIFT descriptors found while building vocabulary.")

    stacked = np.vstack(descriptors).astype(np.float32)
    if len(stacked) > 80000:
        indices = np.random.default_rng(SEED).choice(len(stacked), 80000, replace=False)
        stacked = stacked[indices]

    vocab_size = min(vocab_size, len(stacked))
    if vocab_size < 2:
        raise RuntimeError("Not enough SIFT descriptors to build a vocabulary.")

    kmeans = MiniBatchKMeans(n_clusters=vocab_size, random_state=SEED, batch_size=2048, n_init=2, max_iter=50)
    kmeans.fit(stacked)
    return kmeans.cluster_centers_.astype(np.float32)


def copy_samples_to_splits(samples: list[Sample], output_root: Path) -> None:
    train_samples, val_samples, test_samples = split_samples(samples)
    for split_name, split_samples_list in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        for sample in split_samples_list:
            destination = output_root / split_name / sample.label / sample.path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                destination.write_bytes(sample.path.read_bytes())


def load_samples_from_roots(roots: Iterable[Path]) -> list[Sample]:
    samples: list[Sample] = []
    for root in roots:
        samples.extend(collect_samples(root))
    return samples