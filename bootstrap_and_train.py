"""
EcoTexture AI — Smart Dataset Bootstrap + Training Launcher
============================================================
1. Checks what data already exists in D:\EcoTexture AI\data\raw\trashnet
2. Copies any legacy data from D:\EcoTexture A when available
3. Downloads any missing classes from HuggingFace garythung/trashnet
4. Finds the best available COCO/MET checkpoint for transfer
5. Kicks off fine_tune.py with the right args

Run:
    python bootstrap_and_train.py
"""
from __future__ import annotations

import json
import shutil
import sys
import subprocess
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
# Legacy recovery source and active root
SOURCE_ROOT = Path(r"D:\grad project\trial")
LEGACY_ECO_A_RAW = Path(r"D:\EcoTexture A\data\raw\trashnet")
ECO_AI_RAW = Path(r"D:\EcoTexture AI\data\raw\trashnet")
MODELS_DIR = Path(r"D:\EcoTexture AI\models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
ECO_AI_RAW.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# CHECKPOINT PRIORITY  (COCO → MET → contrastive → None)
# ─────────────────────────────────────────────────────────────
CHECKPOINT_CANDIDATES = [
    # COCO-mixed ConvNeXt (best for real-world clutter robustness)
    SOURCE_ROOT / "benchmark_experiments_external/models/coco_mixed/ConvNeXt_Tiny/seed42_best.weights.h5",
    # MET ConvNeXt (best for texture priors)
    SOURCE_ROOT / "benchmark_experiments_external/models/met/ConvNeXt_Tiny/seed42_best.weights.h5",
    # TimeLens contrastive (hybrid CNN+SIFT, closest architecture match)
    SOURCE_ROOT / "hybrid/research_v2_contrastive/contrastive_final.h5",
    SOURCE_ROOT / "cloud_deployment_backend/contrastive_final.h5",
]

SIFT_CANDIDATES = [
    SOURCE_ROOT / "cloud_deployment_backend/sift_kmeans_centers.npy",
    SOURCE_ROOT / "hybrid/meta/sift_kmeans_centers.npy",
]

# ─────────────────────────────────────────────────────────────
# TRASHNET EXPECTED CLASSES
# ─────────────────────────────────────────────────────────────
TRASHNET_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
MIN_IMAGES_PER_CLASS = 100  # below this we download more

# ─────────────────────────────────────────────────────────────
# STEP 1 — Audit existing data
# ─────────────────────────────────────────────────────────────
def audit_classes(root: Path) -> dict[str, int]:
    counts = {}
    for cls in TRASHNET_CLASSES:
        cls_dir = root / cls
        if cls_dir.exists():
            imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.jpeg")) + list(cls_dir.glob("*.png"))
            counts[cls] = len(imgs)
        else:
            counts[cls] = 0
    return counts

# ─────────────────────────────────────────────────────────────
# STEP 2 — Copy data from EcoTexture A → EcoTexture AI
# ─────────────────────────────────────────────────────────────
def sync_from_eco_a():
    for cls in TRASHNET_CLASSES:
        src = LEGACY_ECO_A_RAW / cls
        dst = ECO_AI_RAW / cls
        if src.exists() and src != dst:
            dst.mkdir(parents=True, exist_ok=True)
            for img in src.glob("*"):
                shutil.copy2(img, dst / img.name)

# ─────────────────────────────────────────────────────────────
# STEP 3 — Download missing classes via HuggingFace datasets
# ─────────────────────────────────────────────────────────────
DOWNLOAD_SCRIPT = r"""
import sys
from pathlib import Path
from datasets import load_dataset

OUTPUT_DIR = Path(r"D:\EcoTexture AI\data\raw\trashnet")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading garythung/trashnet from HuggingFace...")
try:
    ds = load_dataset("garythung/trashnet", split="train", trust_remote_code=True)
    print(f"  Total samples: {len(ds)}")
    
    label_names = ds.features["label"].names if hasattr(ds.features["label"], "names") else None
    
    for i, sample in enumerate(ds):
        label = sample["label"]
        if isinstance(label, int) and label_names:
            label_str = label_names[label]
        else:
            label_str = str(label)
        
        # Normalize class name to match TrashNet structure
        label_str = label_str.lower().replace(" ", "_")
        
        cls_dir = OUTPUT_DIR / label_str
        cls_dir.mkdir(parents=True, exist_ok=True)
        
        img = sample["image"]
        img_path = cls_dir / f"{i:05d}.jpg"
        if not img_path.exists():
            img.save(img_path, format="JPEG", quality=90)
        
        if i % 200 == 0:
            print(f"  Saved {i} images...")
    
    print("Download complete.")
    
    # Print final counts
    for cls_dir in sorted(OUTPUT_DIR.iterdir()):
        if cls_dir.is_dir():
            count = len(list(cls_dir.glob("*.jpg")))
            print(f"  {cls_dir.name}: {count} images")

except Exception as e:
    print(f"HuggingFace download failed: {e}")
    print("Trying alternative: Xenova/waste-classification-data ...")
    try:
        ds2 = load_dataset("Xenova/waste-classification-data", split="train", trust_remote_code=True)
        label_names2 = ds2.features["label"].names if hasattr(ds2.features["label"], "names") else None
        for i, sample in enumerate(ds2):
            label = sample["label"]
            if isinstance(label, int) and label_names2:
                label_str = label_names2[label].lower().replace(" ", "_")
            else:
                label_str = str(label).lower()
            cls_dir = OUTPUT_DIR / label_str
            cls_dir.mkdir(parents=True, exist_ok=True)
            img = sample["image"]
            img_path = cls_dir / f"w{i:05d}.jpg"
            if not img_path.exists():
                img.save(img_path, format="JPEG", quality=90)
            if i % 200 == 0:
                print(f"  Alt dataset: saved {i} images...")
        print("Alternative download complete.")
    except Exception as e2:
        print(f"Both downloads failed: {e2}")
        sys.exit(1)
"""

# ─────────────────────────────────────────────────────────────
# STEP 4 — Copy SIFT centers to models dir
# ─────────────────────────────────────────────────────────────
def setup_sift_centers():
    dst = MODELS_DIR / "sift_centers.npy"
    if dst.exists():
        print(f"[OK] SIFT centers already in {dst}")
        return True
    for candidate in SIFT_CANDIDATES:
        if candidate.exists():
            shutil.copy2(candidate, dst)
            print(f"[OK] Copied SIFT centers from {candidate.name}")
            return True
    print("[!!] No SIFT centers found - will fit from training data")
    return False

# ─────────────────────────────────────────────────────────────
# STEP 5 — Find best checkpoint
# ─────────────────────────────────────────────────────────────
def find_best_checkpoint() -> str | None:
    for c in CHECKPOINT_CANDIDATES:
        if c.exists():
            label = (
                "COCO-mixed ConvNeXt" if "coco" in str(c).lower() else
                "MET ConvNeXt" if "met" in str(c).lower() else
                "TimeLens contrastive hybrid"
            )
            print(f"[OK] Using checkpoint: {label}\n  -> {c}")
            return str(c)
    print("[!!] No checkpoint found - training from ImageNet scratch")
    return None

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  EcoTexture AI - Bootstrap & Train")
    print("=" * 60)

    # -- Audit existing data ------------------------------------
    print("\n[1/5] Auditing existing data...")
    sync_from_eco_a()
    counts = audit_classes(ECO_AI_RAW)
    total  = sum(counts.values())
    print(f"  Current state ({total} total images):")
    missing = []
    for cls, n in counts.items():
        if n < MIN_IMAGES_PER_CLASS:
            missing.append(cls)
            status = "[!!]"
        else:
            status = "[OK]"
        print(f"    {status} {cls}: {n} images")

    # -- Download if needed ------------------------------------
    if missing:
        print(f"\n[2/5] Missing/incomplete classes: {missing}")
        print("  Downloading TrashNet from HuggingFace...")
        
        # Write and run the download script
        dl_script = Path(r"D:\EcoTexture AI\_download_hf.py")
        dl_script.write_text(DOWNLOAD_SCRIPT, encoding="utf-8")
        
        # Install datasets if needed
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "datasets", "Pillow", "-q"],
            check=False
        )
        result = subprocess.run(
            [sys.executable, str(dl_script)],
            capture_output=False
        )
        if result.returncode != 0:
            print("  [!!] Download had errors - continuing with available data")
        
        # Re-audit
        counts = audit_classes(ECO_AI_RAW)
        total  = sum(counts.values())
        print(f"\n  Updated state ({total} total images):")
        for cls, n in counts.items():
            print(f"    {cls}: {n} images")
    else:
        print("\n[2/5] [OK] All classes have sufficient data - skipping download")

    # ── SIFT centers ──────────────────────────────────────────
    print("\n[3/5] Setting up SIFT vocabulary...")
    setup_sift_centers()

    # ── Best checkpoint ───────────────────────────────────────
    print("\n[4/5] Finding best transfer checkpoint (COCO/MET)...")
    ckpt = find_best_checkpoint()

    backbone = "efficientnetb0"
    if ckpt:
        if "convnext" in ckpt.lower():
            backbone = "convnext_tiny"
        elif "mobilenet" in ckpt.lower():
            backbone = "mobilenetv3"

    # ── Launch training ───────────────────────────────────────
    print("\n[5/5] Launching fine-tuning...")
    print("=" * 60)

    cmd = [
        sys.executable, "-u", r"D:\EcoTexture AI\fine_tune.py",
        "--raw-root",       str(ECO_AI_RAW),
        "--processed-root", r"D:\EcoTexture AI\data\processed\trashnet",
        "--epochs-stage1",  "12",
        "--epochs-stage2",  "8",
        "--batch-size",     "16",
        "--backbone",       backbone,
        "--technique",      "contrastive_transfer",
    ]
    if ckpt:
        cmd += ["--contrastive-checkpoint", ckpt]

    print(f"  Command: {' '.join(cmd[1:])}\n")
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
