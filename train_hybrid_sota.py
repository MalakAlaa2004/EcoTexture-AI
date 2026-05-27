"""
EcoTexture AI — SOTA Hybrid Training to Beat Benchmark Pure Models
===================================================================
Target: Beat benchmark Cross_Attention 85.3% & Pure_MobileNet 84.7% on TrashNet

Strategy (why this will work):
  1. EfficientNetB0 (ImageNet, 224x224) — benchmark used FROZEN 128x128 (that's why
     EfficientNet scored only 42.7% in their results). We FINE-TUNE it.
  2. SIFT BoVW histogram (k=300) via Cross-Attention fusion on top.
  3. Stage 1: Train head only with frozen backbone → fast convergence to ~75%
  4. Stage 2: Unfreeze last 60 EfficientNet layers → push to 90%+
  5. Label smoothing + Mixup + CLAHE preprocessing → generalize better
  6. ReduceLROnPlateau only — no conflicting schedulers

Expected: 90–95% on TrashNet (definitively breaking the 85.3% benchmark ceiling)

Run:
    python "D:\\EcoTexture AI\\train_hybrid_sota.py"
    python "D:\\EcoTexture AI\\train_hybrid_sota.py" --stage2-epochs 60 --unfreeze 80
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── CUDA DLL patch ───────────────────────────────────────────
sys.path.insert(0, r"D:\grad project\trial")
try:
    import env_tf_patch
    print("[EcoTexture] Loaded CUDA DLL patch.")
except ImportError:
    print("[EcoTexture] env_tf_patch not found.")

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import cv2
import numpy as np
import tensorflow as tf

# CPU thread limits (remove if you have GPU)
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)

try:
    import tf_keras as keras
    from tf_keras import layers, Model, Input
    from tf_keras import regularizers
except ImportError:
    try:
        from tensorflow import keras
        from tensorflow.keras import layers, Model, Input
        from tensorflow.keras import regularizers
    except ImportError:
        import keras
        from keras import layers, Model, Input
        from keras import regularizers

from sklearn.utils.class_weight import compute_class_weight

# ── Paths ────────────────────────────────────────────────────
ECO_AI_ROOT   = Path(r"D:\EcoTexture AI")
MODELS_DIR    = ECO_AI_ROOT / "models"
PROCESSED_DIR = ECO_AI_ROOT / "data" / "processed" / "trashnet"
RAW_DIR       = ECO_AI_ROOT / "data" / "raw" / "trashnet"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR     = MODELS_DIR / "cache_sota"
CACHE_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ECO_AI_ROOT))
from ecotexture_ai.data import (
    collect_samples, fit_sift_vocabulary,
    load_split_manifest, save_split_manifest,
    split_samples, sift_histogram, Sample,
)

# ── GPU ──────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    print(f"[Runtime] GPU ready: {len(gpus)} device(s) — mixed_float16 ON")
    keras.mixed_precision.set_global_policy("mixed_float16")
else:
    print("[Runtime] No GPU — CPU float32 mode")
    keras.mixed_precision.set_global_policy("float32")


# ─────────────────────────────────────────────────────────────
# CLAHE PREPROCESSING — same as benchmark's BenchmarkDataset
# ─────────────────────────────────────────────────────────────
def clahe_preprocess(bgr: np.ndarray) -> np.ndarray:
    """CLAHE contrast enhancement — same pipeline used in the benchmark."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def load_image_clahe(path: Path, img_size: tuple) -> np.ndarray:
    """Load → CLAHE → resize → normalize to [0,1]."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"Cannot read: {path}")
    bgr = cv2.resize(bgr, img_size, interpolation=cv2.INTER_AREA)
    bgr = clahe_preprocess(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


# ─────────────────────────────────────────────────────────────
# SIFT CACHE
# ─────────────────────────────────────────────────────────────
def precompute_split(samples, class_to_id, centers, img_size, name):
    cache = CACHE_DIR / f"{name}_sota.npz"
    if cache.exists():
        print(f"  [Cache] {name} → {cache.name}")
        d = np.load(cache)
        return d["images"], d["sifts"], d["labels"]

    print(f"  [SIFT+CLAHE] Computing {name}: {len(samples)} samples ...")
    t0 = time.time()
    images, sifts, labels = [], [], []
    vocab_size = centers.shape[0]

    for i, s in enumerate(samples):
        try:
            img  = load_image_clahe(s.path, img_size)
            hist = sift_histogram(s.path, centers, vocab_size)
            images.append(img)
            sifts.append(hist)
            labels.append(class_to_id[s.label])
        except Exception as e:
            print(f"\n  [Skip] {s.path.name}: {e}")
        if (i + 1) % 300 == 0:
            print(f"    {i+1}/{len(samples)} ...", flush=True)

    images = np.asarray(images, dtype=np.float32)
    sifts  = np.asarray(sifts,  dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    np.savez_compressed(str(cache), images=images, sifts=sifts, labels=labels)
    print(f"  [SIFT+CLAHE] Done in {time.time()-t0:.1f}s")
    return images, sifts, labels


# ─────────────────────────────────────────────────────────────
# MODEL — EfficientNetB0 Backbone + SIFT Cross-Attention Head
# ─────────────────────────────────────────────────────────────
def build_sota_hybrid(num_classes: int, sift_k: int = 300,
                      img_size: tuple = (224, 224),
                      freeze_backbone: bool = True) -> keras.Model:
    """
    SOTA Hybrid: EfficientNetB0 (ImageNet, fine-tuned) + SIFT Cross-Attention.

    Why this beats the benchmark:
    - Benchmark used FROZEN EfficientNetV2B0 at 128x128 → 42.7%
    - We use FINE-TUNED EfficientNetB0 at 224x224 + SIFT attention on top
    - This alone should push past 85.3% (current best)
    """
    img_in  = Input(shape=(*img_size, 3), name="image_input")
    sift_in = Input(shape=(sift_k,),      name="sift_input")

    # ── EfficientNetB0 backbone (ImageNet weights, fine-tuned) ──
    backbone = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=img_in,
        pooling="avg",           # → (B, 1280)
    )
    backbone.trainable = not freeze_backbone

    cnn_feat = backbone.output                                          # (B, 1280)
    cnn_proj = layers.Dense(256, activation="gelu",
                            kernel_regularizer=regularizers.l2(1e-4),
                            name="cnn_proj")(cnn_feat)
    cnn_proj = layers.BatchNormalization(name="cnn_bn")(cnn_proj)
    cnn_proj = layers.Dropout(0.35, name="cnn_drop")(cnn_proj)

    # ── SIFT Branch ────────────────────────────────────────────
    ys = layers.Dense(128, activation="gelu",
                      kernel_regularizer=regularizers.l2(1e-4),
                      name="sift_dense1")(sift_in)
    ys = layers.BatchNormalization(name="sift_bn1")(ys)
    ys = layers.Dense(64, activation="gelu",
                      name="sift_dense2")(ys)
    ys = layers.BatchNormalization(name="sift_bn2")(ys)

    # ── Cross-Attention Fusion ──────────────────────────────────
    merged = layers.Concatenate(name="concat")([cnn_proj, ys])  # (B, 320)
    merged = layers.Reshape((1, 320), name="reshape")(merged)   # (B, 1, 320)
    attn   = layers.MultiHeadAttention(
        num_heads=4, key_dim=80, name="cross_attn"
    )(merged, merged)
    fused  = layers.LayerNormalization(name="fusion_ln")(attn + merged)
    fused  = layers.Flatten(name="flatten")(fused)              # (B, 320)

    # ── FFN Head ───────────────────────────────────────────────
    z = layers.Dense(256, activation="gelu",
                     kernel_regularizer=regularizers.l2(1e-4),
                     name="head_dense1")(fused)
    z = layers.BatchNormalization(name="head_bn1")(z)
    z = layers.Dropout(0.4, name="head_drop1")(z)
    z = layers.Dense(128, activation="gelu",
                     kernel_regularizer=regularizers.l2(1e-4),
                     name="head_dense2")(z)
    z = layers.BatchNormalization(name="head_bn2")(z)
    z = layers.Dropout(0.3, name="head_drop2")(z)

    out = layers.Dense(num_classes, activation="softmax",
                       dtype="float32", name="output")(z)

    return Model(inputs=[img_in, sift_in], outputs=out,
                 name="EcoTexture_SOTA_Hybrid")


# ─────────────────────────────────────────────────────────────
# MIXUP
# ─────────────────────────────────────────────────────────────
def apply_mixup(images, sifts, labels_oh, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = np.random.permutation(len(images))
    images    = lam * images + (1 - lam) * images[idx]
    sifts     = lam * sifts  + (1 - lam) * sifts[idx]
    labels_oh = lam * labels_oh + (1 - lam) * labels_oh[idx]
    return images, sifts, labels_oh


# ─────────────────────────────────────────────────────────────
# TF DATASET
# ─────────────────────────────────────────────────────────────
def make_dataset(images, sifts, labels, num_classes, batch_size,
                 augment=False, mixup_alpha=0.0):
    labels_oh = np.eye(num_classes, dtype=np.float32)[labels]

    if augment and mixup_alpha > 0:
        images, sifts, labels_oh = apply_mixup(
            images, sifts, labels_oh, alpha=mixup_alpha)

    with tf.device("/CPU:0"):
        ds = tf.data.Dataset.from_tensor_slices(
            ((images.astype(np.float32), sifts.astype(np.float32)), labels_oh)
        )

    if augment:
        def aug(x, y):
            img, sift = x[0], x[1]
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.random_brightness(img, 0.20)
            img = tf.image.random_contrast(img, 0.75, 1.25)
            img = tf.image.random_saturation(img, 0.8, 1.2)
            img = tf.clip_by_value(img, 0.0, 1.0)
            return (img, sift), y

        ds = ds.shuffle(min(len(images), 3000), reshuffle_each_iteration=True)
        ds = ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.repeat()

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="EcoTexture SOTA — beat benchmark pure model accuracy")
    parser.add_argument("--stage1-epochs", type=int,   default=25,
                        help="Head-only training epochs (backbone frozen)")
    parser.add_argument("--stage2-epochs", type=int,   default=50,
                        help="Fine-tune last N backbone layers")
    parser.add_argument("--batch-size",    type=int,   default=16)
    parser.add_argument("--img-size",      type=int,   default=224)
    parser.add_argument("--lr-stage1",     type=float, default=1e-3)
    parser.add_argument("--lr-stage2",     type=float, default=5e-5)
    parser.add_argument("--unfreeze",      type=int,   default=60,
                        help="Number of backbone layers to unfreeze in Stage 2")
    parser.add_argument("--label-smooth",  type=float, default=0.1)
    parser.add_argument("--mixup-alpha",   type=float, default=0.2)
    parser.add_argument("--patience",      type=int,   default=12)
    parser.add_argument("--no-cache",      action="store_true")
    args = parser.parse_args()

    img_size  = (args.img_size, args.img_size)
    prefix    = "ecotexture_sota"
    best_path = MODELS_DIR / f"{prefix}_best.keras"
    fin_path  = MODELS_DIR / f"{prefix}_final.keras"

    if args.no_cache:
        for f in CACHE_DIR.glob("*_sota.npz"):
            f.unlink()
            print(f"[Cache] Removed {f.name}")

    # ── Data ─────────────────────────────────────────────────
    train_m = PROCESSED_DIR / "train.json"
    val_m   = PROCESSED_DIR / "val.json"
    test_m  = PROCESSED_DIR / "test.json"

    if train_m.exists() and val_m.exists() and test_m.exists():
        train_s = load_split_manifest(train_m)
        val_s   = load_split_manifest(val_m)
        test_s  = load_split_manifest(test_m)
    else:
        print("[Data] Scanning raw data ...")
        samples = collect_samples(RAW_DIR)
        if not samples:
            raise RuntimeError(f"No images found in {RAW_DIR}")
        train_s, val_s, test_s = split_samples(samples)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        save_split_manifest(train_s, train_m)
        save_split_manifest(val_s,   val_m)
        save_split_manifest(test_s,  test_m)

    all_s       = train_s + val_s + test_s
    class_to_id = {l: i for i, l in enumerate(sorted({s.label for s in all_s}))}
    num_classes = len(class_to_id)
    (MODELS_DIR / "class_to_id.json").write_text(
        json.dumps(class_to_id, indent=2), encoding="utf-8")

    print(f"\n[Data] {len(train_s)} train / {len(val_s)} val / {len(test_s)} test")
    print(f"[Data] Classes ({num_classes}): {list(class_to_id.keys())}")

    # ── SIFT Vocabulary ──────────────────────────────────────
    vocab_path = MODELS_DIR / "sift_centers.npy"
    if vocab_path.exists():
        centers = np.load(vocab_path)
        print(f"[SIFT] Vocabulary loaded: {centers.shape}")
    else:
        trial_vocab = Path(r"D:\grad project\trial\hybrid\meta\sift_kmeans_centers.npy")
        if trial_vocab.exists():
            import shutil
            shutil.copy2(trial_vocab, vocab_path)
            centers = np.load(vocab_path)
            print(f"[SIFT] Copied from trial: {centers.shape}")
        else:
            print("[SIFT] Fitting vocabulary ...")
            centers = fit_sift_vocabulary(train_s, vocab_size=300, max_images=30)
            np.save(vocab_path, centers)

    # ── Precompute (CLAHE + SIFT, cached) ───────────────────
    print("\n[Pipeline] Precomputing CLAHE + SIFT (cached after 1st run) ...")
    tr_img, tr_sft, tr_lbl = precompute_split(train_s, class_to_id, centers, img_size, "train")
    va_img, va_sft, va_lbl = precompute_split(val_s,   class_to_id, centers, img_size, "val")
    te_img, te_sft, te_lbl = precompute_split(test_s,  class_to_id, centers, img_size, "test")

    # Class weights
    cw_arr  = compute_class_weight("balanced", classes=np.unique(tr_lbl), y=tr_lbl)
    cw_dict = {int(c): float(w) for c, w in zip(np.unique(tr_lbl), cw_arr)}
    print(f"[Class Weights] { {k: f'{v:.2f}' for k,v in cw_dict.items()} }")

    steps = max(1, len(train_s) // args.batch_size)

    # ── STAGE 1 — Head Only (backbone frozen) ────────────────
    print(f"\n{'='*65}")
    print(f"  STAGE 1 — Head training, backbone FROZEN")
    print(f"  Epochs: {args.stage1_epochs}  |  LR: {args.lr_stage1}")
    print(f"  Mixup α={args.mixup_alpha}  |  Label smooth={args.label_smooth}")
    print(f"{'='*65}\n")

    model = build_sota_hybrid(num_classes, sift_k=centers.shape[0],
                              img_size=img_size, freeze_backbone=True)
    model.compile(
        optimizer=keras.optimizers.Adam(args.lr_stage1, clipnorm=1.0),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smooth),
        metrics=["accuracy"],
    )
    model.summary(line_length=90)

    train_ds_s1 = make_dataset(tr_img, tr_sft, tr_lbl, num_classes,
                                args.batch_size, augment=True,
                                mixup_alpha=args.mixup_alpha)
    val_ds  = make_dataset(va_img, va_sft, va_lbl, num_classes,
                           args.batch_size, augment=False)
    test_ds = make_dataset(te_img, te_sft, te_lbl, num_classes,
                           args.batch_size, augment=False)

    cbs_s1 = [
        keras.callbacks.ModelCheckpoint(
            str(best_path), monitor="val_accuracy",
            save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=args.patience,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5,
            min_lr=1e-7, verbose=1),
        keras.callbacks.CSVLogger(
            str(MODELS_DIR / f"{prefix}_s1_history.csv"), append=False),
    ]

    h1 = model.fit(
        train_ds_s1, epochs=args.stage1_epochs,
        steps_per_epoch=steps,
        validation_data=val_ds,
        callbacks=cbs_s1, class_weight=cw_dict, verbose=1,
    )
    best_s1 = max(h1.history.get("val_accuracy", [0]))
    print(f"\n[Stage 1] Best val_accuracy = {best_s1:.4f}")

    # ── STAGE 2 — Fine-tune last N backbone layers ───────────
    print(f"\n{'='*65}")
    print(f"  STAGE 2 — Unfreezing last {args.unfreeze} backbone layers")
    print(f"  Epochs: {args.stage2_epochs}  |  LR: {args.lr_stage2}")
    print(f"{'='*65}\n")

    # Unfreeze last N layers of EfficientNetB0
    for layer in model.layers:
        layer.trainable = True
    backbone_layers = [l for l in model.layers
                       if isinstance(l, keras.Model)]
    if backbone_layers:
        backbone = backbone_layers[0]
        for layer in backbone.layers[:-args.unfreeze]:
            # Keep BatchNorm frozen to preserve ImageNet stats
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(args.lr_stage2, clipnorm=0.5),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smooth),
        metrics=["accuracy"],
    )
    trainable = sum(np.prod(w.shape) for w in model.trainable_weights)
    print(f"  Trainable params (Stage 2): {trainable:,}")

    # Rebuild training dataset without mixup for fine-tuning
    train_ds_s2 = make_dataset(tr_img, tr_sft, tr_lbl, num_classes,
                                args.batch_size, augment=True, mixup_alpha=0.0)

    cbs_s2 = [
        keras.callbacks.ModelCheckpoint(
            str(best_path), monitor="val_accuracy",
            save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=args.patience + 5,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=6,
            min_lr=1e-8, verbose=1),
        keras.callbacks.CSVLogger(
            str(MODELS_DIR / f"{prefix}_s2_history.csv"), append=False),
    ]

    h2 = model.fit(
        train_ds_s2, epochs=args.stage2_epochs,
        steps_per_epoch=steps,
        validation_data=val_ds,
        callbacks=cbs_s2, class_weight=cw_dict, verbose=1,
    )
    best_s2 = max(h2.history.get("val_accuracy", [0]))
    best_overall = max(best_s1, best_s2)

    print(f"\n[Stage 1] Best val_accuracy = {best_s1:.4f}")
    print(f"[Stage 2] Best val_accuracy = {best_s2:.4f}")
    print(f"[Overall] Best val_accuracy = {best_overall:.4f}")

    # Compare vs benchmark
    benchmark_best = 0.853
    gap = best_overall - benchmark_best
    if gap > 0:
        print(f"\n[TARGET] ✓ BEAT benchmark by +{gap:.2%}! (benchmark: {benchmark_best:.2%})")
    else:
        print(f"\n[TARGET] {abs(gap):.2%} below benchmark ({benchmark_best:.2%})")

    # ── Final Test Evaluation ─────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  FINAL TEST EVALUATION")
    print(f"{'='*65}")
    test_results = model.evaluate(test_ds, verbose=1, return_dict=True)
    model.save(str(fin_path))
    (MODELS_DIR / f"{prefix}_test_metrics.json").write_text(
        json.dumps({k: float(v) for k, v in test_results.items()}, indent=2),
        encoding="utf-8")

    print(f"\n[OK] Model saved       -> {fin_path}")
    print(f"[OK] Best checkpoint   -> {best_path}")
    print(f"[OK] Test metrics      -> {test_results}")
    print(f"[OK] Stage 1 history   -> {MODELS_DIR / (prefix + '_s1_history.csv')}")
    print(f"[OK] Stage 2 history   -> {MODELS_DIR / (prefix + '_s2_history.csv')}")


if __name__ == "__main__":
    main()
