"""
EcoTexture AI — Full From-Scratch Training on TrashNet
=======================================================
Mirrors the proven research benchmark Cross_Attention architecture:
  - Custom 4-block CNN + SIFT BoVW (k=300)
  - Transformer fusion (MultiHeadAttention, num_heads=4, key_dim=64)
  - CategoricalCrossentropy with label_smoothing=0.1
  - ReduceLROnPlateau only (no conflicting cosine scheduler)
  - Class-balanced weights
  - 60 epochs with EarlyStopping patience=15

Run:
    python "D:\\EcoTexture AI\\train_from_scratch.py"
    python "D:\\EcoTexture AI\\train_from_scratch.py" --epochs 80 --batch-size 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── CUDA DLL patch ──────────────────────────────────────────
sys.path.insert(0, r"D:\grad project\trial")
try:
    import env_tf_patch
    print("[EcoTexture] Loaded CUDA DLL environment patch.")
except ImportError:
    print("[EcoTexture] env_tf_patch not found, continuing without it.")

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np

import tensorflow as tf

# Limit threads for CPU training to avoid thermal throttle
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)

# ── Keras import (tf_keras preferred for checkpoint compat) ──
try:
    import tf_keras as keras
    from tf_keras import layers, Model, Input
except ImportError:
    try:
        from tensorflow import keras
        from tensorflow.keras import layers, Model, Input
    except ImportError:
        import keras
        from keras import layers, Model, Input

from sklearn.utils.class_weight import compute_class_weight

# ── Paths ────────────────────────────────────────────────────
ECO_AI_ROOT   = Path(r"D:\EcoTexture AI")
MODELS_DIR    = ECO_AI_ROOT / "models"
PROCESSED_DIR = ECO_AI_ROOT / "data" / "processed" / "trashnet"
RAW_DIR       = ECO_AI_ROOT / "data" / "raw" / "trashnet"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ECO_AI_ROOT))
from ecotexture_ai.data import (
    Sample, collect_samples, fit_sift_vocabulary,
    image_to_array, load_split_manifest, save_split_manifest,
    split_samples, sift_histogram,
)

# ── GPU setup ────────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"[Runtime] GPU ready: {len(gpus)} device(s)")
else:
    print("[Runtime] No GPU found — running on CPU (float32)")


# ─────────────────────────────────────────────────────────────
# MODEL — Cross-Attention (identical to benchmark best variant)
# ─────────────────────────────────────────────────────────────
def build_cross_attention_model(num_classes: int, sift_k: int = 300,
                                 img_size: tuple = (224, 224)) -> keras.Model:
    """
    Replicates the benchmark Cross_Attention architecture that achieved
    the best results on MET/COCO in the research paper.
    No pretrained weights — trains from scratch on TrashNet.
    """
    cnn_in   = Input(shape=(*img_size, 3), name="image_input")
    sift_in  = Input(shape=(sift_k,),      name="sift_input")

    # ── CNN Backbone (4-block custom CNN) ──────────────────
    x = layers.Conv2D(32, 3, padding="same", activation="relu", name="conv1")(cnn_in)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling2D(name="pool1")(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling2D(name="pool2")(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.MaxPooling2D(name="pool3")(x)

    x = layers.Conv2D(256, 3, padding="same", activation="relu", name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    cnn_feat = layers.Dense(256, activation="relu", name="cnn_dense")(x)
    cnn_feat = layers.Dropout(0.4, name="cnn_drop")(cnn_feat)

    # ── SIFT Branch ────────────────────────────────────────
    ys = layers.Dense(128, activation="relu", name="sift_dense1")(sift_in)
    ys = layers.BatchNormalization(name="sift_bn")(ys)
    sift_feat = layers.Dense(64, activation="relu", name="sift_dense2")(ys)

    # ── Transformer Fusion (matches benchmark exactly) ─────
    merged = layers.Concatenate(name="concat")([cnn_feat, sift_feat])   # (B, 320)
    merged = layers.Reshape((1, 320), name="reshape")(merged)            # (B, 1, 320)
    attn   = layers.MultiHeadAttention(num_heads=4, key_dim=64,
                                       name="mha")(merged, merged)
    fused  = layers.LayerNormalization(name="ln")(attn + merged)
    fused  = layers.Flatten(name="flatten")(fused)

    # ── Classification Head ────────────────────────────────
    z   = layers.Dense(128, activation="relu", name="head_dense")(fused)
    z   = layers.Dropout(0.4, name="head_drop")(z)
    out = layers.Dense(num_classes, activation="softmax",
                       dtype="float32", name="output")(z)

    return Model(inputs=[cnn_in, sift_in], outputs=out, name="Cross_Attention_TrashNet")


# ─────────────────────────────────────────────────────────────
# DATASET — Pre-cached numpy arrays (fast training)
# ─────────────────────────────────────────────────────────────
def precompute_split(samples: list, class_to_id: dict,
                     centers: np.ndarray, img_size: tuple,
                     split_name: str) -> tuple:
    """Load images and SIFT histograms into memory for a given split."""
    cache_dir = MODELS_DIR / "cache"
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{split_name}_arrays.npz"

    if cache_file.exists():
        print(f"  [Cache] Loading {split_name} from {cache_file.name} ...")
        d = np.load(cache_file)
        return d["images"], d["sifts"], d["labels"]

    print(f"  [SIFT] Pre-computing {split_name}: {len(samples)} samples ...")
    t0 = time.time()
    images, sifts, labels = [], [], []
    vocab_size = centers.shape[0]

    for i, s in enumerate(samples):
        try:
            img  = image_to_array(s.path, img_size)
            hist = sift_histogram(s.path, centers, vocab_size)
            images.append(img)
            sifts.append(hist)
            labels.append(class_to_id[s.label])
        except Exception as e:
            print(f"\n  [Skip] {s.path.name}: {e}")
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(samples)} done ...", flush=True)

    images = np.asarray(images, dtype=np.float32)
    sifts  = np.asarray(sifts,  dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    np.savez_compressed(str(cache_file), images=images, sifts=sifts, labels=labels)
    print(f"  [SIFT] Done in {time.time()-t0:.1f}s — saved to cache.")
    return images, sifts, labels


def make_tf_dataset(images, sifts, labels, num_classes,
                    batch_size, augment=False, shuffle=True) -> tf.data.Dataset:
    """Build a tf.data pipeline from pre-cached arrays."""
    # One-hot for label smoothing CategoricalCrossentropy
    labels_oh = np.eye(num_classes, dtype=np.float32)[labels]

    with tf.device("/CPU:0"):
        ds = tf.data.Dataset.from_tensor_slices(
            ((images.astype(np.float32), sifts.astype(np.float32)),
             labels_oh)
        )

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(images), 2000),
                        reshuffle_each_iteration=True)

    if augment:
        def aug_fn(x, y):
            img, sift = x[0], x[1]
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.15)
            img = tf.image.random_contrast(img, 0.8, 1.2)
            img = tf.clip_by_value(img, 0.0, 1.0)
            return (img, sift), y
        ds = ds.map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.repeat()

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int,   default=60)
    parser.add_argument("--batch-size",  type=int,   default=16)
    parser.add_argument("--img-size",    type=int,   default=224)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--label-smooth",type=float, default=0.1)
    parser.add_argument("--patience",    type=int,   default=15)
    parser.add_argument("--sift-k",      type=int,   default=300)
    parser.add_argument("--no-cache",    action="store_true",
                        help="Force recompute SIFT even if cache exists")
    args = parser.parse_args()

    img_size   = (args.img_size, args.img_size)
    prefix     = "ecotexture_scratch"
    best_path  = MODELS_DIR / f"{prefix}_best.keras"
    final_path = MODELS_DIR / f"{prefix}_final.keras"

    # ── Clear cache if requested ────────────────────────────
    if args.no_cache:
        for f in (MODELS_DIR / "cache").glob("*_arrays.npz"):
            f.unlink()
            print(f"[Cache] Removed {f.name}")

    # ── Data Splits ─────────────────────────────────────────
    train_m = PROCESSED_DIR / "train.json"
    val_m   = PROCESSED_DIR / "val.json"
    test_m  = PROCESSED_DIR / "test.json"

    if train_m.exists() and val_m.exists() and test_m.exists():
        train_s = load_split_manifest(train_m)
        val_s   = load_split_manifest(val_m)
        test_s  = load_split_manifest(test_m)
        print(f"[Data] Loaded manifests: {len(train_s)} train / {len(val_s)} val / {len(test_s)} test")
    else:
        print("[Data] Manifests not found — scanning raw data ...")
        samples = collect_samples(RAW_DIR)
        if not samples:
            raise RuntimeError(f"No images found in {RAW_DIR}")
        train_s, val_s, test_s = split_samples(samples)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        save_split_manifest(train_s, train_m)
        save_split_manifest(val_s,   val_m)
        save_split_manifest(test_s,  test_m)
        print(f"[Data] Split: {len(train_s)} train / {len(val_s)} val / {len(test_s)} test")

    # ── Class map ────────────────────────────────────────────
    all_s        = train_s + val_s + test_s
    class_to_id  = {l: i for i, l in enumerate(sorted({s.label for s in all_s}))}
    num_classes  = len(class_to_id)
    (MODELS_DIR / "class_to_id.json").write_text(
        json.dumps(class_to_id, indent=2), encoding="utf-8")
    print(f"[Data] Classes ({num_classes}): {list(class_to_id.keys())}")

    # ── SIFT Vocabulary ──────────────────────────────────────
    vocab_path = MODELS_DIR / "sift_centers.npy"
    if vocab_path.exists():
        centers = np.load(vocab_path)
        print(f"[SIFT] Loaded vocabulary: {centers.shape}")
    else:
        # Check trial directory for existing vocabulary
        trial_vocab = Path(r"D:\grad project\trial\hybrid\meta\sift_kmeans_centers.npy")
        if trial_vocab.exists():
            import shutil
            shutil.copy2(trial_vocab, vocab_path)
            centers = np.load(vocab_path)
            print(f"[SIFT] Copied vocab from trial: {centers.shape}")
        else:
            print("[SIFT] Fitting new vocabulary ...")
            centers = fit_sift_vocabulary(train_s, vocab_size=args.sift_k, max_images=30)
            np.save(vocab_path, centers)
            print(f"[SIFT] Vocabulary fitted: {centers.shape}")

    # ── Precompute SIFT + Image Arrays ───────────────────────
    print("\n[Pipeline] Pre-computing all splits (cached after first run) ...")
    tr_img, tr_sft, tr_lbl = precompute_split(train_s, class_to_id, centers, img_size, "train")
    va_img, va_sft, va_lbl = precompute_split(val_s,   class_to_id, centers, img_size, "val")
    te_img, te_sft, te_lbl = precompute_split(test_s,  class_to_id, centers, img_size, "test")

    # ── Class Weights ────────────────────────────────────────
    cw_arr  = compute_class_weight("balanced", classes=np.unique(tr_lbl), y=tr_lbl)
    cw_dict = {int(c): float(w) for c, w in zip(np.unique(tr_lbl), cw_arr)}
    print(f"[Training] Class weights: { {k: f'{v:.2f}' for k,v in cw_dict.items()} }")

    # ── TF Datasets ──────────────────────────────────────────
    steps_per_epoch = max(1, len(train_s) // args.batch_size)
    train_ds = make_tf_dataset(tr_img, tr_sft, tr_lbl, num_classes,
                               args.batch_size, augment=True, shuffle=True)
    val_ds   = make_tf_dataset(va_img, va_sft, va_lbl, num_classes,
                               args.batch_size, augment=False, shuffle=False)
    test_ds  = make_tf_dataset(te_img, te_sft, te_lbl, num_classes,
                               args.batch_size, augment=False, shuffle=False)

    # ── Build Model ──────────────────────────────────────────
    model = build_cross_attention_model(num_classes, sift_k=centers.shape[0],
                                        img_size=img_size)
    model.summary(line_length=90)

    # ── Compile — Label smoothing CategoricalCrossentropy ────
    # (matches benchmark: CategoricalCrossentropy(label_smoothing=0.1))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smooth),
        metrics=["accuracy"],
    )

    # ── Callbacks — NO cosine scheduler, only ReduceLROnPlateau ──
    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            str(best_path), monitor="val_accuracy",
            save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=args.patience,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=6,
            min_lr=1e-7, verbose=1),
        keras.callbacks.CSVLogger(
            str(MODELS_DIR / f"{prefix}_history.csv"), append=False),
    ]

    print(f"\n{'='*60}")
    print(f"  TRAINING FROM SCRATCH — Cross_Attention")
    print(f"  Dataset: TrashNet ({num_classes} classes, {len(train_s)} train samples)")
    print(f"  Epochs: {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    print(f"  Label smoothing: {args.label_smooth}  |  EarlyStop patience: {args.patience}")
    print(f"{'='*60}\n")

    history = model.fit(
        train_ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        callbacks=callbacks_list,
        class_weight=cw_dict,
        verbose=1,
    )

    best_val_acc = max(history.history.get("val_accuracy", [0]))
    print(f"\n[Training] Best val_accuracy = {best_val_acc:.4f}")

    # ── Final Test Evaluation ────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FINAL TEST EVALUATION")
    print(f"{'='*60}")
    test_results = model.evaluate(test_ds, verbose=1, return_dict=True)

    # Save final model
    model.save(str(final_path))
    metrics_path = MODELS_DIR / f"{prefix}_test_metrics.json"
    metrics_path.write_text(
        json.dumps({k: float(v) for k, v in test_results.items()}, indent=2),
        encoding="utf-8")

    print(f"\n[OK] Model saved -> {final_path}")
    print(f"[OK] Best checkpoint -> {best_path}")
    print(f"[OK] Test metrics -> {test_results}")
    print(f"[OK] History -> {MODELS_DIR / (prefix + '_history.csv')}")


if __name__ == "__main__":
    main()
