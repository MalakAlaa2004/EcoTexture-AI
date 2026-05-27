"""
EcoTexture AI — High-Performance Fine-Tuning Pipeline
======================================================
Features:
  - Parallel tf.data pipeline with SIFT pre-computation and CPU pinning.
  - Aligned cross-attention weights (d_model=256, num_heads=4) to prevent layer weight mismatches.
  - Cosine Annealing learning rate schedule.
  - Memory-safe callbacks (TensorBoard histograms disabled) to prevent thermal shutdowns.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

# ── Load CUDA DLL path injections ──
sys.path.insert(0, r"D:\grad project\trial")
try:
    import env_tf_patch
    print("[EcoTexture] Loaded CUDA DLL environment patch.")
except ImportError:
    print("[EcoTexture] Warning: env_tf_patch.py not found in trial folder.")

import numpy as np
import tensorflow as tf

try:
    import tf_keras as keras
    from tf_keras import layers, models, callbacks
except ImportError:
    try:
        from tensorflow import keras
        from tensorflow.keras import layers, models, callbacks
    except ImportError:
        import keras
        from keras import layers, models, callbacks

from sklearn.utils.class_weight import compute_class_weight

from ecotexture_ai.config import (
    EXISTING_SIFT_CENTERS,
    COCO_TEACHER_CKPTS,
    MET_TEACHER_CKPTS,
    EXISTING_HYBRID_CKPTS,
)

# ── Base configurations ──
BATCH_SIZE = 8
EPOCHS_STAGE1 = 10
EPOCHS_STAGE2 = 15
LR_STAGE1 = 1e-3
LR_STAGE2 = 1e-5
MODELS_DIR = Path(r"D:\EcoTexture AI\models")
PROCESSED_DIR = Path(r"D:\EcoTexture AI\data\processed")
RAW_DIR = Path(r"D:\EcoTexture AI\data\raw")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LEGACY_MODELS_DIR = Path(r"D:\EcoTexture A\models")
LEGACY_ECOTEXTURE_CKPTS = [
    MODELS_DIR / "contrastive_transfer_final.h5",
    MODELS_DIR / "contrastive_transfer_best.h5",
    MODELS_DIR / "hybrid_sift_final.h5",
    MODELS_DIR / "hybrid_sift_best.h5",
    LEGACY_MODELS_DIR / "contrastive_transfer_final.keras",
    LEGACY_MODELS_DIR / "contrastive_transfer_best.keras",
    LEGACY_MODELS_DIR / "ecotexture_hybrid_final.keras",
    LEGACY_MODELS_DIR / "ecotexture_hybrid_best.keras",
]
ALL_TEACHER_CKPTS = LEGACY_ECOTEXTURE_CKPTS + COCO_TEACHER_CKPTS + MET_TEACHER_CKPTS + EXISTING_HYBRID_CKPTS

# ── Dynamic Runtime Configuration ──
def configure_runtime() -> None:
    warnings.filterwarnings("ignore")
    try:
        physical_gpus = tf.config.list_physical_devices("GPU")
        if physical_gpus:
            for gpu in physical_gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # Safe policy if GPU is present
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("[Runtime] GPU initialized with mixed_float16.")
        else:
            tf.keras.mixed_precision.set_global_policy("float32")
            print("[Runtime] No GPU found. Falling back to float32.")
    except Exception as exc:
        print(f"[Runtime] Precision policy setup skipped: {exc}")

    try:
        # Prevent threading from locking up laptop CPU
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except Exception as exc:
        print(f"[Runtime] Thread tuning skipped: {exc}")

configure_runtime()

from ecotexture_ai.model import build_ecotexture_model, build_contrastive_transfer_model, unfreeze_last_blocks
from ecotexture_ai.data import (
    Sample, collect_samples, fit_sift_vocabulary,
    image_to_array, load_split_manifest, save_split_manifest,
    split_samples, sift_histogram,
)

# ── Severe but fast data augmentation ──
def build_augmentation() -> keras.Sequential:
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15, fill_mode="reflect"),
            layers.RandomZoom(0.10, fill_mode="reflect"),
            layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )

# ── Optimized Precached TF.Data Pipeline ──
def create_tfdata_pipeline(samples, class_to_id, centers, img_size, batch_size, augment=True):
    n_cls = len(class_to_id)
    vocab_size = centers.shape[0]

    # Pre-load/pre-compute SIFT and image arrays to avoid CPU starvation during training epochs
    print(f"  [Pipeline] Pre-loading and pre-computing SIFT histograms for {len(samples)} samples...")
    t0 = time.time()
    images = []
    sifts = []
    labels = []
    
    for s in samples:
        try:
            img = image_to_array(s.path, img_size)
            hist = sift_histogram(s.path, centers, vocab_size)
            images.append(img)
            sifts.append(hist)
            labels.append(class_to_id[s.label])
        except Exception as e:
            print(f"  [Pipeline] Skipping corrupt sample {s.path}: {e}")

    images = np.asarray(images, dtype=np.float32)
    sifts = np.asarray(sifts, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    print(f"  [Pipeline] Done in {time.time() - t0:.1f}s.")

    # Create TF Dataset
    dataset = tf.data.Dataset.from_tensor_slices(((images, sifts), labels))
    
    if augment:
        aug_fn = build_augmentation()
        dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
        # Apply fast augmentation to the image portion only
        dataset = dataset.map(
            lambda x, y: ((aug_fn(tf.expand_dims(x[0], 0), training=True)[0], x[1]), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# ── Cosine LR Scheduler Callback ──
def get_cosine_lr_callback(initial_lr: float, total_epochs: int) -> callbacks.LearningRateScheduler:
    def lr_fn(epoch):
        cos = np.cos(np.pi * epoch / total_epochs)
        return float(initial_lr * 0.5 * (1.0 + cos))
    return callbacks.LearningRateScheduler(lr_fn, verbose=1)

def main():
    parser = argparse.ArgumentParser(description="EcoTexture AI fine-tuning launcher")
    parser.add_argument("--raw-root", type=Path, default=RAW_DIR / "trashnet")
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_DIR / "trashnet")
    parser.add_argument("--epochs-stage1", type=int, default=EPOCHS_STAGE1)
    parser.add_argument("--epochs-stage2", type=int, default=EPOCHS_STAGE2)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--img-size", type=int, default=192)
    parser.add_argument("--backbone", type=str, default="efficientnetb0", choices=["mobilenetv3", "efficientnetb0", "custom_cnn"])
    parser.add_argument("--technique", choices=["hybrid_sift", "contrastive_transfer"], default="contrastive_transfer")
    parser.add_argument("--contrastive-checkpoint", type=str, default=None)
    parser.add_argument("--unfreeze-n", type=int, default=60)
    parser.add_argument("--target-acc", type=float, default=0.97)
    args = parser.parse_args()

    img_size = (args.img_size, args.img_size)

    # ── Dataset Loading ──
    train_m = args.processed_root / "train.json"
    val_m = args.processed_root / "val.json"
    test_m = args.processed_root / "test.json"
    
    if train_m.exists() and val_m.exists() and test_m.exists():
        train_s = load_split_manifest(train_m)
        val_s   = load_split_manifest(val_m)
        test_s  = load_split_manifest(test_m)
    else:
        samples = collect_samples(args.raw_root)
        if not samples:
            raise RuntimeError(f"No images found in {args.raw_root}")
        train_s, val_s, test_s = split_samples(samples)
        args.processed_root.mkdir(parents=True, exist_ok=True)
        save_split_manifest(train_s, train_m)
        save_split_manifest(val_s,   val_m)
        save_split_manifest(test_s,  test_m)

    all_s = train_s + val_s + test_s
    class_to_id = {l: i for i, l in enumerate(sorted({s.label for s in all_s}))}
    (MODELS_DIR / "class_to_id.json").write_text(
        json.dumps(class_to_id, indent=2), encoding="utf-8"
    )
    print(f"Classes ({len(class_to_id)}): {list(class_to_id.keys())}")

    # ── Vocabulary setup ──
    vocab_path = MODELS_DIR / "sift_centers.npy"
    centers = None
    if vocab_path.exists():
        centers = np.load(vocab_path)
    else:
        for c in EXISTING_SIFT_CENTERS:
            if c.exists():
                centers = np.load(c)
                break
    if centers is None:
        print("Fitting SIFT vocabulary ...")
        centers = fit_sift_vocabulary(train_s, vocab_size=300, max_images=30)
        np.save(vocab_path, centers)
    print(f"SIFT vocabulary: {centers.shape}")

    # Class weight balances
    y_train = np.asarray([class_to_id[s.label] for s in train_s], dtype=np.int64)
    class_weights_arr = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    class_weights = {int(cls): float(weight) for cls, weight in zip(np.unique(y_train), class_weights_arr)}

    # ── Pipelines ──
    train_dataset = create_tfdata_pipeline(train_s, class_to_id, centers, img_size, args.batch_size, augment=True)
    val_dataset = create_tfdata_pipeline(val_s, class_to_id, centers, img_size, args.batch_size, augment=False)
    test_dataset = create_tfdata_pipeline(test_s, class_to_id, centers, img_size, args.batch_size, augment=False)

    # ── Source Checkpoint Auto-discovery ──
    ckpt = args.contrastive_checkpoint
    if not ckpt:
        for candidate in ALL_TEACHER_CKPTS:
            if candidate.exists():
                source = "COCO-mixed ConvNeXt" if "coco" in str(candidate).lower() else "MET ConvNeXt" if "met" in str(candidate).lower() else "TimeLens contrastive"
                print(f"\n[OK] Transfer base: {source}\n  -> {candidate}")
                ckpt = str(candidate)
                break

    # ── Model Construction ──
    n_classes = len(class_to_id)
    if args.technique == "contrastive_transfer":
        model = build_contrastive_transfer_model(
            num_classes=n_classes,
            checkpoint_path=ckpt,
            sift_k=int(centers.shape[0]),
            backbone_name=args.backbone,
            img_size=img_size,
        )
        prefix = "ecotexture_coco_met"
    else:
        model = build_ecotexture_model(
            num_classes=n_classes,
            backbone_name=args.backbone,
            img_size=img_size,
            sift_k=int(centers.shape[0]),
        )
        prefix = "ecotexture_hybrid"

    best_path = MODELS_DIR / f"{prefix}_best.keras"
    final_path = MODELS_DIR / f"{prefix}_final.keras"

    # ── Target stop logic ──
    def stop_at_target(epoch, logs=None):
        logs = logs or {}
        if float(logs.get("val_accuracy", 0.0)) >= args.target_acc:
            print(f"[OK] Target validation accuracy {args.target_acc:.2f} reached; stopping early.")
            model.stop_training = True

    target_stop = callbacks.LambdaCallback(on_epoch_end=stop_at_target)

    # Balanced learning callbacks
    callbacks_list = [
        callbacks.ModelCheckpoint(str(best_path), monitor="val_accuracy", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
        callbacks.TensorBoard(log_dir=str(MODELS_DIR / "logs" / prefix), histogram_freq=0), # Disable histograms for performance
        target_stop,
    ]

    # Calculate steps per epoch
    steps_per_epoch = len(train_s) // args.batch_size

    # Compile with SparseCategoricalCrossentropy
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR_STAGE1),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    print(f"\n{'='*55}\n  STAGE 1 - Head training (backbone frozen)\n{'='*55}")
    # Inject Cosine LR specifically tuned for Stage 1 epochs
    s1_callbacks = callbacks_list + [get_cosine_lr_callback(LR_STAGE1, args.epochs_stage1)]
    model.fit(
        train_dataset,
        epochs=args.epochs_stage1,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        callbacks=s1_callbacks,
        class_weight=class_weights,
    )

    print(f"\n{'='*55}\n  STAGE 2 - Fine-tuning last {args.unfreeze_n} layers\n{'='*55}")
    unfreeze_last_blocks(model, train_last_n=args.unfreeze_n, learning_rate=LR_STAGE2)
    s2_callbacks = callbacks_list + [get_cosine_lr_callback(LR_STAGE2, args.epochs_stage2)]
    model.fit(
        train_dataset,
        epochs=args.epochs_stage2,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        callbacks=s2_callbacks,
        class_weight=class_weights,
    )

    print(f"\n{'='*55}\n  FINAL TEST EVALUATION\n{'='*55}")
    test_results = model.evaluate(test_dataset, verbose=1, return_dict=True)
    model.save(str(final_path))
    (MODELS_DIR / f"{prefix}_test_metrics.json").write_text(
        json.dumps({k: float(v) for k, v in test_results.items()}, indent=2), encoding="utf-8"
    )
    print(f"\n[OK] Model saved -> {final_path}")
    print(f"[OK] Test metrics -> {test_results}")

if __name__ == "__main__":
    main()
