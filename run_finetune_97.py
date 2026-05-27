"""
EcoTexture AI — Aggressive Fine-Tuning to 97%+ Accuracy
=========================================================
Strategy:
  Round 1 : COCO ConvNeXt_Tiny transfer, EfficientNetB0 head-train (Stage 1+2)
  Round 2 : MET ConvNeXt_Tiny transfer if Round 1 < 97%
  Round 3 : Extended epochs + cosine annealing + mixup if still < 97%
  Round 4 : Full unfreeze with very low LR until convergence

Run:
  python run_finetune_97.py
  python run_finetune_97.py --target-acc 0.95   # lower target for quick test
"""
from __future__ import annotations

import sys
sys.path.insert(0, r"D:\grad project\trial")
import env_tf_patch

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
try:
    import tf_keras as keras
except ImportError:
    try:
        from tensorflow import keras
    except ImportError:
        import keras

# ── Force TF_USE_LEGACY_KERAS for checkpoint compatibility ──
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

# ── GPU memory growth ─────────────────────────────────────
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

# ── Paths ─────────────────────────────────────────────────
ECO_AI_ROOT   = Path(r"D:\EcoTexture AI")
MODELS_DIR    = ECO_AI_ROOT / "models"
PROCESSED_DIR = ECO_AI_ROOT / "data" / "processed" / "trashnet"
RAW_DIR       = ECO_AI_ROOT / "data" / "raw" / "trashnet"
REPORTS_DIR   = ECO_AI_ROOT / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Source checkpoints (COCO → MET Cross-Attention models) ─────
TRIAL_ROOT = Path(r"D:\grad project\trial")
CHECKPOINT_ROUNDS = [
    # Round 1: COCO Cross Attention (seed 42)
    {
        "name":     "COCO_CrossAttention_seed42",
        "ckpt":     TRIAL_ROOT / "benchmark_experiments/models/coco_mixed/Cross_Attention_seed42_best.keras",
        "backbone": "custom_cnn",
    },
    # Round 2: MET Cross Attention (seed 42)
    {
        "name":     "MET_CrossAttention_seed42",
        "ckpt":     TRIAL_ROOT / "benchmark_experiments/models/met/Cross_Attention_seed42_best.keras",
        "backbone": "custom_cnn",
    },
    # Round 3: COCO Cross Attention (seed 43)
    {
        "name":     "COCO_CrossAttention_seed43",
        "ckpt":     TRIAL_ROOT / "benchmark_experiments/models/coco_mixed/Cross_Attention_seed43_best.keras",
        "backbone": "custom_cnn",
    },
    # Round 4: MET Cross Attention (seed 43)
    {
        "name":     "MET_CrossAttention_seed43",
        "ckpt":     TRIAL_ROOT / "benchmark_experiments/models/met/Cross_Attention_seed43_best.keras",
        "backbone": "custom_cnn",
    },
    # Round 5: COCO Cross Attention (seed 44)
    {
        "name":     "COCO_CrossAttention_seed44",
        "ckpt":     TRIAL_ROOT / "benchmark_experiments/models/coco_mixed/Cross_Attention_seed44_best.keras",
        "backbone": "custom_cnn",
    },
    # Round 6: MET Cross Attention (seed 44)
    {
        "name":     "MET_CrossAttention_seed44",
        "ckpt":     TRIAL_ROOT / "benchmark_experiments/models/met/Cross_Attention_seed44_best.keras",
        "backbone": "custom_cnn",
    },
]


# ── Import EcoTexture modules ──────────────────────────────
sys.path.insert(0, str(ECO_AI_ROOT))
from ecotexture_ai.model import (
    build_ecotexture_model,
    build_contrastive_transfer_model,
    unfreeze_last_blocks,
    _get_optimizer,
)
from ecotexture_ai.data import (
    Sample, collect_samples, fit_sift_vocabulary,
    image_to_array, load_split_manifest, save_split_manifest,
    split_samples, sift_histogram,
)


# ─────────────────────────────────────────────────────────────
# AUGMENTATION  (aggressive for 97%+)
# ─────────────────────────────────────────────────────────────
def build_augmentation() -> keras.Sequential:
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal_and_vertical"),
        keras.layers.RandomRotation(0.30),
        keras.layers.RandomZoom(0.20),
        keras.layers.RandomContrast(0.25),
        keras.layers.RandomBrightness(0.25),
        keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    ], name="augmentation")


# ─────────────────────────────────────────────────────────────
# MIXUP SEQUENCE
# ─────────────────────────────────────────────────────────────
class MixupHybridSequence(keras.utils.Sequence):
    """Hybrid CNN+SIFT sequence with optional Mixup augmentation."""

    def __init__(self, samples, class_to_id, centers, num_classes,
                 batch_size=16, shuffle=True, augment=False, mixup_alpha=0.0):
        self.samples      = list(samples)
        self.class_to_id  = class_to_id
        self.centers      = centers
        self.num_classes  = num_classes
        self.batch_size   = batch_size
        self.shuffle      = shuffle
        self.augment      = augment
        self.mixup_alpha  = mixup_alpha
        self.aug_fn       = build_augmentation() if augment else None
        self.indices      = np.arange(len(self.samples))

        print(f"  Precomputing SIFT for {len(self.samples)} samples in parallel …", flush=True)
        self.sift_cache = {}
        from multiprocessing.pool import ThreadPool
        def _compute(s):
            return s.path, sift_histogram(s.path, self.centers, self.centers.shape[0])
        with ThreadPool() as pool:
            results = pool.map(_compute, self.samples)
        for path, hist in results:
            self.sift_cache[path] = hist
        print(f"    ✓ Precomputed SIFT for {len(self.samples)} samples.", flush=True)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

    def __getitem__(self, index):
        batch_idx = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.samples[i] for i in batch_idx]

        images, sifts, labels = [], [], []
        for s in batch:
            img = image_to_array(s.path, (224, 224))
            if self.augment and self.aug_fn is not None:
                img = self.aug_fn(tf.expand_dims(img, 0), training=True)[0].numpy()
            images.append(img)
            sifts.append(self.sift_cache[s.path])
            labels.append(self.class_to_id[s.label])

        images = np.asarray(images, dtype=np.float32)
        sifts  = np.asarray(sifts,  dtype=np.float32)
        labels = np.asarray(labels,  dtype=np.int64)

        # One-hot for mixup
        labels_oh = np.eye(self.num_classes, dtype=np.float32)[labels]

        if self.mixup_alpha > 0.0 and self.augment:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            idx2 = np.random.permutation(len(images))
            images    = lam * images + (1 - lam) * images[idx2]
            sifts     = lam * sifts  + (1 - lam) * sifts[idx2]
            labels_oh = lam * labels_oh + (1 - lam) * labels_oh[idx2]
            return (images, sifts), labels_oh

        return (images, sifts), labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.default_rng().shuffle(self.indices)


# ─────────────────────────────────────────────────────────────
# COSINE LR SCHEDULE
# ─────────────────────────────────────────────────────────────
def cosine_schedule(initial_lr: float, epochs: int) -> keras.callbacks.LearningRateScheduler:
    def lr_fn(epoch):
        cos = np.cos(np.pi * epoch / epochs)
        return float(initial_lr * 0.5 * (1.0 + cos))
    return keras.callbacks.LearningRateScheduler(lr_fn, verbose=0)


# ─────────────────────────────────────────────────────────────
# TRAINING ROUND
# ─────────────────────────────────────────────────────────────
def train_one_round(
    round_cfg: dict,
    train_s, val_s, class_to_id, centers,
    epochs_s1: int, epochs_s2: int,
    batch_size: int,
    mixup_alpha: float,
    unfreeze_n: int,
    label_smoothing: float,
    loss_fn,
) -> tuple[keras.Model, float]:
    """Train one full 2-stage fine-tuning round. Returns (model, best_val_acc)."""

    name     = round_cfg["name"]
    ckpt     = round_cfg["ckpt"]
    backbone = round_cfg["backbone"]
    n_cls    = len(class_to_id)
    use_mixup = mixup_alpha > 0.0

    print(f"\n{'='*60}")
    print(f"  ROUND: {name}")
    print(f"  Checkpoint: {Path(str(ckpt)).name if Path(str(ckpt)).exists() else '(missing — ImageNet only)'}")
    print(f"  Mixup α={mixup_alpha:.2f}  Unfreeze={unfreeze_n}  Smoothing={label_smoothing:.2f}")
    print(f"{'='*60}\n")

    ckpt_path = str(ckpt) if Path(str(ckpt)).exists() else None

    model = build_contrastive_transfer_model(
        num_classes=n_cls,
        checkpoint_path=ckpt_path,
        sift_k=int(centers.shape[0]),
        backbone_name=backbone,
    )

    # Recompile with label smoothing for mixup compatibility
    model.compile(
        optimizer=_get_optimizer(learning_rate=1e-3, weight_decay=1e-4),
        loss=loss_fn,
        metrics=["accuracy"],
    )

    best_path = MODELS_DIR / f"ecotexture_{name}_best.keras"

    train_seq = MixupHybridSequence(
        train_s, class_to_id, centers, n_cls,
        batch_size=batch_size, augment=True, mixup_alpha=mixup_alpha,
    )
    val_seq = MixupHybridSequence(
        val_s, class_to_id, centers, n_cls,
        batch_size=batch_size, shuffle=False, mixup_alpha=0.0,
    )

    callbacks_s1 = [
        keras.callbacks.ModelCheckpoint(str(best_path), monitor="val_accuracy",
                                         save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6,
                                       restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.4,
                                           patience=3, min_lr=1e-7, verbose=1),
        cosine_schedule(1e-3, epochs_s1),
        keras.callbacks.TensorBoard(log_dir=str(MODELS_DIR / "logs" / name), histogram_freq=0),
    ]

    print(f"  ── Stage 1: Head training ({epochs_s1} epochs) ──")
    h1 = model.fit(train_seq, validation_data=val_seq,
                   epochs=epochs_s1, callbacks=callbacks_s1)
    best_s1 = max(h1.history.get("val_accuracy", [0]))
    print(f"\n  ✓ Stage 1 best val_acc = {best_s1:.4f}")

    print(f"\n  ── Stage 2: Fine-tuning last {unfreeze_n} layers ({epochs_s2} epochs) ──")
    unfreeze_last_blocks(model, train_last_n=unfreeze_n, learning_rate=1e-5)

    # Recompile after unfreeze (keep label-smoothing loss)
    model.compile(
        optimizer=_get_optimizer(learning_rate=1e-5, weight_decay=1e-5),
        loss=loss_fn,
        metrics=["accuracy"],
    )

    callbacks_s2 = [
        keras.callbacks.ModelCheckpoint(str(best_path), monitor="val_accuracy",
                                         save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8,
                                       restore_best_weights=True, verbose=1),
        cosine_schedule(1e-5, epochs_s2),
        keras.callbacks.TensorBoard(log_dir=str(MODELS_DIR / "logs" / name), histogram_freq=0),
    ]

    h2 = model.fit(train_seq, validation_data=val_seq,
                   epochs=epochs_s2, callbacks=callbacks_s2)
    best_s2 = max(h2.history.get("val_accuracy", [0]))
    best_overall = max(best_s1, best_s2)
    print(f"\n  ✓ Stage 2 best val_acc = {best_s2:.4f}")
    print(f"  ★ Round overall best   = {best_overall:.4f}")

    model.save(str(MODELS_DIR / f"ecotexture_{name}_final.keras"))
    return model, best_overall


# ─────────────────────────────────────────────────────────────
# EVALUATE ON TEST SET
# ─────────────────────────────────────────────────────────────
def evaluate_model(model, test_s, class_to_id, centers, batch_size=16) -> dict:
    n_cls = len(class_to_id)
    seq = MixupHybridSequence(test_s, class_to_id, centers, n_cls,
                               batch_size=batch_size, shuffle=False, mixup_alpha=0.0)
    results = model.evaluate(seq, verbose=1, return_dict=True)
    return results


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="EcoTexture AI — aggressive fine-tuning to 97%+")
    parser.add_argument("--target-acc",    type=float, default=0.97,   help="Stop when val_acc >= this")
    parser.add_argument("--batch-size",    type=int,   default=16)
    parser.add_argument("--epochs-s1",    type=int,   default=15,      help="Stage 1 max epochs per round")
    parser.add_argument("--epochs-s2",    type=int,   default=12,      help="Stage 2 max epochs per round")
    parser.add_argument("--mixup-alpha",   type=float, default=0.2)
    parser.add_argument("--label-smooth",  type=float, default=0.1)
    parser.add_argument("--unfreeze-n",    type=int,   default=30)
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  EcoTexture AI — Fine-Tuning to {:.0f}%+".format(args.target_acc * 100))
    print("="*60)

    # ── Data ────────────────────────────────────────────────
    train_manifest = PROCESSED_DIR / "train.json"
    val_manifest   = PROCESSED_DIR / "val.json"
    test_manifest  = PROCESSED_DIR / "test.json"

    if train_manifest.exists() and val_manifest.exists():
        train_s = load_split_manifest(train_manifest)
        val_s   = load_split_manifest(val_manifest)
    else:
        print("[!] Manifests not found — scanning raw data …")
        samples = collect_samples(RAW_DIR)
        if not samples:
            raise RuntimeError(f"No images found in {RAW_DIR}. Run bootstrap_and_train.py first.")
        train_s, val_s, test_s_new = split_samples(samples)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        save_split_manifest(train_s, train_manifest)
        save_split_manifest(val_s,   val_manifest)
        if test_s_new:
            save_split_manifest(test_s_new, test_manifest)

    test_s = load_split_manifest(test_manifest) if test_manifest.exists() else val_s

    all_s = train_s + val_s
    class_to_id = {l: i for i, l in enumerate(sorted({s.label for s in all_s}))}
    (MODELS_DIR / "class_to_id.json").write_text(
        json.dumps(class_to_id, indent=2), encoding="utf-8"
    )
    print(f"\n  Classes ({len(class_to_id)}): {sorted(class_to_id.keys())}")
    print(f"  Train: {len(train_s)}  Val: {len(val_s)}  Test: {len(test_s)}")

    # ── SIFT vocabulary ──────────────────────────────────────
    vocab_path = MODELS_DIR / "sift_centers.npy"
    if vocab_path.exists():
        centers = np.load(vocab_path)
        print(f"  Loaded SIFT vocab: {centers.shape}")
    else:
        print("  Fitting SIFT vocabulary from training data …")
        centers = fit_sift_vocabulary(train_s, vocab_size=300, max_images=30)
        np.save(vocab_path, centers)
    print(f"  SIFT vocab shape: {centers.shape}")

    # ── Loss function ────────────────────────────────────────
    # Use CategoricalCrossentropy (with label smoothing) — pairs with one-hot labels from mixup
    loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smooth)

    # ── Main training loop ──────────────────────────────────
    best_model       = None
    best_acc_global  = 0.0
    best_round_name  = None
    history_log      = []

    for round_idx, round_cfg in enumerate(CHECKPOINT_ROUNDS):
        t_start = time.time()
        model, val_acc = train_one_round(
            round_cfg, train_s, val_s, class_to_id, centers,
            epochs_s1=args.epochs_s1,
            epochs_s2=args.epochs_s2,
            batch_size=args.batch_size,
            mixup_alpha=args.mixup_alpha,
            unfreeze_n=args.unfreeze_n,
            label_smoothing=args.label_smooth,
            loss_fn=loss_fn,
        )
        elapsed = time.time() - t_start
        history_log.append({
            "round": round_cfg["name"],
            "val_accuracy": round(val_acc, 4),
            "elapsed_min": round(elapsed / 60, 1),
        })

        if val_acc > best_acc_global:
            best_acc_global = val_acc
            best_model      = model
            best_round_name = round_cfg["name"]
            # Save the current best as the canonical model
            best_model.save(str(MODELS_DIR / "ecotexture_best_overall.keras"))
            print(f"\n  ★ New global best: {best_acc_global:.4f} ({best_round_name})")

        print(f"\n  ── Progress after Round {round_idx + 1} ──")
        print(f"     Current best: {best_acc_global:.4f}  Target: {args.target_acc:.4f}")

        if best_acc_global >= args.target_acc:
            print(f"\n  🎉 TARGET REACHED! {best_acc_global:.4f} >= {args.target_acc:.4f}")
            break

        # Escalate for next round if still below target
        if round_idx + 1 < len(CHECKPOINT_ROUNDS):
            # Extend epochs each round
            args.epochs_s1 = min(args.epochs_s1 + 5, 30)
            args.epochs_s2 = min(args.epochs_s2 + 3, 20)
            # Increase unfreeze breadth
            args.unfreeze_n = min(args.unfreeze_n + 10, 80)
            print(f"\n  ↑ Escalating: epochs_s1={args.epochs_s1} epochs_s2={args.epochs_s2} unfreeze={args.unfreeze_n}")

    # ── Final test evaluation ────────────────────────────────
    print("\n" + "="*60)
    print("  FINAL TEST SET EVALUATION")
    print("="*60)

    if best_model is not None:
        test_results = evaluate_model(best_model, test_s, class_to_id, centers, args.batch_size)
        final_acc = test_results.get("accuracy", 0.0)
        print(f"\n  ✅ Test accuracy: {final_acc:.4f} ({final_acc:.1%})")
        print(f"  Best round: {best_round_name}")
    else:
        print("  [!] No model trained — check data paths.")
        final_acc = 0.0

    # ── Save report ──────────────────────────────────────────
    report = {
        "target_accuracy": args.target_acc,
        "final_test_accuracy": round(final_acc, 4),
        "best_round": best_round_name,
        "best_val_accuracy": round(best_acc_global, 4),
        "target_reached": best_acc_global >= args.target_acc,
        "rounds": history_log,
    }
    report_path = REPORTS_DIR / "finetune_97_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  Report saved → {report_path}")

    # ── Convert best model to TFLite (Float16 quantized) ──────
    if best_model is not None:
        try:
            print("\n  ── Optimising and Quantising Model to TFLite ──")
            tflite_path = MODELS_DIR / "ecotexture_best_overall_float16.tflite"
            converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            tflite_model = converter.convert()
            tflite_path.write_bytes(tflite_model)
            print(f"  ✅ Quantised TFLite model saved → {tflite_path} ({len(tflite_model)/(1024*1024):.2f} MB)")
        except Exception as e:
            print(f"  [!] TFLite export/quantisation failed: {e}")

    if best_acc_global < args.target_acc:
        print(f"\n  ⚠️  Target NOT reached after {len(CHECKPOINT_ROUNDS)} rounds.")
        print(f"     Best achieved: {best_acc_global:.4f}")
        print(f"     Consider: adding more training data, increasing epochs, or running a full unfreeze pass.")
    else:
        print(f"\n  🏆 Mission complete! EcoTexture AI reached {best_acc_global:.1%} accuracy.")

    return 0 if best_acc_global >= args.target_acc else 1


if __name__ == "__main__":
    sys.exit(main())
