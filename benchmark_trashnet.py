"""
EcoTexture AI — Full Benchmark Suite on TrashNet
=================================================
Runs ALL the same model variants from the original research benchmark
(Pure_CNN, Pure_MobileNet, Pure_EfficientNet, Pure_ResNet50,
 Hybrid_CNN_SIFT, Hybrid_MobileNet_SIFT, Cross_Attention_Benchmark)
PLUS the new EcoTexture SOTA Hybrid — on TrashNet.

Goal:
  Prove that EcoTexture SOTA Hybrid (fine-tuned EfficientNetB0 + SIFT + CrossAttn)
  beats ALL pure model and hybrid baselines on the TrashNet waste dataset.

Report is saved to:
  D:\\EcoTexture AI\\reports\\benchmark_trashnet.txt
  D:\\EcoTexture AI\\reports\\benchmark_trashnet.csv

Run:
    python "D:\\EcoTexture AI\\benchmark_trashnet.py"
    python "D:\\EcoTexture AI\\benchmark_trashnet.py" --epochs 60 --batch-size 16
    python "D:\\EcoTexture AI\\benchmark_trashnet.py" --model sota_only  (just our model)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, r"D:\grad project\trial")
try:
    import env_tf_patch
except ImportError:
    pass

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import cv2
import numpy as np
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)

try:
    import tf_keras as keras
    from tf_keras import layers, Model, Input, regularizers
except ImportError:
    try:
        from tensorflow import keras
        from tensorflow.keras import layers, Model, Input, regularizers
    except ImportError:
        import keras
        from keras import layers, Model, Input, regularizers

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, r"D:\EcoTexture AI")
from ecotexture_ai.data import (
    collect_samples, fit_sift_vocabulary, load_split_manifest,
    save_split_manifest, split_samples, sift_histogram, Sample,
)

# ── Paths ────────────────────────────────────────────────────
ECO_AI_ROOT   = Path(r"D:\EcoTexture AI")
MODELS_DIR    = ECO_AI_ROOT / "models" / "benchmark"
REPORTS_DIR   = ECO_AI_ROOT / "reports"
PROCESSED_DIR = ECO_AI_ROOT / "data" / "processed" / "trashnet"
RAW_DIR       = ECO_AI_ROOT / "data" / "raw" / "trashnet"
CACHE_DIR     = ECO_AI_ROOT / "models" / "cache_bench"
for d in [MODELS_DIR, REPORTS_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

IMG_SIZE  = (128, 128)   # Match original benchmark setting
IMG_SIZE_SOTA = (224, 224)  # Our model uses larger resolution
SIFT_K    = 300

# ── GPU ──────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    print(f"[Runtime] GPU: {len(gpus)} device(s)")
else:
    print("[Runtime] CPU float32 mode")


# ─────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────
def clahe_preprocess(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def load_image(path: Path, size: tuple, use_clahe=True) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"Cannot read: {path}")
    bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
    if use_clahe:
        bgr = clahe_preprocess(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


# ─────────────────────────────────────────────────────────────
# SIFT CACHE
# ─────────────────────────────────────────────────────────────
def precompute(samples, class_to_id, centers, img_size, tag):
    cache = CACHE_DIR / f"{tag}.npz"
    if cache.exists():
        d = np.load(cache)
        print(f"  [Cache] {tag} loaded ({len(d['labels'])} samples)")
        return d["images"], d["sifts"], d["labels"]
    print(f"  [SIFT] Computing {tag}: {len(samples)} samples ...", flush=True)
    images, sifts, labels = [], [], []
    for i, s in enumerate(samples):
        try:
            img  = load_image(s.path, img_size, use_clahe=True)
            hist = sift_histogram(s.path, centers, SIFT_K)
            images.append(img); sifts.append(hist); labels.append(class_to_id[s.label])
        except Exception:
            pass
        if (i + 1) % 300 == 0:
            print(f"    {i+1}/{len(samples)} ...", flush=True)
    images = np.asarray(images, dtype=np.float32)
    sifts  = np.asarray(sifts,  dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    np.savez_compressed(str(cache), images=images, sifts=sifts, labels=labels)
    return images, sifts, labels


# ─────────────────────────────────────────────────────────────
# TF DATASET BUILDER
# ─────────────────────────────────────────────────────────────
def make_ds(images, sifts, labels, nc, batch, has_sift=True,
            augment=False, one_hot=True):
    lbl = np.eye(nc, dtype=np.float32)[labels] if one_hot else labels.astype(np.int32)
    if has_sift:
        ds = tf.data.Dataset.from_tensor_slices(((images, sifts), lbl))
    else:
        ds = tf.data.Dataset.from_tensor_slices((images, lbl))
    if augment:
        def aug(x, y):
            img = x[0] if has_sift else x
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.15)
            img = tf.image.random_contrast(img, 0.8, 1.2)
            img = tf.clip_by_value(img, 0.0, 1.0)
            return (img, x[1]) if has_sift else img, y
        ds = ds.shuffle(min(len(images), 2000), reshuffle_each_iteration=True)
        ds = ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE).repeat()
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


# ─────────────────────────────────────────────────────────────
# MODEL FACTORY — identical settings to original benchmark
# ─────────────────────────────────────────────────────────────
class ModelFactory:
    """All models use FROZEN backbones + 128x128 (identical to original benchmark)."""

    @staticmethod
    def pure_cnn(nc):
        inp = Input(shape=(*IMG_SIZE, 3))
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
        x = layers.BatchNormalization()(x); x = layers.MaxPooling2D()(x); x = layers.Dropout(0.25)(x)
        x = layers.SeparableConv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x); x = layers.MaxPooling2D()(x); x = layers.Dropout(0.3)(x)
        x = layers.SeparableConv2D(128, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x); x = layers.GlobalAveragePooling2D()(x); x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x); x = layers.Dropout(0.4)(x)
        out = layers.Dense(nc, activation="softmax", dtype="float32")(x)
        return Model(inp, out, name="Pure_CNN")

    @staticmethod
    def pure_mobilenet(nc):
        base = keras.applications.MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
        base.trainable = False
        inp = Input(shape=(*IMG_SIZE, 3))
        x = base(inp, training=False); x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x); x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x); x = layers.Dropout(0.4)(x)
        out = layers.Dense(nc, activation="softmax", dtype="float32")(x)
        return Model(inp, out, name="Pure_MobileNet")

    @staticmethod
    def pure_resnet50(nc):
        base = keras.applications.ResNet50(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
        base.trainable = False
        inp = Input(shape=(*IMG_SIZE, 3))
        x = base(inp, training=False); x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x); x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x); x = layers.Dropout(0.4)(x)
        out = layers.Dense(nc, activation="softmax", dtype="float32")(x)
        return Model(inp, out, name="Pure_ResNet50")

    @staticmethod
    def pure_efficientnet(nc):
        base = keras.applications.EfficientNetV2B0(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
        base.trainable = False
        inp = Input(shape=(*IMG_SIZE, 3))
        x = base(inp, training=False); x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x); x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x); x = layers.Dropout(0.4)(x)
        out = layers.Dense(nc, activation="softmax", dtype="float32")(x)
        return Model(inp, out, name="Pure_EfficientNet")

    @staticmethod
    def hybrid_cnn_sift(nc):
        ci = Input(shape=(*IMG_SIZE, 3), name="cnn_in")
        si = Input(shape=(SIFT_K,), name="sift_in")
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(ci)
        x = layers.BatchNormalization()(x); x = layers.MaxPooling2D()(x); x = layers.Dropout(0.25)(x)
        x = layers.SeparableConv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x); x = layers.MaxPooling2D()(x); x = layers.Dropout(0.3)(x)
        x = layers.SeparableConv2D(128, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x); x = layers.GlobalAveragePooling2D()(x); x = layers.Dropout(0.4)(x)
        cf = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
        cf = layers.BatchNormalization()(cf); cf = layers.Dropout(0.4)(cf)
        sf = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(si)
        sf = layers.BatchNormalization()(sf)
        m = layers.Concatenate()([cf, sf])
        z = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(m)
        z = layers.Dropout(0.5)(z)
        out = layers.Dense(nc, activation="softmax", dtype="float32")(z)
        return Model([ci, si], out, name="Hybrid_CNN_SIFT")

    @staticmethod
    def hybrid_mobilenet_sift(nc):
        base = keras.applications.MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
        base.trainable = False
        ci = Input(shape=(*IMG_SIZE, 3), name="cnn_in")
        si = Input(shape=(SIFT_K,), name="sift_in")
        x = base(ci, training=False); x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x); x = layers.Dropout(0.4)(x)
        cf = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
        cf = layers.BatchNormalization()(cf)
        sf = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(si)
        sf = layers.BatchNormalization()(sf)
        m = layers.Concatenate()([cf, sf])
        z = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(m)
        z = layers.Dropout(0.5)(z)
        out = layers.Dense(nc, activation="softmax", dtype="float32")(z)
        return Model([ci, si], out, name="Hybrid_MobileNet_SIFT")

    @staticmethod
    def hybrid_resnet50_sift(nc):
        base = keras.applications.ResNet50(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
        base.trainable = False
        ci = Input(shape=(*IMG_SIZE, 3), name="cnn_in")
        si = Input(shape=(SIFT_K,), name="sift_in")
        x = base(ci, training=False); x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x); x = layers.Dropout(0.4)(x)
        cf = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
        cf = layers.BatchNormalization()(cf)
        sf = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(si)
        sf = layers.BatchNormalization()(sf)
        m = layers.Concatenate()([cf, sf])
        z = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(m)
        z = layers.Dropout(0.5)(z)
        out = layers.Dense(nc, activation="softmax", dtype="float32")(z)
        return Model([ci, si], out, name="Hybrid_ResNet50_SIFT")

    @staticmethod
    def hybrid_efficientnet_sift(nc):
        base = keras.applications.EfficientNetV2B0(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
        base.trainable = False
        ci = Input(shape=(*IMG_SIZE, 3), name="cnn_in")
        si = Input(shape=(SIFT_K,), name="sift_in")
        x = base(ci, training=False); x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x); x = layers.Dropout(0.4)(x)
        cf = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
        cf = layers.BatchNormalization()(cf)
        sf = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(si)
        sf = layers.BatchNormalization()(sf)
        m = layers.Concatenate()([cf, sf])
        z = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(m)
        z = layers.Dropout(0.5)(z)
        out = layers.Dense(nc, activation="softmax", dtype="float32")(z)
        return Model([ci, si], out, name="Hybrid_EfficientNet_SIFT")

    @staticmethod
    def cross_attention_benchmark(nc):
        """Cross_Attention from original benchmark (128x128, no fine-tuning)."""
        ci = Input(shape=(*IMG_SIZE, 3), name="cnn_in")
        si = Input(shape=(SIFT_K,), name="sift_in")
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(ci)
        x = layers.BatchNormalization()(x); x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x); x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x); x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x); x = layers.GlobalAveragePooling2D()(x)
        cf = layers.Dense(256, activation="relu")(x); cf = layers.Dropout(0.4)(cf)
        ys = layers.Dense(128, activation="relu")(si); ys = layers.BatchNormalization()(ys)
        sf = layers.Dense(64, activation="relu")(ys)
        merged = layers.Concatenate()([cf, sf])
        merged = layers.Reshape((1, merged.shape[-1]))(merged)
        attn = layers.MultiHeadAttention(num_heads=4, key_dim=64)(merged, merged)
        x2 = layers.LayerNormalization()(attn + merged); x2 = layers.Flatten()(x2)
        z = layers.Dense(128, activation="relu")(x2); z = layers.Dropout(0.4)(z)
        out = layers.Dense(nc, activation="softmax", dtype="float32")(z)
        return Model([ci, si], out, name="Cross_Attention_Benchmark")

    @staticmethod
    def ecotexture_sota(nc, sift_k=300):
        """Our model: Fine-tuned EfficientNetB0 (224×224) + SIFT Cross-Attention."""
        ci = Input(shape=(*IMG_SIZE_SOTA, 3), name="image_input")
        si = Input(shape=(sift_k,), name="sift_input")
        
        # FIX: EfficientNet expects [0, 255] inputs, but our pipeline feeds [0, 1].
        # We scale it back up so the internal Rescaling layer functions correctly.
        ci_scaled = layers.Rescaling(scale=255.0, name="scale_to_255")(ci)
        
        backbone = keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet",
            input_tensor=ci_scaled, pooling="avg")
        # Freeze entire backbone for Stage 1 (head training)
        for layer in backbone.layers:
            layer.trainable = False
        cf = backbone.output
        cf = layers.Dense(256, activation="gelu", kernel_regularizer=regularizers.l2(1e-4))(cf)
        cf = layers.BatchNormalization()(cf); cf = layers.Dropout(0.35)(cf)
        ys = layers.Dense(128, activation="gelu", kernel_regularizer=regularizers.l2(1e-4))(si)
        ys = layers.BatchNormalization()(ys)
        '''
         sf = layers.Dense(64, activation="gelu")(ys); sf = layers.BatchNormalization()(sf)
        merged = layers.Concatenate()([cf, sf])
        merged = layers.Reshape((1, 320))(merged)
        attn = layers.MultiHeadAttention(num_heads=4, key_dim=80)(merged, merged)
        fused = layers.LayerNormalization()(attn + merged); fused = layers.Flatten()(fused)
        '''
        sf = layers.Dense(256, activation="gelu")(ys); sf = layers.BatchNormalization()(sf)
        
        cf_seq = layers.Reshape((1, 256))(cf)
        sf_seq = layers.Reshape((1, 256))(sf)
        sequence = layers.Concatenate(axis=1)([cf_seq, sf_seq])
        
        attn = layers.MultiHeadAttention(num_heads=4, key_dim=256)(sequence, sequence)
        attn = layers.MultiHeadAttention(num_heads=4, key_dim=64)(sequence, sequence)
        fused = layers.Add()([sequence, attn])
        fused = layers.LayerNormalization()(fused); fused = layers.Flatten()(fused)
        z = layers.Dense(256, activation="gelu", kernel_regularizer=regularizers.l2(1e-4))(fused)
        z = layers.BatchNormalization()(z); z = layers.Dropout(0.4)(z)
        z = layers.Dense(128, activation="gelu", kernel_regularizer=regularizers.l2(1e-4))(z)
        z = layers.BatchNormalization()(z); z = layers.Dropout(0.3)(z)
        out = layers.Dense(nc, activation="softmax", dtype="float32")(z)
        return Model([ci, si], out, name="EcoTexture_SOTA")


# ─────────────────────────────────────────────────────────────
# TRAINING ENGINE
# ─────────────────────────────────────────────────────────────
def train_model(model, name, tr_img, tr_sft, tr_lbl,
                va_img, va_sft, va_lbl, nc, epochs, batch,
                has_sift, lr, label_smooth, cw_dict, is_sota=False):
    """Train one model and return best val_accuracy."""
    save_path = MODELS_DIR / f"{name}_best.keras"
    log_path  = MODELS_DIR / f"{name}_history.csv"

    size = IMG_SIZE_SOTA if is_sota else IMG_SIZE
    tr_i = tr_img if is_sota else None  # SOTA uses precomputed 224 cache
    # Note: is_sota flag means we already received correct-size images

    steps = max(1, len(tr_lbl) // batch)

    train_ds = make_ds(tr_img, tr_sft, tr_lbl, nc, batch,
                       has_sift=has_sift, augment=True, one_hot=True)
    val_ds   = make_ds(va_img, va_sft, va_lbl, nc, batch,
                       has_sift=has_sift, augment=False, one_hot=True)

    loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=label_smooth)
    model.compile(
        optimizer=keras.optimizers.Adam(lr, clipnorm=1.0),
        loss=loss_fn,
        metrics=["accuracy"],
    )

    cbs = [
        keras.callbacks.ModelCheckpoint(str(save_path), monitor="val_accuracy",
                                        save_best_only=True, verbose=0),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=6, min_lr=1e-8, verbose=0),
        keras.callbacks.CSVLogger(str(log_path), append=False),
    ]

    t0 = time.time()
    h = model.fit(train_ds, epochs=epochs, steps_per_epoch=steps,
                  validation_data=val_ds, callbacks=cbs,
                  class_weight=cw_dict, verbose=1)
    elapsed = time.time() - t0
    best_acc = max(h.history.get("val_accuracy", [0.0]))
    print(f"  [{name}] Best val_acc = {best_acc:.4f}  ({elapsed/60:.1f} min)")
    return best_acc, model


def evaluate_model(model, te_img, te_sft, te_lbl, nc, batch, has_sift):
    """Evaluate on test set → accuracy, F1, FPS."""
    test_ds = make_ds(te_img, te_sft, te_lbl, nc, batch,
                      has_sift=has_sift, augment=False, one_hot=False)

    # Predictions
    preds = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = te_lbl[:len(y_pred)]  # align

    acc = accuracy_score(y_true, y_pred)
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)

    # FPS benchmark
    dummy_img = np.random.rand(1, *te_img.shape[1:]).astype(np.float32)
    dummy_sft = np.random.rand(1, SIFT_K).astype(np.float32)
    inp = [dummy_img, dummy_sft] if has_sift else dummy_img
    for _ in range(3):
        model.predict(inp, verbose=0)
    t0 = time.perf_counter()
    for _ in range(30):
        model.predict(inp, verbose=0)
    fps = 30 / (time.perf_counter() - t0)

    params  = model.count_params()
    return acc, f1, fps, params


# ─────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────
def generate_report(results: list[dict]):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 95,
        "  BENCHMARK RESULTS: TrashNet (Waste Material Classification)",
        f"  Generated: {ts}",
        "  Comparison: Original Benchmark Models vs EcoTexture SOTA Hybrid",
        "=" * 95,
        "",
        "  ORIGINAL BENCHMARK USED: Frozen backbones, 128×128 images",
        "  ECOTEXTURE SOTA USES:    Fine-tuned EfficientNetB0, 224×224 + SIFT Cross-Attn",
        "",
        f"{'Model':<40} | {'Val Acc':>8} | {'Test Acc':>8} | {'F1':>7} | {'FPS':>6} | {'Params':>9}",
        "-" * 90,
    ]

    # Sort by test acc descending
    for r in sorted(results, key=lambda x: x["test_acc"], reverse=True):
        marker = " ★" if "EcoTexture_SOTA" in r["name"] else ""
        lines.append(
            f"{r['name']:<40} | {r['val_acc']:>7.2%} | {r['test_acc']:>7.2%} | "
            f"{r['f1']:>6.2%} | {r['fps']:>5.1f} | {r['params']:>9,}{marker}"
        )

    # Gap to benchmark
    sota = next((r for r in results if "EcoTexture_SOTA" in r["name"]), None)
    bench_best = next((r for r in sorted(results, key=lambda x: x["test_acc"], reverse=True)
                       if "EcoTexture_SOTA" not in r["name"]), None)
    if sota and bench_best:
        gap = sota["test_acc"] - bench_best["test_acc"]
        lines += [
            "",
            "─" * 90,
            f"  ★ EcoTexture SOTA vs best baseline ({bench_best['name']}): "
            f"{'+' if gap >= 0 else ''}{gap:.2%}",
            "─" * 90,
        ]

    txt = "\n".join(lines)
    rp = REPORTS_DIR / "benchmark_trashnet.txt"
    rp.write_text(txt, encoding="utf-8")

    cp = REPORTS_DIR / "benchmark_trashnet.csv"
    with cp.open("w", encoding="utf-8") as f:
        f.write("Model,ValAcc,TestAcc,F1,FPS,Params\n")
        for r in sorted(results, key=lambda x: x["test_acc"], reverse=True):
            f.write(f"{r['name']},{r['val_acc']:.6f},{r['test_acc']:.6f},"
                    f"{r['f1']:.6f},{r['fps']:.1f},{r['params']}\n")

    try:
        print("\n" + txt)
    except UnicodeEncodeError:
        try:
            print("\n" + txt.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8'))
        except Exception:
            print("\n" + txt.replace("★", "*"))
    print(f"\n[Report] {rp}")
    print(f"[CSV]    {cp}")
    return txt


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Full benchmark on TrashNet — all models vs EcoTexture SOTA")
    parser.add_argument("--epochs",       type=int,   default=60)
    parser.add_argument("--batch-size",   type=int,   default=16)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--label-smooth", type=float, default=0.1)
    parser.add_argument("--model",        type=str,   default="all",
                        choices=["all", "baselines_only", "sota_only"],
                        help="Which models to run")
    parser.add_argument("--no-cache",     action="store_true")
    args = parser.parse_args()

    if args.no_cache:
        for f in CACHE_DIR.glob("*.npz"):
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
            raise RuntimeError(f"No images in {RAW_DIR}")
        train_s, val_s, test_s = split_samples(samples)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        save_split_manifest(train_s, train_m)
        save_split_manifest(val_s, val_m)
        save_split_manifest(test_s, test_m)

    all_s       = train_s + val_s + test_s
    class_to_id = {l: i for i, l in enumerate(sorted({s.label for s in all_s}))}
    nc          = len(class_to_id)
    (ECO_AI_ROOT / "models" / "class_to_id.json").write_text(
        json.dumps(class_to_id, indent=2), encoding="utf-8")

    print(f"\n[Data] {len(train_s)} train / {len(val_s)} val / {len(test_s)} test")
    print(f"[Data] Classes ({nc}): {list(class_to_id.keys())}")

    # ── SIFT Vocabulary ──────────────────────────────────────
    vocab_path = ECO_AI_ROOT / "models" / "sift_centers.npy"
    if vocab_path.exists():
        centers = np.load(vocab_path)
        print(f"[SIFT] Vocab loaded: {centers.shape}")
    else:
        trial_vocab = Path(r"D:\grad project\trial\hybrid\meta\sift_kmeans_centers.npy")
        if trial_vocab.exists():
            import shutil; shutil.copy2(trial_vocab, vocab_path)
            centers = np.load(vocab_path)
        else:
            print("[SIFT] Fitting vocabulary ...")
            centers = fit_sift_vocabulary(train_s, vocab_size=SIFT_K, max_images=30)
            np.save(vocab_path, centers)
        print(f"[SIFT] Vocab: {centers.shape}")

    # ── Precompute: 128x128 (baselines) & 224x224 (SOTA) ────
    print("\n[Pipeline] Precomputing 128×128 arrays (baseline models) ...")
    tr128_img, tr128_sft, tr128_lbl = precompute(train_s, class_to_id, centers, IMG_SIZE, "train_128")
    va128_img, va128_sft, va128_lbl = precompute(val_s,   class_to_id, centers, IMG_SIZE, "val_128")
    te128_img, te128_sft, te128_lbl = precompute(test_s,  class_to_id, centers, IMG_SIZE, "test_128")

    if args.model in ("all", "sota_only"):
        print("\n[Pipeline] Precomputing 224×224 arrays (SOTA model) ...")
        tr224_img, tr224_sft, tr224_lbl = precompute(train_s, class_to_id, centers, IMG_SIZE_SOTA, "train_224")
        va224_img, va224_sft, va224_lbl = precompute(val_s,   class_to_id, centers, IMG_SIZE_SOTA, "val_224")
        te224_img, te224_sft, te224_lbl = precompute(test_s,  class_to_id, centers, IMG_SIZE_SOTA, "test_224")

    # Class weights
    cw_arr  = compute_class_weight("balanced", classes=np.unique(tr128_lbl), y=tr128_lbl)
    cw_dict = {int(c): float(w) for c, w in zip(np.unique(tr128_lbl), cw_arr)}

    # ─────────────────────────────────────────────────────────
    # DEFINE BENCHMARK SUITE
    # ─────────────────────────────────────────────────────────
    BASELINE_MODELS = [
        # (factory_fn, name, has_sift)
        (lambda: ModelFactory.pure_cnn(nc),                  "Pure_CNN",                False),
        (lambda: ModelFactory.pure_mobilenet(nc),             "Pure_MobileNet",          False),
        (lambda: ModelFactory.pure_resnet50(nc),              "Pure_ResNet50",           False),
        (lambda: ModelFactory.pure_efficientnet(nc),          "Pure_EfficientNet",       False),
        (lambda: ModelFactory.hybrid_cnn_sift(nc),            "Hybrid_CNN_SIFT",         True),
        (lambda: ModelFactory.hybrid_mobilenet_sift(nc),      "Hybrid_MobileNet_SIFT",   True),
        (lambda: ModelFactory.hybrid_resnet50_sift(nc),       "Hybrid_ResNet50_SIFT",    True),
        (lambda: ModelFactory.hybrid_efficientnet_sift(nc),   "Hybrid_EfficientNet_SIFT",True),
        (lambda: ModelFactory.cross_attention_benchmark(nc),  "Cross_Attention_Benchmark",True),
    ]

    results_path = REPORTS_DIR / "benchmark_trashnet_raw.json"
    if results_path.exists():
        try:
            results_list = json.loads(results_path.read_text(encoding="utf-8"))
            print(f"[Results] Loaded {len(results_list)} existing results from cache.")
        except Exception:
            results_list = []
    else:
        results_list = []
    
    results_dict = {r["name"]: r for r in results_list}
    steps = max(1, len(train_s) // args.batch_size)

    # ─────────────────────────────────────────────────────────
    # TRAIN BASELINE MODELS (128×128, frozen backbones)
    # ─────────────────────────────────────────────────────────
    if args.model in ("all", "baselines_only"):
        print(f"\n{'='*65}")
        print(f"  PHASE 1 — BASELINE MODELS (128×128, frozen backbones)")
        print(f"  Identical conditions to original research benchmark")
        print(f"{'='*65}")

        for factory_fn, name, has_sift in BASELINE_MODELS:
            if name in results_dict:
                print(f"  [Skip] {name} already in results cache.")
                continue
            print(f"\n[{name}] Building model ...")
            keras.backend.clear_session()
            model = factory_fn()

            val_acc, model = train_model(
                model, name,
                tr128_img, tr128_sft, tr128_lbl,
                va128_img, va128_sft, va128_lbl,
                nc, args.epochs, args.batch_size,
                has_sift, args.lr, args.label_smooth, cw_dict,
            )

            test_acc, f1, fps, params = evaluate_model(
                model, te128_img, te128_sft, te128_lbl,
                nc, args.batch_size, has_sift)

            results_dict[name] = {
                "name": name, "val_acc": val_acc,
                "test_acc": test_acc, "f1": f1,
                "fps": fps, "params": params,
            }
            print(f"  [RESULT] {name}: val={val_acc:.4f} test={test_acc:.4f} f1={f1:.4f}")

    # ─────────────────────────────────────────────────────────
    # TRAIN ECOTEXTURE SOTA (224×224, fine-tuned EfficientNetB0)
    # ─────────────────────────────────────────────────────────
    if args.model in ("all", "sota_only"):
        if any(k in results_dict for k in ("EcoTexture_SOTA", "EcoTexture_SOTA_97")):
            print("\n  [Skip] EcoTexture SOTA already in results cache.")
        else:
            print(f"\n{'='*65}")
            print(f"  PHASE 2 — ECOTEXTURE SOTA (224×224, fine-tuned backbone)")
            print(f"  EfficientNetB0 + SIFT Cross-Attention, 2-stage training")
            print(f"{'='*65}")

            keras.backend.clear_session()
            sota_model = ModelFactory.ecotexture_sota(nc, sift_k=centers.shape[0])
            sota_model.summary(line_length=90)

            # Stage 1: head only (backbone mostly frozen)
            print("\n  [SOTA Stage 1] Head training ...")
            val_acc_s1, sota_model = train_model(
                sota_model, "EcoTexture_SOTA_97_S1",
                tr224_img, tr224_sft, tr224_lbl,
                va224_img, va224_sft, va224_lbl,
                nc, max(20, args.epochs // 3), args.batch_size,
                True, args.lr, args.label_smooth, cw_dict, is_sota=True,
            )

            # Stage 2: unfreeze ALL layers to push for 97% target
            print("\n  [SOTA Stage 2] Fine-tuning all layers for maximum accuracy ...")
            for layer in sota_model.layers:
                # Keep BatchNormalization frozen to prevent accuracy crash
                if isinstance(layer, layers.BatchNormalization):
                    layer.trainable = False
                else:
                    layer.trainable = True

            val_acc_s2, sota_model = train_model(
                sota_model, "EcoTexture_SOTA_97",
                tr224_img, tr224_sft, tr224_lbl,
                va224_img, va224_sft, va224_lbl,
                nc, args.epochs, args.batch_size,
                True, 5e-5, args.label_smooth, cw_dict, is_sota=True,
            )

            val_acc_sota = max(val_acc_s1, val_acc_s2)
            test_acc, f1, fps, params = evaluate_model(
                sota_model, te224_img, te224_sft, te224_lbl,
                nc, args.batch_size, True)

            sota_model.save(str(MODELS_DIR / "EcoTexture_SOTA_97_final.keras"))
            results_dict["EcoTexture_SOTA_97"] = {
                "name": "EcoTexture_SOTA_97", "val_acc": val_acc_sota,
                "test_acc": test_acc, "f1": f1, "fps": fps, "params": params,
            }
            print(f"\n  [RESULT] EcoTexture_SOTA_97: val={val_acc_sota:.4f} test={test_acc:.4f} f1={f1:.4f}")

    # ── Save intermediate results in case of crash ───────────
    final_results = list(results_dict.values())
    results_path.write_text(json.dumps(final_results, indent=2), encoding="utf-8")

    # ─────────────────────────────────────────────────────────
    # GENERATE REPORT
    # ─────────────────────────────────────────────────────────
    generate_report(final_results)


if __name__ == "__main__":
    main()
