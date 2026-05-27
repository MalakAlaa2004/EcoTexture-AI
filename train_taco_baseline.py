"""
TACO Dataset Baseline Training
================================
Trains the EcoTexture AI hybrid model on the downloaded TACO subset.
~2 hour budget: 15 head-tuning epochs + 25 fine-tuning epochs.
"""
from __future__ import annotations
import os, json, pathlib, time
import numpy as np
import cv2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import tf_keras as keras
from tf_keras import layers

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
DATA_DIR  = pathlib.Path(r"D:\EcoTexture AI\data\raw\taco\images")
MODEL_DIR = pathlib.Path(r"D:\EcoTexture AI\models\taco_baseline")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE  = (224, 224)
BATCH     = 16
K_VOCAB   = 150        # smaller vocab for TACO subset
HEAD_LR   = 1e-3
FINETUNE_LR = 2.5e-5
HEAD_EPOCHS = 15
FT_EPOCHS   = 25
NUM_CLASSES = 6

CLASSES = sorted(["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"])

# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
def load_data():
    images, labels = [], []
    for cls_idx, cls in enumerate(CLASSES):
        cls_dir = DATA_DIR / cls
        if not cls_dir.exists():
            print(f"  [skip] {cls} — no directory found")
            continue
        files = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(cls_idx)
    return np.array(images, dtype=np.float32) / 255.0, np.array(labels)

print("[TACO] Loading images...")
X, y = load_data()
print(f"[TACO] Loaded {len(X)} images across {len(np.unique(y))} classes")

if len(X) < 20:
    print("[TACO] Not enough images to train. Download more first.")
    exit(1)

# Class distribution
for i, cls in enumerate(CLASSES):
    count = int((y == i).sum())
    print(f"  {cls}: {count}")

# Train/val split
from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"[TACO] Train: {len(X_tr)}, Val: {len(X_val)}")

# ─────────────────────────────────────────────────────────────
# SIFT VOCABULARY (from training set)
# ─────────────────────────────────────────────────────────────
print("[TACO] Extracting SIFT features for vocabulary...")
sift = cv2.SIFT_create(nfeatures=300)
all_descs = []
for img_f32 in X_tr:
    img_u8 = (img_f32 * 255).astype(np.uint8)
    gray   = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    if des is not None:
        all_descs.append(des)

if not all_descs:
    print("[TACO] No SIFT descriptors found — skipping SIFT branch")
    USE_SIFT = False
else:
    all_descs = np.vstack(all_descs).astype(np.float32)
    print(f"[TACO] Clustering {len(all_descs)} descriptors into {K_VOCAB} words...")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, _, centers = cv2.kmeans(all_descs, K_VOCAB, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
    np.save(MODEL_DIR / "taco_vocab_centers.npy", centers)
    USE_SIFT = True
    print("[TACO] Vocabulary ready.")

def extract_bow(img_f32):
    img_u8 = (img_f32 * 255).astype(np.uint8)
    gray   = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    if des is None or len(des) == 0:
        return np.zeros(K_VOCAB, dtype=np.float32)
    # Assign each desc to nearest center
    des = des.astype(np.float32)
    hist = np.zeros(K_VOCAB, dtype=np.float32)
    for d in des:
        dists = np.linalg.norm(centers - d, axis=1)
        hist[np.argmin(dists)] += 1
    norm = hist.sum()
    return (hist / norm) if norm > 0 else hist

if USE_SIFT:
    print("[TACO] Computing BoW histograms for train/val sets...")
    bow_tr  = np.array([extract_bow(x) for x in X_tr])
    bow_val = np.array([extract_bow(x) for x in X_val])
    print("[TACO] BoW done.")

# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────
def build_model(use_sift: bool = True, k: int = K_VOCAB, num_classes: int = NUM_CLASSES):
    cnn_in = keras.Input(shape=(*IMG_SIZE, 3), name="cnn_input")
    backbone = keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet",
        input_shape=(*IMG_SIZE, 3), pooling="avg"
    )
    backbone.trainable = False
    cnn_feat = backbone(cnn_in, training=False)      # (B, 1280)

    if use_sift:
        sift_in = keras.Input(shape=(k,), name="sift_input")
        cf = layers.Dense(256, activation="relu", name="proj_cnn")(cnn_feat)
        sf = layers.Dense(256, activation="relu", name="proj_sift")(sift_in)
        cf_seq = layers.Reshape((1, 256))(cf)
        sf_seq = layers.Reshape((1, 256))(sf)
        sequence = layers.Concatenate(axis=1)([cf_seq, sf_seq])
        attn = layers.MultiHeadAttention(num_heads=4, key_dim=64, name="cross_attn")(sequence, sequence)
        fused = layers.Add()([sequence, attn])
        fused = layers.LayerNormalization()(fused)
        fused = layers.Flatten()(fused)
        fused = layers.Dropout(0.4)(fused)
        out = layers.Dense(num_classes, activation="softmax", name="classifier")(fused)
        model = keras.Model(inputs=[cnn_in, sift_in], outputs=out)
    else:
        cf = layers.Dense(256, activation="relu")(cnn_feat)
        cf = layers.Dropout(0.4)(cf)
        out = layers.Dense(num_classes, activation="softmax", name="classifier")(cf)
        model = keras.Model(inputs=cnn_in, outputs=out)
    return model

model = build_model(use_sift=USE_SIFT)
model.summary()

# ─────────────────────────────────────────────────────────────
# TRAINING HELPERS
# ─────────────────────────────────────────────────────────────
if USE_SIFT:
    train_inputs = [X_tr, bow_tr]
    val_inputs   = [X_val, bow_val]
else:
    train_inputs = X_tr
    val_inputs   = X_val

y_tr_cat  = keras.utils.to_categorical(y_tr,  NUM_CLASSES)
y_val_cat = keras.utils.to_categorical(y_val, NUM_CLASSES)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        str(MODEL_DIR / "taco_best.keras"),
        monitor="val_accuracy", save_best_only=True, verbose=1
    ),
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=7, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
]

# ─────────────────────────────────────────────────────────────
# STAGE 1: HEAD TUNING
# ─────────────────────────────────────────────────────────────
print("\n[TACO] === STAGE 1: Head Tuning ===")
model.compile(
    optimizer=keras.optimizers.Adam(HEAD_LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
t0 = time.time()
h1 = model.fit(
    train_inputs, y_tr_cat,
    validation_data=(val_inputs, y_val_cat),
    epochs=HEAD_EPOCHS, batch_size=BATCH,
    callbacks=callbacks, verbose=1,
)
print(f"[TACO] Stage 1 done in {(time.time()-t0)/60:.1f} min")

# ─────────────────────────────────────────────────────────────
# STAGE 2: FINE-TUNING
# ─────────────────────────────────────────────────────────────
print("\n[TACO] === STAGE 2: Fine-Tuning ===")
# Unfreeze backbone except BN layers
backbone_layer = model.get_layer("efficientnetb0")
backbone_layer.trainable = True
for layer in backbone_layer.layers:
    if isinstance(layer, keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(FINETUNE_LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
t1 = time.time()
h2 = model.fit(
    train_inputs, y_tr_cat,
    validation_data=(val_inputs, y_val_cat),
    epochs=FT_EPOCHS, batch_size=BATCH,
    callbacks=callbacks, verbose=1,
)
print(f"[TACO] Stage 2 done in {(time.time()-t1)/60:.1f} min")
print(f"[TACO] Total training time: {(time.time()-t0)/60:.1f} min")

# ─────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────
best = keras.models.load_model(str(MODEL_DIR / "taco_best.keras"))
loss, acc = best.evaluate(val_inputs, y_val_cat, verbose=0)
print(f"\n[TACO] Best val accuracy: {acc:.4f} ({acc*100:.2f}%)")

# Save results
results = {
    "dataset": "TACO subset",
    "num_classes": NUM_CLASSES,
    "train_samples": len(X_tr),
    "val_samples": len(X_val),
    "best_val_acc": float(acc),
    "use_sift": USE_SIFT,
    "vocab_size": K_VOCAB,
}
with open(MODEL_DIR / "taco_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("[TACO] Results saved.")
print(json.dumps(results, indent=2))
