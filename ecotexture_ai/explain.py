"""
EcoTexture AI — Explainability Module
=======================================
GradCAM attention heatmaps + SIFT keypoint visualisation.
These visuals are the core of the WISE educational narrative:
"show students *why* the AI made that decision."
"""
from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf
try:
    import tf_keras as keras
except ImportError:
    try:
        from tensorflow import keras
    except ImportError:
        import keras


# ─────────────────────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────────────────────
def get_gradcam_heatmap(
    model: keras.Model,
    img_array: np.ndarray,        # (1, H, W, 3) float32 [0,1]
    sift_array: np.ndarray,       # (1, SIFT_K) float32
    last_conv_layer_name: str | None = None,
    pred_index: int | None = None,
) -> np.ndarray:
    """
    Compute a GradCAM heatmap for the CNN branch.
    Returns a (H, W, 3) uint8 heatmap overlay.
    """
    # Find the last conv layer automatically if not specified
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break

    if last_conv_layer_name is None:
        # Fallback — just return a blank heatmap
        h, w = img_array.shape[1:3]
        return np.zeros((h, w, 3), dtype=np.uint8)

    # Build gradient model
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output,
        ],
    )

    with tf.GradientTape() as tape:
        inputs       = [tf.cast(img_array, tf.float32), tf.cast(sift_array, tf.float32)]
        conv_outputs, predictions = grad_model(inputs)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap).numpy()
    heatmap      = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)

    # Resize to original image size
    h, w = img_array.shape[1:3]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )

    # Blend with original image
    orig_uint8 = (img_array[0] * 255).astype(np.uint8)
    orig_bgr   = cv2.cvtColor(orig_uint8, cv2.COLOR_RGB2BGR)
    overlay    = cv2.addWeighted(orig_bgr, 0.55, heatmap_colored, 0.45, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────────────────
# SIFT KEYPOINT VISUALISATION
# ─────────────────────────────────────────────────────────────
def draw_sift_keypoints(
    img_rgb: np.ndarray,      # (H, W, 3) float32 [0,1] or uint8
    n_features: int = 500,
    rich: bool = True,
) -> np.ndarray:
    """
    Draw SIFT keypoints on the image with a premium look.
    Returns (H, W, 3) uint8 RGB.
    """
    if img_rgb.dtype != np.uint8:
        img_uint8 = (np.clip(img_rgb, 0, 1) * 255).astype(np.uint8)
    else:
        img_uint8 = img_rgb.copy()

    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)

    sift = cv2.SIFT_create(nfeatures=n_features)
    kps, _ = sift.detectAndCompute(gray, None)

    if not kps:
        return img_uint8

    if rich:
        # Draw rings + orientation arrows
        overlay = img_uint8.copy()
        for kp in kps:
            cx, cy = int(kp.pt[0]), int(kp.pt[1])
            r      = max(3, int(kp.size / 2))
            # Outer ring (semi-transparent gold)
            cv2.circle(overlay, (cx, cy), r + 2, (255, 200, 40), 1, cv2.LINE_AA)
            # Inner dot
            cv2.circle(overlay, (cx, cy), 2, (40, 240, 160), -1, cv2.LINE_AA)
            # Orientation line
            angle_rad = np.radians(kp.angle)
            ex = int(cx + r * np.cos(angle_rad))
            ey = int(cy + r * np.sin(angle_rad))
            cv2.line(overlay, (cx, cy), (ex, ey), (255, 80, 80), 1, cv2.LINE_AA)
        result = cv2.addWeighted(img_uint8, 0.6, overlay, 0.4, 0)
    else:
        result = cv2.drawKeypoints(
            img_uint8, kps, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )

    # Add keypoint count badge
    cv2.putText(
        result, f"SIFT: {len(kps)} keypoints",
        (10, result.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 100), 2, cv2.LINE_AA,
    )
    return result


# ─────────────────────────────────────────────────────────────
# SIFT BOW HISTOGRAM  (for predict pipeline)
# ─────────────────────────────────────────────────────────────
def compute_sift_histogram(img_bgr: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """CLAHE-enhanced SIFT → BoVW histogram (vocab_size-dim)."""
    from scipy.cluster.vq import vq
    vocab_size = centers.shape[0]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    sift = cv2.SIFT_create(nfeatures=500)
    _, descriptors = sift.detectAndCompute(gray, None)
    hist = np.zeros(vocab_size, dtype=np.float32)
    if descriptors is not None:
        words, _ = vq(descriptors, centers)
        for w in words:
            hist[w] += 1
        total = hist.sum()
        if total > 0:
            hist /= total
    return hist
