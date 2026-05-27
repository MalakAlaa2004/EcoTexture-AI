"""
EcoTexture AI — Production Inference Pipeline
==============================================
Loads assets once, runs hybrid CNN + SIFT prediction with GradCAM.
"""
from __future__ import annotations

import json
from pathlib import Path

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

from ecotexture_ai.config import EXISTING_SIFT_CENTERS, IMG_SIZE, MODELS_DIR, SIFT_K, WASTE_CLASSES
from ecotexture_ai.explain import compute_sift_histogram, draw_sift_keypoints, get_gradcam_heatmap
from ecotexture_ai.recommendations import (
    get_arabic_label,
    get_bin_colour,
    get_co2_savings,
    get_educational_insight,
    get_recommendation,
    get_sdg_tags,
)


ModelBundle = tuple[keras.Model, np.ndarray, dict[int, str]]

_CACHED: ModelBundle | None = None
LEGACY_MODELS_DIR = Path(r"D:\EcoTexture A\models")


def _candidate_model_paths(models_dir: Path) -> list[Path]:
    return [
        models_dir / "benchmark" / "EcoTexture_SOTA_97_best.keras",
        models_dir / "benchmark" / "EcoTexture_SOTA_best.keras",
        models_dir / "benchmark" / "EcoTexture_SOTA_final.keras",
        models_dir / "ecotexture_coco_met_final.h5",
        models_dir / "ecotexture_hybrid_final.h5",
        models_dir / "ecotexture_final.h5",
        models_dir / "ecotexture_best.h5",
        models_dir / "contrastive_transfer_best.h5",
        models_dir / "ecotexture_coco_met_best.h5",
        models_dir / "ecotexture_hybrid_best.h5",
        models_dir / "ecotexture_coco_met_final.keras",
        models_dir / "ecotexture_hybrid_final.keras",
        models_dir / "ecotexture_final.keras",
        models_dir / "ecotexture_best.keras",
        models_dir / "contrastive_transfer_best.keras",
        models_dir / "ecotexture_coco_met_best.keras",
        models_dir / "ecotexture_hybrid_best.keras",
        LEGACY_MODELS_DIR / "contrastive_transfer_final.h5",
        LEGACY_MODELS_DIR / "contrastive_transfer_best.h5",
        LEGACY_MODELS_DIR / "hybrid_sift_final.h5",
        LEGACY_MODELS_DIR / "hybrid_sift_best.h5",
        LEGACY_MODELS_DIR / "contrastive_transfer_final.keras",
        LEGACY_MODELS_DIR / "contrastive_transfer_best.keras",
        LEGACY_MODELS_DIR / "ecotexture_hybrid_final.keras",
        LEGACY_MODELS_DIR / "ecotexture_hybrid_best.keras",
    ]


def load_assets(force_reload: bool = False) -> ModelBundle:
    global _CACHED
    if _CACHED is not None and not force_reload:
        return _CACHED

    candidates = _candidate_model_paths(MODELS_DIR)
    model_path = next((path for path in candidates if path.exists()), None)
    if model_path is None:
        raise FileNotFoundError(
            f"No trained model checkpoint found in {MODELS_DIR}. Checked: {[path.name for path in candidates]}"
        )

    old_policy = keras.mixed_precision.global_policy().name
    try:
        keras.mixed_precision.set_global_policy("mixed_float16")
        model = keras.models.load_model(str(model_path), compile=False)
    except Exception as e:
        print(f"[predict.py] Load failed under mixed_float16: {e}. Trying standard load...")
        keras.mixed_precision.set_global_policy(old_policy)
        try:
            model = keras.models.load_model(str(model_path), compile=False)
        except Exception as e2:
            from ecotexture_ai.model import CrossAttentionFusion
            model = keras.models.load_model(
                str(model_path),
                custom_objects={"CrossAttentionFusion": CrossAttentionFusion},
                compile=False
            )
    
    # ── Class map ─────────────────────────────────────────────
    class_map_candidates = [MODELS_DIR / "class_to_id.json", LEGACY_MODELS_DIR / "class_to_id.json"]
    class_map_path = next((path for path in class_map_candidates if path.exists()), None)
    if class_map_path is not None:
        class_to_id: dict[str, int] = json.loads(class_map_path.read_text(encoding="utf-8"))
    else:
        class_to_id = {name: idx for idx, name in enumerate(WASTE_CLASSES)}
    id_to_class: dict[int, str] = {v: k for k, v in class_to_id.items()}

    # ── SIFT centers ──────────────────────────────────────────
    vocab_candidates = [MODELS_DIR / "sift_centers.npy", LEGACY_MODELS_DIR / "sift_centers.npy", *EXISTING_SIFT_CENTERS]
    centers = None
    for c in vocab_candidates:
        if c.exists():
            centers = np.load(c)
            break
    if centers is None:
        raise FileNotFoundError("sift_centers.npy not found. Run training first.")

    _CACHED = (model, centers, id_to_class)
    return _CACHED


def predict_image(
    img_bgr: np.ndarray,
    model: keras.Model,
    centers: np.ndarray,
    id_to_class: dict[int, str],
    lang: str = "en",
    top_k: int = 3,
) -> dict:
    """
    Full hybrid inference: CNN + SIFT → cross-attention → classification.
    Returns a rich result dict with heatmaps, recommendations, and SDG tags.
    """
    try:
        # Dynamically determine the input shape required by the loaded model
        cnn_shape = model.input_shape[0]
        h, w = cnn_shape[1], cnn_shape[2]
    except Exception:
        h, w = IMG_SIZE
        
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # CNN input
    cnn_in = cv2.resize(img_rgb, (w, h)).astype(np.float32) / 255.0
    cnn_in = np.expand_dims(cnn_in, 0)

    # SIFT input
    sift_hist = compute_sift_histogram(img_bgr, centers)
    sift_in   = np.expand_dims(sift_hist, 0)

    # Inference
    probs = model.predict([cnn_in, sift_in], verbose=0)[0]
    pred_idx   = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    label      = id_to_class.get(pred_idx, "Unknown")

    # Top-K
    top_k_idx = np.argsort(probs)[::-1][:top_k]
    top_k_out = [
        {"label": id_to_class.get(int(i), "?"), "confidence": float(probs[i])}
        for i in top_k_idx
    ]

    # Explainability
    heatmap = get_gradcam_heatmap(model, cnn_in, sift_in, pred_index=pred_idx)
    sift_vis = draw_sift_keypoints(img_rgb)
    label_ar = get_arabic_label(label)

    return {
        "label":          label,
        "label_ar":       label_ar,
        "confidence":     confidence,
        "top_k":          top_k_out,
        "recommendation": get_recommendation(label, lang),
        "insight":        get_educational_insight(label, lang),
        "impact_note":    f"Recycling this saves ~{get_co2_savings(label):.1f} kg CO₂",
        "bin_colour":     get_bin_colour(label),
        "sdg_tags":       get_sdg_tags(label),
        "heatmap":        heatmap,        # (H, W, 3) uint8 RGB
        "sift_overlay":   sift_vis,       # (H, W, 3) uint8 RGB
        "sift_active":    bool(sift_hist.sum() > 0),
    }
