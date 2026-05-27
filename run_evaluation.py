from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from ecotexture_ai.config import IMG_SIZE, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, ensure_dirs
from ecotexture_ai.data import image_to_array, load_split_manifest, sift_histogram
from ecotexture_ai.model import CrossAttentionFusion, build_contrastive_transfer_model


def _load_model(model_path: Path, num_classes: int, technique: str):
    custom_objects = {
        "CrossAttentionFusion": CrossAttentionFusion,
        "Custom>CrossAttentionFusion": CrossAttentionFusion,
        "EcoTexture>CrossAttentionFusion": CrossAttentionFusion,
    }
    if model_path.suffix == ".weights.h5" and technique == "contrastive_transfer":
        model = build_contrastive_transfer_model(num_classes=num_classes)
        model.load_weights(model_path)
        return model

    # Try loading with tf_keras first, as it was used to save the benchmark models
    try:
        import tf_keras
        load_fn = tf_keras.models.load_model
    except ImportError:
        load_fn = tf.keras.models.load_model

    try:
        return load_fn(model_path, custom_objects=custom_objects, compile=False)
    except Exception as e:
        try:
            return tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False, safe_mode=False)
        except Exception:
            if technique == "contrastive_transfer":
                model = build_contrastive_transfer_model(num_classes=num_classes)
                model.load_weights(model_path)
                return model
            raise e


def _first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate EcoTexture model")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--class-map-path", type=Path, default=None)
    parser.add_argument("--centers-path", type=Path, default=None)
    parser.add_argument("--test-manifest", type=Path, default=PROCESSED_DIR / "trashnet" / "test.json")
    parser.add_argument("--report-subdir", type=str, default="")
    parser.add_argument("--technique", type=str, default="hybrid_sift", choices=["hybrid_sift", "contrastive_transfer"])
    args = parser.parse_args()

    ensure_dirs()
    reports_dir = REPORTS_DIR / args.report_subdir if args.report_subdir else REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)

    legacy_models_dir = Path(r"D:\EcoTexture A\models")

    model_path = args.model_path or _first_existing([
        MODELS_DIR / "ecotexture_coco_met_final.keras",
        MODELS_DIR / "ecotexture_hybrid_final.keras",
        MODELS_DIR / "contrastive_transfer_final.keras",
        MODELS_DIR / "ecotexture_final.keras",
        legacy_models_dir / "contrastive_transfer_final.keras",
        legacy_models_dir / "contrastive_transfer_best.keras",
        legacy_models_dir / "ecotexture_hybrid_final.keras",
        legacy_models_dir / "ecotexture_hybrid_best.keras",
    ])
    if model_path is None:
        raise FileNotFoundError("No model checkpoint found in either EcoTexture root")

    class_map_path = args.class_map_path or _first_existing([
        MODELS_DIR / "class_to_id.json",
        legacy_models_dir / "class_to_id.json",
    ])
    if class_map_path is None:
        raise FileNotFoundError("No class_to_id.json found in either EcoTexture root")

    centers_path = args.centers_path or _first_existing([
        MODELS_DIR / "sift_centers.npy",
        legacy_models_dir / "sift_centers.npy",
    ])
    if centers_path is None:
        raise FileNotFoundError("No sift_centers.npy found in either EcoTexture root")

    class_to_id = json.loads(class_map_path.read_text(encoding="utf-8"))
    id_to_class = {idx: label for label, idx in class_to_id.items()}
    labels = [id_to_class[i] for i in range(len(id_to_class))]

    centers = np.load(centers_path)
    model = _load_model(model_path, num_classes=len(class_to_id), technique=args.technique)
    test_samples = load_split_manifest(args.test_manifest)

    y_true: list[int] = []
    imgs = []
    sifts = []

    print(f"Loading and pre-processing {len(test_samples)} test samples...")
    for i, sample in enumerate(test_samples):
        x_img = image_to_array(sample.path, IMG_SIZE)
        x_sift = sift_histogram(sample.path, centers, centers.shape[0])
        imgs.append(x_img)
        sifts.append(x_sift)
        y_true.append(class_to_id[sample.label])
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(test_samples)} images...")

    imgs_arr = np.asarray(imgs, dtype=np.float32)
    sifts_arr = np.asarray(sifts, dtype=np.float32)
    y_true_arr = np.asarray(y_true, dtype=np.int64)

    print("Running batch predictions...")
    if len(model.inputs) == 1:
        y_prob_arr = model.predict(imgs_arr, batch_size=32, verbose=1)
    else:
        y_prob_arr = model.predict([imgs_arr, sifts_arr], batch_size=32, verbose=1)

    y_pred_arr = np.argmax(y_prob_arr, axis=1).astype(np.int64)

    np.save(reports_dir / "test_truth.npy", y_true_arr)
    np.save(reports_dir / "test_pred.npy", y_pred_arr)
    np.save(reports_dir / "test_prob.npy", y_prob_arr)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro"))
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(len(labels))))
    report = classification_report(y_true_arr, y_pred_arr, target_names=labels, output_dict=True, zero_division=0)

    (reports_dir / "summary.json").write_text(
        json.dumps({"accuracy": acc, "macro_f1": macro_f1, "test_size": int(len(y_true_arr))}, indent=2),
        encoding="utf-8",
    )
    (reports_dir / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_title("EcoTexture AI Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(reports_dir / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# EcoTexture AI Evaluation",
        "",
        f"- Accuracy: {acc:.4f}",
        f"- Macro F1: {macro_f1:.4f}",
        f"- Test samples: {len(y_true_arr)}",
        "",
        "## Per-class F1",
    ]
    for label in labels:
        lines.append(f"- {label}: {report[label]['f1-score']:.4f}")

    (reports_dir / "evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"accuracy": acc, "macro_f1": macro_f1, "test_size": int(len(y_true_arr))}, indent=2))


if __name__ == "__main__":
    main()
