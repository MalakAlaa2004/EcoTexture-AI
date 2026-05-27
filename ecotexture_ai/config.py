"""
EcoTexture AI — Central Configuration
======================================
Single source of truth for all paths, hyper-parameters and class metadata.
Hybrid architecture: CNN (EfficientNetB0) + SIFT BoVW (λ=50) + Cross-Attention Fusion.
"""
from __future__ import annotations

from pathlib import Path

# ─────────────────────────────────────────────────────────────
# PROJECT ROOT
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
RAW_DIR      = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS_DIR  = PROJECT_ROOT / "reports"
ASSETS_DIR   = PROJECT_ROOT / "assets"

# ─────────────────────────────────────────────────────────────
# MODEL HYPER-PARAMETERS
# ─────────────────────────────────────────────────────────────
IMG_SIZE      = (224, 224)         # Input resolution for CNN backbone
SIFT_K        = 300               # BoVW vocabulary size (λ=50)
SIFT_WEIGHT   = 50.0              # SIFT branch weight
EMBED_DIM     = 256               # Cross-attention fusion output dim
NUM_HEADS     = 4                 # Multi-head attention heads
TEMPERATURE   = 0.07              # Contrastive loss temperature
BATCH_SIZE    = 32
LR_STAGE1     = 1e-3              # Head-only stage
LR_STAGE2     = 1e-5              # Full fine-tune stage
EPOCHS_STAGE1 = 15
EPOCHS_STAGE2 = 10
MAX_UPLOAD_SIZE_MB = 10
SEED          = 42

# ─────────────────────────────────────────────────────────────
# WASTE CLASSES  (Arabic / English bilingual)
# ─────────────────────────────────────────────────────────────
WASTE_CLASSES = [
    "Cardboard",
    "Glass",
    "Metal",
    "Organic",
    "Paper",
    "Plastic_PET",
    "Plastic_HDPE",
    "Plastic_PVC",
    "Styrofoam",
    "Textile",
    "E-Waste",
    "Hazardous",
]

ARABIC_LABELS = {
    "Cardboard":    "كرتون",
    "Glass":        "زجاج",
    "Metal":        "معدن",
    "Organic":      "عضوي",
    "Paper":        "ورق",
    "Plastic_PET":  "بلاستيك PET",
    "Plastic_HDPE": "بلاستيك HDPE",
    "Plastic_PVC":  "بلاستيك PVC",
    "Styrofoam":    "فوم",
    "Textile":      "نسيج",
    "E-Waste":      "إلكترونيات",
    "Hazardous":    "مواد خطرة",
}

# ─────────────────────────────────────────────────────────────
# RECYCLING & IMPACT DATA
# ─────────────────────────────────────────────────────────────
RECYCLING_BINS = {
    "Cardboard":    "blue",
    "Glass":        "green",
    "Metal":        "yellow",
    "Organic":      "brown",
    "Paper":        "blue",
    "Plastic_PET":  "yellow",
    "Plastic_HDPE": "yellow",
    "Plastic_PVC":  "red",
    "Styrofoam":    "black",
    "Textile":      "purple",
    "E-Waste":      "orange",
    "Hazardous":    "red",
}

CO2_SAVINGS_KG = {
    "Cardboard":    0.9,
    "Glass":        0.3,
    "Metal":        4.0,
    "Organic":      0.5,
    "Paper":        0.8,
    "Plastic_PET":  1.5,
    "Plastic_HDPE": 1.2,
    "Plastic_PVC":  0.0,
    "Styrofoam":    0.0,
    "Textile":      3.6,
    "E-Waste":      2.0,
    "Hazardous":    0.0,
}

SDG_TAGS = {
    "Cardboard":    ["SDG 12", "SDG 15"],
    "Glass":        ["SDG 12"],
    "Metal":        ["SDG 9", "SDG 12"],
    "Organic":      ["SDG 2", "SDG 12", "SDG 13"],
    "Paper":        ["SDG 12", "SDG 15"],
    "Plastic_PET":  ["SDG 14", "SDG 12"],
    "Plastic_HDPE": ["SDG 12"],
    "Plastic_PVC":  ["SDG 3", "SDG 12"],
    "Styrofoam":    ["SDG 14"],
    "Textile":      ["SDG 12"],
    "E-Waste":      ["SDG 9", "SDG 12"],
    "Hazardous":    ["SDG 3", "SDG 12"],
}

# ─────────────────────────────────────────────────────────────
# EXISTING SIFT CENTERS (reuse from prior runs to accelerate)
# ─────────────────────────────────────────────────────────────
EXISTING_SIFT_CENTERS = [
    Path(r"D:\grad project\trial\cloud_deployment_backend\sift_kmeans_centers.npy"),
    Path(r"D:\timelens app\timelens_flutter_app\backend\deploy\sift_kmeans_centers.npy"),
]

# ── TEACHER CHECKPOINTS (COCO / MET research models) ─────────
COCO_TEACHER_CKPTS = [
    Path(r"D:\grad project\trial\benchmark_experiments_external\models\coco_mixed\ConvNeXt_Tiny\seed42_best.weights.h5"),
    Path(r"D:\grad project\trial\benchmark_experiments_external\models\coco_mixed\ViT_Tiny\seed42_best.weights.h5"),
    Path(r"D:\grad project\trial\benchmark_experiments_external\models\coco_mixed\CLIP_LinearProbe\seed42_best.weights.h5"),
]

MET_TEACHER_CKPTS = [
    Path(r"D:\grad project\trial\benchmark_experiments_external\models\met\ConvNeXt_Tiny\seed42_best.weights.h5"),
    Path(r"D:\grad project\trial\benchmark_experiments_external\models\met\ViT_Tiny\seed42_best.weights.h5"),
    Path(r"D:\grad project\trial\benchmark_experiments_external\models\met\CLIP_LinearProbe\seed42_best.weights.h5"),
]

EXISTING_HYBRID_CKPTS = [
    Path(r"D:\grad project\trial\hybrid\research_v2_contrastive\contrastive_final.h5"),
    Path(r"D:\grad project\trial\cloud_deployment_backend\contrastive_final.h5"),
]

# ─────────────────────────────────────────────────────────────
# DATASETS
# ─────────────────────────────────────────────────────────────
DATASET_REGISTRY = {
    "trashnet":  "https://github.com/garythung/trashnet",
    "taco":      "http://tacodataset.org",
    "garythung": "https://huggingface.co/datasets/garythung/trashnet",
    "wastenet":  "https://huggingface.co/datasets/Xenova/waste-classification-data",
}


def ensure_dirs() -> None:
    for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, ASSETS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
