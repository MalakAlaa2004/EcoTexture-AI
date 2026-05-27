"""
Download TACO dataset (subset) for augmenting TrashNet training.
TACO annotations are on GitHub, images on Flickr — we pull what we can.
"""
import os, json, urllib.request, pathlib, sys, shutil, time
from collections import Counter

DATA_DIR = pathlib.Path(r"D:\EcoTexture AI\data\raw\taco")
DATA_DIR.mkdir(parents=True, exist_ok=True)

ANN_URL = "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
ANN_FILE = DATA_DIR / "annotations.json"

print("[TACO] Downloading annotations...")
try:
    req = urllib.request.Request(ANN_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        content = resp.read()
    ANN_FILE.write_bytes(content)
    print(f"[TACO] Annotations saved -> {ANN_FILE}")
except Exception as e:
    print(f"[TACO] Failed to download annotations: {e}")
    sys.exit(1)

with open(ANN_FILE, encoding="utf-8") as f:
    ann = json.load(f)

print(f"[TACO] Total images in dataset: {len(ann['images'])}")
print(f"[TACO] Total annotations: {len(ann['annotations'])}")
print(f"[TACO] Categories: {[c['name'] for c in ann['categories']]}")

# Map TACO categories -> TrashNet-compatible classes
TACO_MAP = {
    # Plastic
    "Plastic bottle": "Plastic", "Plastic bag & wrapper": "Plastic",
    "Drink carton": "Plastic", "Styrofoam piece": "Trash",
    # Metal
    "Aluminium foil": "Metal", "Can": "Metal",
    # Paper
    "Paper": "Paper", "Cardboard": "Cardboard",
    # Glass
    "Glass bottle": "Glass", "Glass jar": "Glass",
    # Trash / other
    "Cigarette": "Trash", "Rope & strings": "Trash", "Other plastic": "Plastic",
}

cat_id_to_name = {c["id"]: c["name"] for c in ann["categories"]}

# Build image_id -> best class label mapping
from collections import Counter
img_labels = {}
for a in ann["annotations"]:
    iid = a["image_id"]
    cat = cat_id_to_name.get(a["category_id"], "")
    label = TACO_MAP.get(cat, None)
    if label:
        img_labels.setdefault(iid, []).append(label)

# Resolve each image to majority class
resolved = {}
for iid, labels in img_labels.items():
    resolved[iid] = Counter(labels).most_common(1)[0][0]

print(f"[TACO] Images with mappable labels: {len(resolved)}")

# Save the image URL mapping to disk (images need separate download)
img_id_to_info = {img["id"]: img for img in ann["images"]}
mapping = []
for iid, label in resolved.items():
    info = img_id_to_info.get(iid, {})
    mapping.append({
        "image_id": iid,
        "file_name": info.get("file_name", ""),
        "flickr_url": info.get("flickr_url", ""),
        "label": label,
    })

mapping_file = DATA_DIR / "taco_label_mapping.json"
with open(mapping_file, "w", encoding="utf-8") as f:
    json.dump(mapping, f, indent=2)

print(f"[TACO] Label mapping saved -> {mapping_file}")

# Download as many images as possible (capped at 300 for speed)
import time

CLASSES = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]
class_dirs = {c: DATA_DIR / "images" / c for c in CLASSES}
for d in class_dirs.values():
    d.mkdir(parents=True, exist_ok=True)

downloaded = {c: 0 for c in CLASSES}
MAX_PER_CLASS = 60

print("[TACO] Downloading images (capped at 60 per class)...")
for item in mapping:
    url = item["flickr_url"]
    label = item["label"]
    if not url or label not in class_dirs:
        continue
    if downloaded[label] >= MAX_PER_CLASS:
        continue
    fname = class_dirs[label] / f"taco_{item['image_id']}.jpg"
    if fname.exists():
        downloaded[label] += 1
        continue
    try:
        urllib.request.urlretrieve(url, fname)
        downloaded[label] += 1
        if sum(downloaded.values()) % 20 == 0:
            print(f"  Progress: {dict(downloaded)}")
        time.sleep(0.1)
    except Exception:
        pass

print(f"[TACO] Download complete: {dict(downloaded)}")
print("[TACO] Done.")
