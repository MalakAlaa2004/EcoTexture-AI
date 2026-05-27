
import sys
from pathlib import Path
from datasets import load_dataset

OUTPUT_DIR = Path(r"D:\EcoTexture AI\data\raw\trashnet")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading garythung/trashnet from HuggingFace...")
try:
    ds = load_dataset("garythung/trashnet", split="train", trust_remote_code=True)
    print(f"  Total samples: {len(ds)}")
    
    label_names = ds.features["label"].names if hasattr(ds.features["label"], "names") else None
    
    for i, sample in enumerate(ds):
        label = sample["label"]
        if isinstance(label, int) and label_names:
            label_str = label_names[label]
        else:
            label_str = str(label)
        
        # Normalize class name to match TrashNet structure
        label_str = label_str.lower().replace(" ", "_")
        
        cls_dir = OUTPUT_DIR / label_str
        cls_dir.mkdir(parents=True, exist_ok=True)
        
        img = sample["image"]
        img_path = cls_dir / f"{i:05d}.jpg"
        if not img_path.exists():
            img.save(img_path, format="JPEG", quality=90)
        
        if i % 200 == 0:
            print(f"  Saved {i} images...")
    
    print("Download complete.")
    
    # Print final counts
    for cls_dir in sorted(OUTPUT_DIR.iterdir()):
        if cls_dir.is_dir():
            count = len(list(cls_dir.glob("*.jpg")))
            print(f"  {cls_dir.name}: {count} images")

except Exception as e:
    print(f"HuggingFace download failed: {e}")
    print("Trying alternative: Xenova/waste-classification-data ...")
    try:
        ds2 = load_dataset("Xenova/waste-classification-data", split="train", trust_remote_code=True)
        label_names2 = ds2.features["label"].names if hasattr(ds2.features["label"], "names") else None
        for i, sample in enumerate(ds2):
            label = sample["label"]
            if isinstance(label, int) and label_names2:
                label_str = label_names2[label].lower().replace(" ", "_")
            else:
                label_str = str(label).lower()
            cls_dir = OUTPUT_DIR / label_str
            cls_dir.mkdir(parents=True, exist_ok=True)
            img = sample["image"]
            img_path = cls_dir / f"w{i:05d}.jpg"
            if not img_path.exists():
                img.save(img_path, format="JPEG", quality=90)
            if i % 200 == 0:
                print(f"  Alt dataset: saved {i} images...")
        print("Alternative download complete.")
    except Exception as e2:
        print(f"Both downloads failed: {e2}")
        sys.exit(1)
