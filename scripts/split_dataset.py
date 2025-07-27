from flower_classifier_db.database import get_engine, get_session, ImageMetadata
from flower_classifier_db.utils import get_label_to_id
import random
import csv
from pathlib import Path
import pandas as pd

DATASET_DIR = Path("data/tf_flowers")
OUTPUT_CSV = Path("data/split.csv")
VAL_RATIO = 0.2

# Collect all image paths with labels.
samples = []
for class_dir in DATASET_DIR.iterdir():
    if not class_dir.is_dir():
        continue
    label = class_dir.name
    for img_path in class_dir.glob("*.jpg"):
        samples.append((str(img_path.relative_to(DATASET_DIR.parent)), label))

# Shuffle and split.
random.seed(123)
random.shuffle(samples)
split_idx = int(len(samples) * (1 - VAL_RATIO))
train_samples = samples[:split_idx]
val_samples = samples[split_idx:]

engine = get_engine()
session = get_session(engine)
records = session.query(ImageMetadata.path, ImageMetadata.id).all()
path_to_id = {path: image_id for path, image_id in records}

# Write split to CSV.
label_to_id = get_label_to_id()
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "path", "label", "label_id", "split"])  # Header.
    for path, label in train_samples:
        writer.writerow([path_to_id[path], path, label, label_to_id[label], "train"])
    for path, label in val_samples:
        writer.writerow([path_to_id[path], path, label, label_to_id[label], "val"])
print("[x] data/split.csv created.")

# Write split to DB.
df = pd.read_csv("data/split.csv")  # your CSV with 'path' and 'split' columns
split_map = dict(zip(df["path"], df["split"]))

for path, split in split_map.items():
    record = session.query(ImageMetadata).filter_by(path=path).first()
    if record:
        record.split = split

session.commit()
print("[x] Split values populated.")
