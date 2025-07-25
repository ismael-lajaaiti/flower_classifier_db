from pathlib import Path
import tensorflow_datasets as tfds
import tensorflow as tf
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

RAW_DIR = Path("data/tf_flowers_raw")
OUTPUT_DIR = Path("data/tf_flowers")


def download_tf_flowers():
    print("Downloading tf_flowers using TFDS...")
    ds, info = tfds.load(
        "tf_flowers",
        split="train",
        shuffle_files=False,
        as_supervised=True,
        data_dir=RAW_DIR,
        with_info=True,
    )

    label_names = info.features["label"].names
    print(f"[x] Classes: {label_names}")

    # Create class directories.
    for class_name in label_names:
        (OUTPUT_DIR / class_name).mkdir(parents=True, exist_ok=True)

    print("[x] Saving images to disk...")

    for i, (image, label) in enumerate(tqdm(ds)):
        class_name = label_names[int(label.numpy())]
        img_np = image.numpy()
        img_pil = Image.fromarray(img_np)
        img_path = OUTPUT_DIR / class_name / f"{i:05d}.jpg"
        img_pil.save(img_path)

    print(f"[x] Saved {i+1} images to {OUTPUT_DIR}")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    download_tf_flowers()
