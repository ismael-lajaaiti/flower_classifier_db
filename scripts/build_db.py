from flower_classifier_db.db import (
    get_engine,
    create_tables,
    get_session,
    ImageMetadata,
)
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import os

DATA_DIR = Path("data/tf_flowers")
DB_PATH = Path("data/flowers.db")


def build_metadata_db():

    if DB_PATH.exists():
        choice = (
            input(f"Warning: Database {DB_PATH} already exists. Overwrite? [y/N]: ")
            .strip()
            .lower()
        )
        if choice != "y":
            print("Aborted. Existing database preserved.")
            return
        else:
            os.remove(DB_PATH)
            print("Old database removed.")

    engine = get_engine(DB_PATH)
    create_tables(engine)
    session = get_session(engine)

    for class_dir in DATA_DIR.iterdir():
        if not class_dir.is_dir():
            continue  # Useful to ignore e.g. .DS_STORE.
        label = class_dir.name
        for img_path in tqdm(list(class_dir.glob("*.jpg")), desc=f"Processing {label}"):
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"Error reading {img_path}: {e}")
                continue

            record = ImageMetadata(
                path=str(img_path.relative_to(DATA_DIR.parent)),
                label=label,
                width=width,
                height=height,
            )
            session.add(record)

    session.commit()
    print(f"[x] Metadata DB created at {DB_PATH}")


if __name__ == "__main__":
    build_metadata_db()
