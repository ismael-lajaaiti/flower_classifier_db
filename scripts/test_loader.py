from flower_classifier_db.dataloader import get_dataloaders_from_csv

CSV_PATH = "data/split.csv"

train_loader, val_loader = get_dataloaders_from_csv(CSV_PATH)

print(f"[x] Loaded {len(classes)} classes: {classes}")

images, labels = next(iter(train_loader))
print(f"[x] Batch shape: {images.shape}")
print(f"[x] Labels: {labels}")
