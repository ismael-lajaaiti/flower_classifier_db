from flower_classifier_db.data import get_dataloaders

train_loader, val_loader, classes = get_dataloaders()
print(f"[x] Loaded {len(classes)} classes: {classes}")

images, labels = next(iter(train_loader))
print(f"[x] Batch shape: {images.shape}")
print(f"[x] Labels: {labels}")
