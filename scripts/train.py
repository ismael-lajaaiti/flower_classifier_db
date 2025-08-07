import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from flower_classifier_db.dataloader import get_dataloaders_from_csv
from flower_classifier_db.database import (
    get_engine,
    get_session,
    TrainingLog,
    ImageMetadata,
)
from pathlib import Path
from tqdm import tqdm
from sqlalchemy import inspect

# 0. Setup DB session.
db_path = str(Path("data/flowers.db"))
engine = get_engine(db_path)
session = get_session(engine)
class_names = session.query(ImageMetadata.label).distinct().all()


def log_epoch(session, epoch, train_loss, train_acc, val_loss, val_acc):
    log_entry = TrainingLog(
        epoch=epoch,
        train_loss=train_loss,
        train_accuracy=train_acc,
        val_loss=val_loss,
        val_accuracy=val_acc,
    )
    session.add(log_entry)
    session.commit()


if inspect(engine).has_table("training_logs"):
    session.query(TrainingLog).delete()
    session.commit()


# 1. Configuration.
epochs = 7
data_dir = str(Path("data/tf_flowers"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load data.
batch_size = 32
split_file = "data/split.csv"
train_loader, val_loader = get_dataloaders_from_csv(split_file, batch_size=batch_size)

# 3. Model setup.
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)

# 4. Loss and optimizer.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 5. Training Loop.
for epoch in range(epochs):
    # --- Training. ---
    model.train()
    train_loss = 0.0
    train_correct = 0

    print(f"\nEpoch {epoch + 1}/{epochs} - Training")
    for images, labels, _ in tqdm(train_loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()

    train_loss = train_loss / len(train_loader.dataset)
    train_acc = train_correct / len(train_loader.dataset)

    # --- Validation. ---
    model.eval()
    val_loss = 0.0
    val_correct = 0

    print(f"Validating..")
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_acc = val_correct / len(val_loader.dataset)

    # ---- Print and save results. ----
    log_epoch(session, epoch + 1, train_loss, train_acc, val_loss, val_acc)
    print(
        f"[Epoch {epoch+1}] "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
    )
    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), f"models/flower_classifier_epoch{epoch+1}.pt")

session.close()
