from flower_classifier_db.database import (
    get_engine,
    get_session,
    Prediction,
    ImageMetadata,
)
from flower_classifier_db.dataloader import get_dataloaders_from_csv
from flower_classifier_db.utils import get_id_to_label
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import pandas as pd
from sqlalchemy import inspect
import os
import re

# 1. Setup session and dataloader.
engine = get_engine()
session = get_session(engine)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load dataloader and dataframe.
batch_size = 32
split_file = "data/split.csv"
df = pd.read_csv(split_file)
val_df = df[df["split"] == "val"].reset_index(drop=True)
_, val_loader = get_dataloaders_from_csv(split_file, batch_size=batch_size)
id_to_label = get_id_to_label()
label_count = len(id_to_label)

# Clean table.
if inspect(engine).has_table("predictions"):
    session.query(Prediction).delete()
    session.commit()

# 3. Get all saved models.
model_dir = "models"
model_files = sorted(
    [f for f in os.listdir(model_dir) if f.endswith(".pt") and "epoch" in f]
)

if not model_files:
    print("No model found in 'models/'.")
    exit()


# 4. Predict for each model.
for model_file in model_files:
    match = re.search(r"epoch(\d+)", model_file)
    epoch_num = int(match.group(1)) if match else None

    # Load model
    class_names = [label[0] for label in session.query(ImageMetadata.label).distinct()]
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, label_count)
    model.load_state_dict(
        torch.load(os.path.join(model_dir, model_file), map_location=device)
    )
    model = model.to(device)
    model.eval()

    # Predict
    for images, labels, ids in val_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        confidences, preds = torch.max(probs, dim=1)

        for i in range(len(images)):
            img_id = ids[i].item()
            true_label = id_to_label[labels[i].item()]
            predicted_label = id_to_label[preds[i].item()]
            confidence = confidences[i].item()
            session.add(
                Prediction(
                    image_id=img_id,
                    true_label=true_label,
                    predicted_label=predicted_label,
                    confidence=confidence,
                    epoch=epoch_num,
                )
            )

    session.commit()
    print(f"[✓] Predictions from {model_file} saved.")

session.close()
print("\nAll predictions saved.")
