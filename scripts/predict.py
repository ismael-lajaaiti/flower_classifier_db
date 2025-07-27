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
import pandas as pd

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

# 3. Load model.
class_names = session.query(ImageMetadata.label).distinct().all()
model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("models/flower_classifier.pt", map_location=device))
model.to(device)
model.eval()

# 4. Predict and store predictions.
for images, labels in val_loader:
    images = images.to(device)
    outputs = model(images)
    probs = torch.softmax(outputs, dim=1)
    confidences, preds = torch.max(probs, dim=1)

    for i in range(len(images)):
        image_id = int(val_df.iloc[i]["id"])  # get image ID from val_df
        true_label = id_to_label[labels[i].item()]
        predicted_label = id_to_label[preds[i].item()]
        confidence = confidences[i].item()

        session.add(
            Prediction(
                image_id=image_id,
                true_label=true_label,
                predicted_label=predicted_label,
                confidence=confidence,
            )
        )

session.commit()
print("[x] Predictions saved to DB.")
