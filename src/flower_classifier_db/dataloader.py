import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path


class FlowerDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path("data") / row["path"]
        image = Image.open(img_path).convert("RGB")
        label_id = int(row["label_id"])
        img_id = row.id
        if self.transform:
            image = self.transform(image)
        return image, label_id, img_id


def get_dataloaders_from_csv(split_csv, batch_size=32):
    df = pd.read_csv(split_csv)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    train_ds = FlowerDataset(df[df["split"] == "train"], transform)
    val_ds = FlowerDataset(df[df["split"] == "val"], transform)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    return train_dl, val_dl
