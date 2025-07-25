from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from pathlib import Path
import torch


def get_dataloaders(
    data_dir: str = "data/tf_flowers/", batch_size: int = 32, train_prop: float = 0.8
):

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = datasets.ImageFolder(root=Path(data_dir), transform=transform)
    n = len(dataset)

    train_size = int(train_prop * n)
    val_size = n - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, dataset.classes
