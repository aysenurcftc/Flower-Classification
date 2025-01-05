import os
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List
from PIL import Image
from pathlib import Path


def create_dataloaders(
    train_dir: str, test_dir: str, transform: transforms.Compose, batch_size: int
):
    """Creates training and testing DataLoaders.

    Args:
        train_dir (str) : path to training directory
        test_dir (str) : path to testing directory
        transform : torchvision transforms to perform on training and testing data.
        batch_size: Number of samples per batch

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
    """

    train_data = ImageFolder(train_dir, transform=transform)
    test_data = ImageFolder(test_dir, transform=transform)

    class_names = train_data.classes

    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, pin_memory=True
    )

    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, pin_memory=True
    )

    return train_dataloader, test_dataloader, class_names


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds subdirectories in the given directory and maps class names to indices.

    Args:
        directory (str):  Path to the directory

    Returns:
        Tuple[List[str], Dict[str, int]]
        * A sorted list of class names
        * A dictionary mapping class names to unique indices
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    return classes, class_to_idx


class ImageFolder(Dataset):
    def __init__(self, target_dir: str, transform=None) -> None:
        self.paths = list(Path(target_dir).glob("*/*.jpg"))
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(target_dir)

    def load_image(self, index: int) -> Image.Image:
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        img = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx, class_name
