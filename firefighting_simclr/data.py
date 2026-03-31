from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# a crop from an image based on a bounding box annotation, along with metadata about the category and split
@dataclass(frozen=True)
class CropRecord:
    image_path: Path
    bbox: Tuple[float, float, float, float]
    category_id: int
    category_name: str
    split: str
    image_id: int
    annotation_id: int

# provides two transformed views of the same input image for SimCLR contrastive learning
class SimCLRViewTransform:
    def __init__(self, image_size: int) -> None:
        blur_kernel = max(3, int(image_size * 0.1))
        if blur_kernel % 2 == 0:
            blur_kernel += 1

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4,
                            contrast=0.4,
                            saturation=0.2,
                            hue=0.05,
                        )
                    ],
                    p=0.6,
                ),
                transforms.RandomAffine(
                    degrees=8,
                    translate=(0.05, 0.05),
                    scale=(0.9, 1.1),
                    shear=4,
                ),
                transforms.RandomGrayscale(p=0.15),
                transforms.GaussianBlur(kernel_size=blur_kernel, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    # takes an input image and augment it twice for contrastive learning
    def __call__(self, image: Image.Image):
        return self.transform(image), self.transform(image)

# Build a simpler transform for supervised training, without the color jitter, grayscale, or blur.
def build_supervised_transform(image_size: int, train: bool) -> transforms.Compose:
    steps = [transforms.Resize((image_size, image_size))]
    if train:
        steps.append(
            transforms.RandomAffine(
                degrees=6,
                translate=(0.04, 0.04),
                scale=(0.92, 1.08),
                shear=3,
            )
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(steps)

# load the COCO-format JSON annotation file and return a list of CropRecord objects and a mapping of category IDs to names
def _load_coco_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

# load the records for a given split (train, valid, or test) 
# and return a list of CropRecord objects and a mapping of category IDs to names
def load_split_records(data_dir: Path | str, split: str) -> tuple[list[CropRecord], dict[int, str]]:
    data_dir = Path(data_dir)
    annotation_path = data_dir / split / "_annotations.coco.json"
    raw = _load_coco_json(annotation_path)

    categories = {
        int(category["id"]): category["name"]
        for category in raw["categories"]
        if int(category["id"]) != 0
    }
    image_by_id = {int(image["id"]): image for image in raw["images"]}

    records: list[CropRecord] = []
    for annotation in raw["annotations"]:
        category_id = int(annotation["category_id"])
        if category_id == 0:
            continue

        image = image_by_id[int(annotation["image_id"])]
        records.append(
            CropRecord(
                image_path=data_dir / split / image["file_name"],
                bbox=tuple(float(value) for value in annotation["bbox"]),
                category_id=category_id,
                category_name=categories[category_id],
                split=split,
                image_id=int(annotation["image_id"]),
                annotation_id=int(annotation["id"]),
            )
        )

    return records, categories

# build a mapping from category IDs to contiguous label indices starting from 0, 
# and return the mapping along with a list of class names in the order of the label indices
def build_label_mapping(data_dir: Path | str) -> tuple[dict[int, int], list[str]]:
    train_records, categories = load_split_records(data_dir, "train")
    train_category_ids = sorted({record.category_id for record in train_records})
    label_mapping = {
        category_id: label_index for label_index, category_id in enumerate(train_category_ids)
    }
    class_names = [categories[category_id] for category_id in train_category_ids]
    return label_mapping, class_names


def filter_records(records: Sequence[CropRecord], allowed_category_ids: Iterable[int]) -> list[CropRecord]:
    allowed = set(allowed_category_ids)
    return [record for record in records if record.category_id in allowed]

# extract a square crop from the image based on the bounding box, with optional padding, 
# and ensure that the crop is at least 1x1 pixels even for small boxes near the border of the image
def extract_crop(
    image: Image.Image,
    bbox: Tuple[float, float, float, float],
    padding_ratio: float,
) -> Image.Image:
    image_width, image_height = image.size
    x, y, width, height = bbox

    center_x = x + width / 2
    center_y = y + height / 2
    side = max(width, height) * (1.0 + padding_ratio * 2.0)

    left = max(0.0, center_x - side / 2.0)
    top = max(0.0, center_y - side / 2.0)
    right = min(float(image_width), center_x + side / 2.0)
    bottom = min(float(image_height), center_y + side / 2.0)

    # Guarantee at least a 1x1 crop even for very small boxes near the border.
    if right <= left:
        right = min(float(image_width), left + 1.0)
    if bottom <= top:
        bottom = min(float(image_height), top + 1.0)

    return image.crop((left, top, right, bottom))

# A PyTorch Dataset that loads crops of the images based on the bounding boxes in the records, 
# applies the given transform, and optionally maps category IDs to label indices.
class SymbolCropDataset(Dataset):
    def __init__(
        self,
        records: Sequence[CropRecord],
        transform: Optional[Callable] = None,
        label_mapping: Optional[dict[int, int]] = None,
        padding_ratio: float = 0.15,
    ) -> None:
        self.records = list(records)
        self.transform = transform
        self.label_mapping = label_mapping
        self.padding_ratio = padding_ratio

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.image_path) as image:
            image = image.convert("RGB")
            crop = extract_crop(image, record.bbox, self.padding_ratio)

        sample = self.transform(crop) if self.transform is not None else crop

        if self.label_mapping is None:
            return sample

        label = self.label_mapping[record.category_id]
        return sample, label

