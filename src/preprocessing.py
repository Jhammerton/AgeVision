"""Shared image transforms for training and inference."""

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(
    image_size: int = 224,
    training: bool = False,
    strong_augmentation: bool = False,
):
    # UTKFace is aligned; center-cropping gives uploaded portraits the same square framing.
    operations = [transforms.Resize(image_size + 32), transforms.CenterCrop(image_size)]
    if training and strong_augmentation:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.2, 0.2, 0.15, 0.05)],
                    p=0.8,
                ),
                transforms.RandomAffine(
                    degrees=8,
                    translate=(0.04, 0.04),
                    scale=(0.95, 1.05),
                ),
                transforms.RandomGrayscale(p=0.05),
            ]
        )
    elif training:
        operations.extend([transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.1, 0.1)])
    operations.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(operations)
