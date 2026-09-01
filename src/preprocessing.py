"""Shared image transforms for training and inference."""

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int = 224, training: bool = False):
    # UTKFace is aligned; center-cropping gives uploaded portraits the same square framing.
    operations = [transforms.Resize(image_size + 32), transforms.CenterCrop(image_size)]
    if training:
        operations.extend([transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.1, 0.1)])
    operations.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(operations)
