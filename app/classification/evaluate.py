from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


MODEL_PATH = Path("models/pvision_classifier.pth")
VALIDATION_PATH = Path("datasets/validation")


def load_validation_dataset() -> tuple[DataLoader, list[str]]:
    """Load validation dataset."""

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(
        root=VALIDATION_PATH,
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
    )

    print(f"Validation images : {len(dataset)}")
    print(f"Classes           : {', '.join(dataset.classes)}\n")

    return dataloader, dataset.classes


def load_model(device: torch.device) -> torch.nn.Module:
    """Load the trained ResNet18 model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )

    model.to(device)
    model.eval()

    return model


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Evaluate model predictions."""

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)

            outputs = model(images)

            predictions = torch.argmax(outputs, dim=1)

            y_true.extend(labels.numpy())
            y_pred.extend(predictions.cpu().numpy())

    return y_true, y_pred


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}\n")

    validation_loader, class_names = load_validation_dataset()

    model = load_model(device)

    y_true, y_pred = evaluate(
        model,
        validation_loader,
        device,
    )

    accuracy = accuracy_score(y_true, y_pred)

    print(f"Validation Accuracy: {accuracy:.2%}\n")

    print("Classification Report\n")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
        )
    )

    print("Confusion Matrix\n")

    print(
        confusion_matrix(
            y_true,
            y_pred,
        )
    )


if __name__ == "__main__":
    main()