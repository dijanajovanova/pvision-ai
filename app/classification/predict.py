from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


MODEL_PATH = Path("models/pvision_classifier.pth")

CLASS_NAMES = [
    "defective",
    "good"
]


def load_model(device: torch.device) -> torch.nn.Module:
    """Load the trained classifier."""

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )

    model.to(device)
    model.eval()

    return model


def predict_image(
    image_path: str,
    model: torch.nn.Module,
    device: torch.device,
):

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    return (
        CLASS_NAMES[prediction.item()],
        confidence.item()
    )


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = load_model(device)

    image_path = "images/test.jpg"

    prediction, confidence = predict_image(
        image_path,
        model,
        device,
    )

    print(f"\nImage       : {image_path}")
    print(f"Prediction  : {prediction}")
    print(f"Confidence  : {confidence:.2%}")


if __name__ == "__main__":
    main()