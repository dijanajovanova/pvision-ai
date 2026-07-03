from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


MODEL_PATH = Path("models/pvision_classifier.pth")
CLASS_NAMES = ["Defective", "Healthy"]


@st.cache_resource
def load_model() -> torch.nn.Module:
    """Load the trained classification model."""

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu")
    )

    model.eval()

    return model


def predict(image: Image.Image) -> tuple[str, float]:
    """Predict the condition of a solar panel image."""

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = transform(image).unsqueeze(0)

    model = load_model()

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    return (
        CLASS_NAMES[prediction.item()],
        confidence.item(),
    )


st.set_page_config(
    page_title="PVision AI",
    page_icon="☀️",
    layout="centered",
)

st.title("PVision AI")

st.write(
    "Upload a solar panel image and let the model classify it as healthy or defective."
)

uploaded_file = st.file_uploader(
    "Select a JPG image",
    type=["jpg"],
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True,
    )

    if st.button("Analyze Image"):

        prediction, confidence = predict(image)

        st.subheader("Prediction")

        st.metric(
            label="Panel Status",
            value=prediction,
        )

        st.metric(
            label="Confidence",
            value=f"{confidence:.2%}",
        )