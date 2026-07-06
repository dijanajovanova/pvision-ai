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
    """Predict whether a solar panel is healthy or defective."""

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
            dim=1,
        )

    return (
        CLASS_NAMES[prediction.item()],
        confidence.item(),
    )


st.set_page_config(
    page_title="PVision AI",
    page_icon="☀️",
    layout="wide",
)

st.title("☀️ PVision AI")
st.caption("AI-powered photovoltaic panel inspection")

st.divider()

left_column, right_column = st.columns(2)

with left_column:

    st.subheader("Upload Image")

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

with right_column:

    st.subheader("Inspection Result")

    if uploaded_file:

        if st.button("Analyze Image", use_container_width=True):

            prediction, confidence = predict(image)

            if prediction == "Healthy":
                st.success("🟢 Panel Status: Healthy")
            else:
                st.error("🔴 Panel Status: Defective")

            st.metric(
                label="Confidence",
                value=f"{confidence:.2%}",
            )

            st.progress(confidence)

            st.subheader("Inspection Summary")

            if prediction == "Healthy":

                st.write(
                    """
                    No visible defects were detected.

                    The panel appears to be in good condition and
                    no immediate maintenance is recommended.
                    """
                )

            else:

                st.write(
                    """
                    A defect was detected with high confidence.

                    A manual inspection is recommended to verify
                    the damage before returning the panel to service.
                    """
                )

    else:

        st.info("Upload an image to start the inspection.")