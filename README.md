# ☀️ PVision AI

PVision AI is a computer vision project focused on detecting defects in solar panels using deep learning.

The goal of this project is to build an AI-powered inspection system capable of analyzing photovoltaic panel images, classifying them as **Healthy** or **Defective**, and generating professional inspection reports. As the project evolves, I plan to expand it into a complete inspection platform that can identify different types of defects, locate them on the panel, and automatically generate detailed engineering reports.

## Built With

- Python
- PyTorch
- Torchvision
- Streamlit
- ReportLab
- OpenCV
- Scikit-learn

## Current Features

- Train an image classification model using ResNet18
- Evaluate model performance
- Predict whether a solar panel is healthy or defective
- Streamlit web application for image upload and inspection
- Generate professional PDF inspection reports
- Display prediction confidence and AI inspection summary
- Save and load trained models

## Project Structure

```text
PVision AI
│
├── app
│   ├── classification
│   ├── reports
│   ├── ui
│   └── cv
│
├── datasets
├── images
├── models
├── outputs
├── reports
└── README.md
```

## Current Results

**Validation Accuracy**

**80%**

The current model uses **ResNet18** with transfer learning on a custom dataset of healthy and defective solar panel images.

## Run the Project

### Train the model

```bash
python app/classification/train.py
```

### Evaluate the model

```bash
python app/classification/evaluate.py
```

### Run the Streamlit application

```bash
python -m streamlit run app/ui/main.py
```

## Future Improvements

- Improve model accuracy using a larger dataset
- Train the model to recognize multiple defect types
- Integrate YOLOv8 for defect localization
- Detect cracks, hotspots, broken cells, dust and other defects
- Generate AI-assisted engineering inspection reports
- Improve the Streamlit user interface
- Deploy the application online

---

**Project Status:** 🚧 Active Development

This project is actively under development. New features and improvements are continuously being added as I expand PVision AI into a complete AI-powered solar panel inspection platform.

