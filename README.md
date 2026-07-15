# PVision AI

PVision AI is a computer vision project that focuses on detecting defects in solar panels using deep learning.

The goal of this project is to build an AI system that can inspect solar panel images, classify them as healthy or defective, and later identify the exact location of the defect. In future versions, I plan to integrate object detection, an LLM for automatic inspection reports, and a simple web interface. 

---

## Built With
 
- Python
- PyTorch
- Torchvision
- YOLOv8
- OpenCV
- Scikit-learn

---

## Current Features

- Train an image classification model
- Evaluate the model
- Predict whether a solar panel is healthy or defective
- Save and load trained models

---

## Project Structure

```
PVision AI
│
├── app
│   ├── classification
│   └── cv
│
├── datasets
├── images
├── models
├── outputs
└── README.md
```

---

## Current Results

Validation Accuracy

80%

Confusion Matrix

```
[[20 6]
 [4 20]]
```

The model was trained using transfer learning with ResNet18 on a custom dataset of healthy and defective solar panel images.

---

## Run the Project

Train the model

```bash
python app/classification/train.py
```

Evaluate the model

```bash
python app/classification/evaluate.py
```

Predict a new image

```bash
python app/classification/predict.py
```

---

## Future Improvements

- Improve model accuracy
- Train on a larger dataset
- Add YOLO defect detection
- Build a Streamlit interface
- Generate AI inspection reports

---

This project is still under development and will continue to grow as I add new features and improve the model.
