# Kidney CT Image Classification System

> An end-to-end deep learning system for automated kidney CT scan classification, leveraging ResNet50 transfer learning and FastAPI deployment to support computer-aided diagnosis (CAD).

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Medical AI](https://img.shields.io/badge/Medical-AI-success)

---

# Overview

This project presents an end-to-end medical image classification pipeline for automatically identifying kidney conditions from CT scans. The system classifies images into four clinically relevant categories:

- Normal
- Cyst
- Tumor
- Stone

Built on a pretrained **ResNet50** backbone, the model applies transfer learning to achieve high classification accuracy while minimizing training time and overfitting. A lightweight **FastAPI** service exposes the trained model through a REST API for real-time inference.

---

# Features

- Four-class kidney CT image classification
- Transfer Learning with ResNet50
- End-to-end PyTorch training pipeline
- Medical image preprocessing & augmentation
- FastAPI inference service
- Confusion matrix and performance visualization
- Modular project structure
- Production-ready REST API

---

# System Architecture

```
Kidney CT Image
        │
Preprocessing
(Resize • Normalize • Augmentation)
        │
PyTorch DataLoader
        │
ResNet50 Backbone (Frozen)
        │
Classification Head
        │
Prediction
(Normal • Cyst • Tumor • Stone)
        │
FastAPI REST API
```

---

# Dataset

The project uses a public Kidney CT Scan dataset containing four diagnostic categories.

| Class | Description |
|--------|-------------|
| Normal | Healthy kidney |
| Cyst | Fluid-filled cyst |
| Tumor | Kidney tumor |
| Stone | Kidney stone |

The dataset is organized using the standard **ImageFolder** directory structure for PyTorch.

---

# Training Pipeline

The complete workflow includes:

- Image preprocessing
- Dataset normalization
- Data augmentation
- Transfer Learning
- Model optimization
- Validation
- Model checkpointing
- Performance evaluation

---

# Model

Backbone:

- ResNet50 (ImageNet pretrained)

Training Configuration:

- CrossEntropyLoss
- Adam Optimizer
- Batch Size = 32
- Input Resolution = 224×224
- Frozen feature extractor
- Fine-tuned classification head

---

# Inference API

The trained model is deployed using FastAPI.

```
POST /predict
```

Input:

```
Kidney CT Image
```

Output:

```json
{
  "prediction": "Tumor"
}
```

---

# Results

- ~95% Validation Accuracy
- Stable transfer learning convergence
- Strong class separability across four kidney conditions
- Faster convergence compared to training from scratch

---

# Technologies

- Python
- PyTorch
- Torchvision
- FastAPI
- NumPy
- Pillow
- Matplotlib
- Scikit-learn

---

# Repository Structure

```
Kidney-CT-Image-Classification-System
│
├── app
│   └── app.py
│
├── src
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── model.py
│   └── train.py
│
├── notebooks
├── outputs
├── requirements.txt
└── README.md
