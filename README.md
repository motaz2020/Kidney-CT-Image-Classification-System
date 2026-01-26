# Kidney-CT-Image-Classification-System

## 📌 Project Overview

This project presents a multi-class medical image classification system for automated analysis of kidney CT scans.
The model classifies kidney images into four clinical categories:

Normal

Cyst

Tumor

Stone

The goal is to support computer-aided diagnosis (CAD) by leveraging deep learning techniques to enhance the accuracy and efficiency of kidney disease detection from CT imagery.

Medical image classification using CNNs is a well-established approach in radiology and diagnostic support systems.

## 🧠 System Pipeline

The project implements a complete computer vision and deep learning workflow, including:

Data preprocessing

Image resizing and normalization

Noise handling and contrast consistency

Data augmentation

Rotation, flipping, zooming, and shifting

To reduce overfitting and improve generalization

Modeling

Custom CNN architectures

Transfer Learning models (pretrained CNN backbones)

Training & Evaluation

Multi-class classification

Validation accuracy tracking

Confusion matrix and performance metrics

Inference

Predicting kidney condition from unseen CT scans

## 📊 Results

Achieved approximately 95% validation accuracy on multi-class classification.

Demonstrated strong separability between kidney disease categories.

Transfer learning models showed faster convergence and higher stability compared to training from scratch.

Deep learning has been shown to outperform traditional methods in kidney CT image analysis tasks.

## 🛠️ Technologies Used

Programming Language: Python

Deep Learning: TensorFlow, Keras

Image Processing: OpenCV, Pillow

Data Handling: NumPy, Pandas

Visualization: Matplotlib, Seaborn

## 🧬 Dataset

The project is based on a public Kidney CT Scan Image Dataset containing labeled samples of:

Normal kidneys

Kidney cysts

Kidney tumors

Kidney stones

Public kidney CT datasets are widely used in medical AI research and benchmarking.
