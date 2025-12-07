Diabetic Retinopathy Detection and Classification using ResNet-50

This project presents an automated deep learning–based system for early detection and multi-class classification of Diabetic Retinopathy (DR) from retinal fundus images. DR is a major microvascular complication of diabetes and a leading cause of preventable blindness worldwide. Early-stage detection is critical, but manual diagnosis is time-consuming, subjective, and requires expert ophthalmologists — making large-scale screening difficult in rural and resource-limited settings.

This model uses ResNet-50, a powerful convolutional neural network, along with transfer learning and multiple preprocessing techniques, to classify retinal images into the five DR severity stages:

No DR

Mild DR

Moderate DR

Severe DR

Proliferative DR

Features

Automated DR detection and multi-class classification

Powered by ResNet-50 with transfer learning

Preprocessing includes resizing, normalization, and augmentation

Handles dataset imbalance using augmentation

Evaluation metrics include Accuracy, AUC, Precision, Recall, F1-score, and Cohen’s Kappa

Model interpretability using Grad-CAM heatmaps

Trained on APTOS 2019 Blindness Detection dataset

Suitable for large-scale screening

Provides transparent predictions appropriate for clinical assistance

Dataset

The project uses the APTOS 2019 Blindness Detection dataset from Kaggle, which contains over 3,500 high-resolution retinal fundus images annotated by medical experts with DR severity labels (0–4).

Model Architecture

Base network: ResNet-50 pretrained on ImageNet

Fine-tuned top layers for DR classification

Optimized using:

Learning rate scheduling

Batch normalization

Early stopping

Adam optimizer

Preprocessing and Data Augmentation

Image resizing (224×224)

Normalization

Random rotation

Horizontal flip

Zoom range

Brightness adjustments

These techniques help reduce overfitting and improve generalization, especially for imbalanced medical datasets.

Evaluation Metrics

Accuracy

AUC Score

Confusion Matrix

Precision and Recall

F1-Score

Cohen’s Kappa

Grad-CAM visual explanations

These metrics ensure quantitative reliability and clinical usefulness.

Results

The model achieves high classification accuracy across all severity levels.
Grad-CAM visualizations highlight important retinal regions influencing predictions, improving model interpretability and trust for healthcare professionals.

Future Enhancements

Testing additional models such as VGG16, InceptionV3, and EfficientNet

Hyperparameter tuning using Grid Search Cross-Validation (GSCV)

Deployment as a web or mobile-based screening tool

Integration with tele-ophthalmology platforms for remote diagnosis

Tech Stack

Python

TensorFlow / Keras

OpenCV

NumPy

Matplotlib
