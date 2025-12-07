# Diabetic Retinopathy Detection and Classification using ResNet-50

This project presents an automated deep learning–based system for early detection and multi-class classification of Diabetic Retinopathy (DR) from retinal fundus images. DR is a major microvascular complication of diabetes and a leading cause of preventable blindness worldwide. Early-stage detection is critical, but manual diagnosis is time-consuming, subjective, and requires expert ophthalmologists, making large-scale screening difficult in rural and resource-limited settings.

This model uses the ResNet-50 convolutional neural network with transfer learning and advanced preprocessing methods to classify retinal fundus images into five DR severity stages:

- No DR  
- Mild DR  
- Moderate DR  
- Severe DR  
- Proliferative DR  

---

## Features

- Automated detection and classification of Diabetic Retinopathy  
- ResNet-50-based transfer learning model  
- Preprocessing steps include resizing, normalization, and data augmentation  
- Handles dataset imbalance using augmentation techniques  
- Evaluation metrics include Accuracy, AUC, Precision, Recall, F1-score, and Cohen’s Kappa  
- Improved model interpretability using Grad-CAM visualizations  
- Trained on the APTOS 2019 Blindness Detection dataset  
- Suitable for large-scale, real-world screening applications  

---

## Dataset

The model is trained on the APTOS 2019 Blindness Detection Dataset (Kaggle), which contains more than 3,500 high-resolution retinal fundus images annotated by medical experts with DR severity levels (0–4). Each label corresponds to a specific severity stage of Diabetic Retinopathy.

---

## Model Architecture

- Base model: ResNet-50 pretrained on ImageNet  
- Customized classification layers added for DR severity prediction  
- Optimization techniques used:  
  - Learning rate scheduling  
  - Batch normalization  
  - Early stopping  
  - Adam optimizer  

---

## Preprocessing and Data Augmentation

The following preprocessing techniques are applied to enhance model performance and reduce overfitting:

- Image resizing to 224×224  
- Pixel normalization  
- Random rotation  
- Horizontal flipping  
- Zoom transformations  
- Brightness adjustments  

These techniques help the model generalize well across diverse fundus images.

---

## Evaluation Metrics

The model performance is assessed using the following metrics:

- Accuracy  
- AUC (Area Under the ROC Curve)  
- Precision  
- Recall  
- F1-Score  
- Cohen’s Kappa  
- Confusion Matrix  
- Grad-CAM visual explanations  

---

## Results

The trained ResNet-50 model demonstrates strong predictive performance across all five DR severity categories.  
Grad-CAM heatmaps provide interpretability by highlighting the retinal regions most influential in the model’s decision-making process.

---

## Future Enhancements

- Testing additional models such as VGG16, InceptionV3, and EfficientNet  
- Hyperparameter tuning using Grid Search Cross-Validation (GSCV)  
- Deployment as a cloud-based or mobile application for real-time screening  
- Integration with tele-ophthalmology systems  

---

## Tech Stack

- Python  
- TensorFlow / Keras  
- NumPy  
- OpenCV  
- Matplotlib  
- Seaborn  
- Pillow 
