import numpy as np
import pandas as pd
import os
import cv2
import random
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# ================================
# Parse command line arguments
# ================================
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train", "demo"], required=True,
                    help="Choose 'train' to train the model or 'demo' to load saved model")
args = parser.parse_args()

# ================================
# Load dataset
# ================================
trainLabels = pd.read_csv("trainLabels.csv", dtype={'image': str})
trainLabels['image'] = trainLabels['image'].str.strip().str.lower()

listing = os.listdir("Dataset/")
print("Total images:", len(listing))

filepaths, labels = [], []
for file in listing:
    base = os.path.basename("Dataset/" + file)
    fileName = os.path.splitext(base)[0]
    filepaths.append("Dataset/" + file)
    # assume matching entry exists; otherwise this will raise IndexError
    labels.append(trainLabels.loc[trainLabels.image == fileName, 'level'].values[0])

filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')
image_df = pd.concat([filepaths, labels], axis=1)
image_df['Label'] = image_df['Label'].astype(str)

# Train/Val/Test split
train_df, temp_df = train_test_split(image_df, test_size=0.30, stratify=image_df['Label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['Label'], random_state=42)

print("Train size:", len(train_df))
print("Validation size:", len(val_df))
print("Test size:", len(test_df))

# ================================
# Data Generators
# ================================
datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
    rotation_range=30, zoom_range=0.15,
    width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.15, horizontal_flip=True,
    fill_mode="nearest"
)

train_gen = datagen.flow_from_dataframe(train_df, x_col='Filepath', y_col='Label',
                                        target_size=(224,224), class_mode='categorical', batch_size=8, shuffle=True)

val_gen = datagen.flow_from_dataframe(val_df, x_col='Filepath', y_col='Label',
                                      target_size=(224,224), class_mode='categorical', batch_size=8, shuffle=True)

test_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input).flow_from_dataframe(
    test_df, x_col='Filepath', y_col='Label',
    target_size=(224,224), class_mode='categorical', batch_size=8, shuffle=False
)

class_labels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

# ================================
# Build Model (only for training mode)
# ================================
def build_model():
    base_model = tf.keras.applications.ResNet50(input_shape=(224,224,3), include_top=False,
                                                weights='imagenet', pooling='avg')
    # make last layers trainable
    for layer in base_model.layers[-30:]:
        layer.trainable = True

    x = tf.keras.layers.Dense(256, activation='relu')(base_model.output)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(5, activation='softmax')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

# ================================
# Grad-CAM Functions
# ================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv5_block3_out"):
    grad_model = tf.keras.models.Model([model.inputs],
                                       [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    # handle None grads gracefully
    if grads is None:
        grads = tf.zeros_like(conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        heatmap = tf.zeros_like(heatmap)
    else:
        heatmap = heatmap / max_val
    return heatmap.numpy()

def overlay_gradcam(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))

    # ensure heatmap is finite and normalized (avoid divide-by-zero)
    if np.isnan(heatmap).any():
        heatmap = np.nan_to_num(heatmap)
    hm_max = np.max(heatmap) if heatmap.size > 0 else 0
    if hm_max <= 0:
        heatmap_norm = np.zeros_like(heatmap, dtype=np.float32)
    else:
        heatmap_norm = heatmap / float(hm_max)

    heatmap_uint8 = np.uint8(255 * heatmap_norm)

    # use new Matplotlib colormap API (mpl.colormaps)
    jet = mpl.colormaps.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]   # values in [0,1]
    jet_heatmap = jet_colors[heatmap_uint8]   # map index -> RGB
    jet_heatmap = np.uint8(jet_heatmap * 255) # convert to 0-255 uint8

    jet_heatmap = cv2.resize(jet_heatmap, (img.shape[1], img.shape[0]))
    superimposed_img = cv2.addWeighted(img, 1 - alpha, jet_heatmap, alpha, 0)
    return superimposed_img

# ================================
# Training Mode (No Callbacks, Run All Epochs)
# ================================
if args.mode == "train":
    model = build_model()

    # Compute class weights for imbalance
    class_weights = dict(enumerate(len(train_df) / (train_df['Label'].value_counts().sort_index())))

    # Train for all epochs (no EarlyStopping or ReduceLROnPlateau)
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=50,  # all epochs will run
        class_weight=class_weights,
        verbose=1
    )

    # Save trained model
    model.save("dr_model.h5")
    print("✅ Model saved as dr_model.h5")

    # Save training plots
    pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot()
    plt.title("Accuracy")
    plt.savefig("accuracy.png")
    plt.close()

    pd.DataFrame(history.history)[['loss', 'val_loss']].plot()
    plt.title("Loss")
    plt.savefig("loss.png")
    plt.close()

    pd.DataFrame(history.history)[['auc', 'val_auc']].plot()
    plt.title("AUC")
    plt.savefig("auc.png")
    plt.close()

# ================================
# Demo Mode (Evaluation + Grad-CAM)
# ================================
if args.mode == "demo":
    if not os.path.exists("dr_model.h5"):
        raise FileNotFoundError("No trained model found! Run with --mode train first.")
    model = load_model("dr_model.h5")
    print("✅ Loaded saved model.")

# ================================
# Evaluation on Test Set
# ================================
# model must be defined (either from train or demo)
test_loss, test_acc, test_auc = model.evaluate(test_gen)
print(f"\nFinal Test Loss: {test_loss:.4f}")
print(f"Final Test Accuracy: {test_acc:.4f}")
print(f"Final Test AUC: {test_auc:.4f}")

pred_probs = model.predict(test_gen)
y_pred = np.argmax(pred_probs, axis=1)
y_true = test_gen.classes

report = classification_report(y_true, y_pred, target_names=class_labels)
kappa = cohen_kappa_score(y_true, y_pred)

print("\nClassification Report:\n", report)
print(f"Cohen’s Kappa: {kappa:.4f}")

# Save metrics report
with open("metrics_report.txt", "w") as f:
    f.write(report)
    f.write(f"\nCohen’s Kappa: {kappa:.4f}\n")

# Confusion Matrix
cmatrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cmatrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix - Test Set")
plt.savefig("confusion_matrix.png")
plt.close()

# ================================
# Grad-CAM Results
# ================================
os.makedirs("gradcam_results", exist_ok=True)
# pick 5 random samples from test_df (if dataset smaller, adjust)
sample_count = min(5, len(test_df))
sample_indices = random.sample(range(len(test_df)), sample_count)

gradcam_display = []
for idx in sample_indices:
    img_path = test_df.iloc[idx]['Filepath']
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Could not read image: {img_path}")
        continue
    img = cv2.resize(img, (224,224))
    img_array = np.expand_dims(img, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

    heatmap = make_gradcam_heatmap(img_array, model)
    gradcam_img = overlay_gradcam(img_path, heatmap)

    preds = model.predict(img_array, verbose=0)
    pred_class = np.argmax(preds[0])
    true_class = int(test_df.iloc[idx]['Label'])

    save_path = f"gradcam_results/{os.path.basename(img_path)}"
    cv2.imwrite(save_path, gradcam_img)
    print(f"✅ Grad-CAM saved: {save_path} | True: {class_labels[true_class]}, Pred: {class_labels[pred_class]}")

    gradcam_display.append((img_path, gradcam_img, class_labels[true_class], class_labels[pred_class]))

# Show 2 Grad-CAMs Live (if available)
if len(gradcam_display) >= 2:
    print("\n📊 Showing 2 Grad-CAM examples live...")
    for orig_path, grad_img, true_class, pred_class in random.sample(gradcam_display, 2):
        print(f"🔍 Displaying Grad-CAM for: {os.path.basename(orig_path)} | True: {true_class}, Pred: {pred_class}")

        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)
        plt.imshow(cv2.cvtColor(cv2.imread(orig_path), cv2.COLOR_BGR2RGB))
        plt.title(f"True: {true_class}")
        plt.axis("off")

        plt.subplot(1,2,2)
        plt.imshow(cv2.cvtColor(grad_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Pred: {pred_class}")
        plt.axis("off")
        plt.show()
else:
    print("Not enough grad-cam samples to display live.")
