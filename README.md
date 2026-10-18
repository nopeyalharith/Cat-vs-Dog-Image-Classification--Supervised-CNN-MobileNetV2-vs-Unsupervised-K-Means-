# Cat vs Dog Image Classification: Supervised vs Unsupervised Learning

A machine learning project (SAIA 2113, Universiti Teknologi Malaysia) that tackles
binary cat/dog image classification with three approaches and compares them on the
same dataset:

1. **Plain CNN**, a custom convolutional network trained from scratch
2. **MobileNetV2 transfer learning**, an ImageNet-pretrained model with a retrained classifier head
3. **K-Means clustering**, an unsupervised baseline run on PCA-reduced pixel features

## Dataset
- Source: [Cats and Dogs (Kaggle)](https://www.kaggle.com/datasets/marquis03/cats-and-dogs)
- 345 images in 2 classes (cat, dog)
- 275 training / 70 validation (about an 80/20 split)

## Methodology
- **Preprocessing:** resize to 128×128, normalise pixels to [0, 1], and augment the
  training set only (rotation, zoom, shifts, horizontal flip)
- **Plain CNN:** three convolutional blocks with BatchNormalization and Dropout,
  trained with Adam, binary cross-entropy and early stopping
- **MobileNetV2:** frozen ImageNet base with a GlobalAveragePooling and Dense head
- **K-Means:** images flattened and normalised, reduced with PCA to 50 dimensions,
  k = 2 chosen with the Elbow Method, and cluster IDs aligned to class labels

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Plain CNN | 65.71% | 0.6571 | 1.0000 | 0.7931 |
| MobileNetV2 (transfer learning) | **95.71%** | 0.9574 | 0.9783 | 0.9677 |
| K-Means (unsupervised) | ~59.71% | n/a | n/a | n/a |

**Key findings**
- The plain CNN overfit the small dataset and predicted "dog" for every image
  (perfect recall, zero cat precision).
- Transfer learning raised validation accuracy by about 30 points over the plain CNN
  with only 275 training images.
- K-Means performed only slightly better than chance. Flattening images discards
  spatial structure, and pixel values alone don't separate the classes well.

## Repository Structure
├── supervised_learning.ipynb              # Plain CNN + MobileNetV2 (Parts 1-2)
├── SAIA2113_Part3_Part4.ipynb             # K-Means clustering + comparison
├── train/ , val/                          # Image data (cat/ and dog/ folders)
├── train.csv , val.csv                    # File paths and labels
└── report.docx                            # Full written report

## Tech Stack
Python, TensorFlow/Keras, scikit-learn, NumPy, Matplotlib, Seaborn

## Authors
Naufal, Harith, and Ahmad Faiz Fahmi, Faculty of Artificial Intelligence, UTM

