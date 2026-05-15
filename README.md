# Skin Cancer Detection using Deep Learning

A deep learning project for automated skin cancer detection using convolutional neural networks (CNN) on the ISIC 2024 Challenge dataset.

## 📋 Project Overview

This project develops a CNN-based model to classify skin lesions and detect skin cancer from dermoscopic images. The model is trained on the ISIC (International Skin Imaging Collaboration) 2024 Challenge dataset, which contains thousands of high-quality skin lesion images.

**Key Objective:** Build an accurate and efficient deep learning model to assist dermatologists in identifying malignant melanoma and other skin cancer types.

## 📊 Dataset

- **Source:** [ISIC 2024 Challenge on Kaggle](https://www.kaggle.com/competitions/isic-2024-challenge)
- **Size:** Thousands of dermoscopic images
- **Format:** JPG images with corresponding labels
- **Classes:** Multiple skin lesion categories including melanoma and benign nevi
- **Train/Test Split:** Separate training and test sets provided

### Dataset Structure
```
isic-2024-challenge/
├── train-image/
│   └── image/
│       ├── ISIC_9874656.jpg
│       ├── ISIC_9874659.jpg
│       └── ... (thousands of images)
├── train-metadata.csv
└── test-image/
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Google Colab (recommended) or local GPU
- Kaggle API credentials

### Installation & Setup

1. **Install Required Libraries**
   ```python
   !pip install kaggle
   ```

2. **Mount Google Drive** (if using Colab)
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Configure Kaggle API**
   - Download `kaggle.json` from [Kaggle Account Settings](https://www.kaggle.com/settings/account)
   - Place it in your Google Drive or local machine
   ```python
   !mkdir -p ~/.kaggle
   !cp /content/drive/MyDrive/ColabNotebooks/kaggle.json ~/.kaggle/
   !chmod 600 ~/.kaggle/kaggle.json
   ```

4. **Download ISIC 2024 Dataset**
   ```python
   !kaggle competitions download -c isic-2024-challenge
   !unzip isic-2024-challenge.zip
   ```

## 📁 Project Structure

```
Skin-Cancer-Detection/
├── DL_PROJECT.ipynb          # Main Jupyter notebook with full pipeline
├── README.md                 # This file
├── train-image/              # Training images
│   └── image/
├── test-image/               # Test images (optional)
│   └── image/
└── train-metadata.csv        # Image labels and metadata
```

## 🔬 Model Architecture

The project uses a **Convolutional Neural Network (CNN)** with the following approach:

- **Base Model:** Transfer learning with pre-trained networks (ResNet, EfficientNet, or VGG)
- **Input Size:** 224×224 RGB images
- **Data Augmentation:** Rotation, flip, zoom, brightness adjustment
- **Normalization:** ImageNet normalization
- **Output:** Binary or multi-class classification

### Key Layers
- Convolutional blocks for feature extraction
- Max pooling for dimensionality reduction
- Dropout layers for regularization
- Fully connected layers for classification
- Softmax activation for multi-class output

## 💻 Usage

### Running the Notebook

1. Open `DL_PROJECT.ipynb` in Google Colab or Jupyter
2. Execute cells sequentially:
   - **Setup:** Install dependencies and mount storage
   - **Data Loading:** Load ISIC images and metadata
   - **Preprocessing:** Normalize and augment images
   - **Model Training:** Train the CNN on training data
   - **Evaluation:** Assess performance on validation/test data
   - **Visualization:** Plot results and predictions

### Example Code

```python
# Load and preprocess data
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Build and compile model
from tensorflow.keras import Sequential
model = Sequential([...])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(train_datagen, epochs=50, validation_data=val_datagen)
```

## 📈 Results

*Results will be updated after model training completion*

### Expected Performance Metrics
- **Accuracy:** Percentage of correct predictions
- **Precision:** True positives / (True positives + False positives)
- **Recall:** True positives / (True positives + False negatives)
- **F1-Score:** Harmonic mean of precision and recall
- **ROC-AUC:** Area under the receiver operating characteristic curve

### Visualizations
- Training/validation accuracy and loss curves
- Confusion matrix
- ROC curves per class
- Sample predictions with confidence scores

## 🔧 Customization

You can modify the following parameters:

```python
# Model hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
IMAGE_SIZE = (224, 224)

# Data split
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
```

## 📚 Dependencies

- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- OpenCV
- Kaggle API

## 🎯 Features

✅ Automated data downloading from Kaggle  
✅ Image preprocessing and augmentation  
✅ Transfer learning with pre-trained models  
✅ Training with validation monitoring  
✅ Comprehensive evaluation metrics  
✅ Visualization of predictions  
✅ Model saving and loading  

## 🚨 Important Notes

- **Privacy:** Never share `kaggle.json` publicly
- **GPU:** Use GPU for faster training (available in Colab)
- **Dataset:** Respect ISIC dataset terms of use
- **Disclaimer:** This model is for educational purposes and should not be used for medical diagnosis without professional validation

## 📖 References

- [ISIC Challenge Website](https://www.isic-archive.com/)
- [Kaggle ISIC 2024 Competition](https://www.kaggle.com/competitions/isic-2024-challenge)
- [Deep Learning for Medical Image Analysis](https://arxiv.org/abs/1702.05747)
- [Transfer Learning in Deep Learning](https://cs231n.github.io/transfer-learning/)

## 👤 Author

**Ritupriya176**  
GitHub: [@Ritupriya176](https://github.com/Ritupriya176)

## 📄 License

This project is open-source and available under the MIT License.

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. The model and predictions should not be used for medical diagnosis or treatment decisions without consultation with qualified healthcare professionals. Always follow ethical guidelines when working with medical data.

---

**Last Updated:** May 15, 2026  
**Status:** In Development

For questions or contributions, please open an issue or pull request on GitHub.
