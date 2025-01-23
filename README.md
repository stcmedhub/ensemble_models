
# Deep Learning Image Classification with InceptionV3 and EfficientNetV2B2 for Ensemble generation

This repository contains code for training deep learning models using the InceptionV3 or EfficientNetV2B2 architecture to classify images. The models are trained with image augmentation techniques and evaluated using several metrics such as accuracy, AUC, precision, recall, and others.

## Overview

The project uses TensorFlow 2.x (including Keras) along with several image preprocessing and augmentation techniques. It includes data generators for training and validation, model architecture for InceptionV3 and EfficientNetV2B2, and callbacks for model checkpoints and logging.

## Features

- **InceptionV3 Model**: Used as the base model for image classification.
- **EfficientNetV2B2 Model**: Used as the base model for image classification.
- **Data Augmentation**: Advanced augmentations using the `albumentations` library.
- **Image Preprocessing**: Custom preprocessing for input images.
- **Metrics**: Evaluation includes binary accuracy, precision, recall, AUC, and confusion matrix metrics (True Positives, True Negatives, etc.).
- **Callbacks**: Model checkpoints and CSV logging during training.

## Setup

### Prerequisites

Make sure you have `Python` (preferably Python 3.8) and `pip` installed. Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Installing Dependencies

Install all the required packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### File Structure

```
.
├── README.md             # This file
├── requirements.txt       # Python dependencies
├── train_effinet.ipynb    # Model definition and training script
├── train_inception.ipynb    # Model definition and training script
```

### Data

Make sure you have the following directories for training and validation datasets:

```
/trainset/               # Directory for training images
/valset/                 # Directory for validation images
```

The directory structure should be organized as:

```
/trainset/
    /class1/
    /class0/

/valset/
    /class1/
    /class0/
```

### Training

To train the model, simply run the notebooks:

This will initiate training with the specified parameters, and it will save the model weights and architecture to files. You can adjust parameters like `batch_size`, `epochs`, and other configurations as needed.

### Model Saving

During and after training, the model weights and architecture are saved to the following files:

- `model_name.h5`: Model weights.
- `model_name.json`: Model architecture in JSON format.

### Callbacks

- **Model Checkpoints**: Model weights are saved at the end of each epoch.
- **CSV Logger**: Training history is logged in a CSV file for later analysis.

## Model Architecture

The architecture is based on **InceptionV3** or **EfficientNetV2B2** with the following modifications:

1. **Base Model**: InceptionV3 or EfficientNetV2B2.
2. **Custom Top Layers**: 
   - Global Average Pooling.
   - Dropout and Dense layers for regularization and classification.

## Image Augmentation

This project utilizes the `albumentations` library for image augmentation. Some of the augmentations used include:

- **Random Rotation**
- **Hue and Saturation Shifts**
- **Gaussian Noise**
- **Random Fog, Rain, Snow, and Spatter**
- **Coarse Dropout and Pixel Dropout**

These augmentations help improve the robustness and generalization of the model.

## Evaluation Metrics

The model is evaluated using the following metrics:

- **Binary Accuracy**
- **AUC (Area Under Curve)**
- **Precision**
- **Recall**
- **Confusion Matrix**: True Positives, True Negatives, False Positives, False Negatives.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
