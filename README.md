
# Deep Learning Image Classification with InceptionV3

This repository contains code for training a deep learning model using the InceptionV3 architecture to classify images. The model is trained with advanced image augmentation techniques and evaluated using several metrics such as accuracy, AUC, precision, recall, and others.

## Overview

The project uses TensorFlow 2.x (including Keras) along with several image preprocessing and augmentation techniques. It includes data generators for training and validation, model architecture for InceptionV3, and callbacks for model checkpoints and logging.

## Features

- **InceptionV3 Model**: Used as the base model for image classification.
- **Data Augmentation**: Advanced augmentations using the `albumentations` library.
- **Image Preprocessing**: Custom preprocessing for input images.
- **Metrics**: Evaluation includes binary accuracy, precision, recall, AUC, and confusion matrix metrics (True Positives, True Negatives, etc.).
- **Callbacks**: Model checkpoints and CSV logging during training.

## Setup

### Prerequisites

Make sure you have `Python` (preferably Python 3.8 or higher) and `pip` installed. Create a virtual environment and install dependencies:

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
├── model.py               # Model definition and training script
├── data_preprocessing.py  # Image preprocessing and augmentation
└── train.py               # Training script
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
    /positive/
    /negative/

/valset/
    /positive/
    /negative/
```

### Training

To train the model, simply run the `train.py` script:

```bash
python train.py
```

This will initiate training with the specified parameters, and it will save the model weights and architecture to files. You can adjust the `batch_size`, `epochs`, and other configurations as needed.

### Model Saving

During and after training, the model weights and architecture are saved to the following files:

- `model_name.h5`: Model weights.
- `model_name.json`: Model architecture in JSON format.

### Callbacks

- **Model Checkpoints**: Model weights are saved at the end of each epoch.
- **CSV Logger**: Training history is logged in a CSV file for later analysis.

## Model Architecture

The architecture is based on **InceptionV3** with the following modifications:

1. **Base Model**: InceptionV3 without the top classification layers (for transfer learning).
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
