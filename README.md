## Team Machine Learning
</br>
<p align="center">
  <a href="https://www.tensorflow.org/" target="blank"><img src="https://www.tensorflow.org/images/tf_logo_social.png" width="240" alt="TensorFlow Logo" /></a>
  <a href="https://pytorch.org/" target="blank"><img src="https://pytorch.org/assets/images/logo.svg" width="240" alt="PyTorch Logo" /></a>
</p>

## Machine Learning Models Overview

This repository contains two main features:

1. **Time Series Prediction**:
   - Utilizes LSTM (Long Short-Term Memory) models for sequence prediction.
   - Built and trained using TensorFlow.

2. **Optical Character Recognition (OCR)**:
   - **Text Detection**: Uses EAST (Efficient and Accurate Scene Text) with MobileNet as the backbone.
   - **Text Recognition**: Implements CRNN (Convolutional Recurrent Neural Network) with MobileNet using the docTR library (PyTorch).

<p align="center">
  <a href="#" target="blank"><img src="https://res.cloudinary.com/dwm0tvqar/image/upload/v1734004301/project%20architecture.png" alt="Project Architecture" /></a>
</p>

## Project Setup

### Clone the Repository

```bash
$ git clone https://github.com/<your-repo-name>/machine-learning.git
$ cd machine-learning
```

### Install Dependencies

```bash
$ pip install -r requirements.txt
```

### Set Up Environment

Ensure the following environment variables are properly configured:

- `DATA_PATH`: Path to the dataset.
- `MODEL_SAVE_PATH`: Path to save trained models.
- `OCR_CONFIG`: Configuration file path for OCR models.

Create a `.env` file in the root directory:

```env
DATA_PATH=path/to/data
MODEL_SAVE_PATH=path/to/save/models
OCR_CONFIG=config/ocr_config.yaml
```

### Time Series Model Setup

Train the LSTM model:

```bash
$ python time_series/train.py
```

Run predictions:

```bash
$ python time_series/predict.py
```

### OCR Model Setup

#### Text Detection
Train the EAST MobileNet model:

```bash
$ python ocr/text_detection/train.py
```

#### Text Recognition
Fine-tune the CRNN MobileNet model with docTR:

```bash
$ python ocr/text_recognition/train.py
```

Run the OCR pipeline:

```bash
$ python ocr/run_ocr.py
```

## Compile and Run the Project

### Development

```bash
$ python main.py
```

### Production Mode

Deploy the models and application using a preferred platform, such as Google Cloud or AWS.

## Run Tests

### Unit Tests

```bash
$ pytest tests/unit
```

### Integration Tests

```bash
$ pytest tests/integration
```

### Test Coverage

```bash
$ pytest --cov=.
```
