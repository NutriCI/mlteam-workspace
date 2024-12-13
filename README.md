# Machine Learning Team
</br>
<p align="center">
  <a href="https://www.tensorflow.org/" target="blank"><img src="https://www.tensorflow.org/images/tf_logo_social.png" width="360" alt="TensorFlow Logo" /></a>
  <a href="https://pytorch.org/" target="blank"><img src="https://pytorch.org/assets/images/logo.svg" width="360" alt="PyTorch Logo" /></a>
</p>

## 🧑‍💻Meet the Engineers👩‍💻

- Hasta Suci Purbaningsih - M254B4KX1747 ([@hastasuci](https://github.com/hastasuci))
- Azhar Anas - M254B4KY0760 ([@DyoSch](https://github.com/DyoSch))
- Farras Rafi’ Permana - M010B4KY1421  ([@frrp16](https://github.com/frrp16))

## Machine Learning Models Overview

This repository contains two main features:

1. **Time Series Prediction**:
   - Utilizes LSTM (Long Short-Term Memory) models for sequence prediction.
   - Built and trained using TensorFlow.

2. **Optical Character Recognition (OCR)**:
   - **Text Detection**: Uses EAST (Efficient and Accurate Scene Text) with MobileNet as the backbone.
   - **Text Recognition**: Implements CRNN (Convolutional Recurrent Neural Network) with MobileNet using the docTR library (PyTorch).


## Project Setup

### Clone the Repository

```bash
$ git clone https://github.com/NutriCI/mlteam-workspace.git
```

### Install Dependencies
Install the latest [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.anaconda.com/miniconda/install/) first. Run command bellow only on Windows OS

```bash
$ cd mlteam-workspace/ocr_model/deployment
$ conda env create --file environment.yml
```

### Time Series Model Setup

Train the LSTM model:

```bash
$ cd mlteam-workspace/time_series_model/analysis
$ code .
```

Run predictions:

```bash
$ cd mlteam-workspace/time_series_model
$ python app.py
```

### OCR Model Setup

#### Table Detection (Opsional)
Train the SSD Resnet50 FPN model:

```bash
$ cd mlteam-workspace/ocr_model/table_detection
$ code .
```

#### Text Detection
Train the EAST MobileNet model:

```bash
$ cd mlteam-workspace/ocr_model/1_text_detection
$ code .
```

#### Text Recognition
Fine-tune the CRNN MobileNet model with docTR PyTorch:

```bash
$ cd mlteam-workspace/ocr_model/2_ocr
$ code .
```

Run the OCR pipeline:

```bash
$ cd mlteam-workspace/ocr_model/deployment
$ python main.py
```
*Note: Currently, in the OCR pipeline, we have removed the table detection step to reduce prediction time
