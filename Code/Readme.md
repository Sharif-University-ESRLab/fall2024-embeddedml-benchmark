# Code

This repository contains the codebase for the EmbeddedML-Benchmark project, which focuses on benchmarking various machine learning models across different tasks. The project is organized into several subdirectories, each corresponding to a specific task or model. Below is a detailed overview of each component.

## Main Directory

- **main.py**: Serves as the entry point for the project, orchestrating the execution of various tasks and models.
- **Readme.md**: Provides an overview of the project and instructions for users.
- **requirements.txt**: Lists the Python dependencies required to run the project.
- **Runner.c**: Contains C code for executing models on embedded devices.

## Subdirectories

### AnomalyDetection

This module focuses on detecting anomalies within datasets.

- **CoreDetection.ipynb**: A Jupyter notebook detailing the anomaly detection process, including data preprocessing, model training, and evaluation.
- **KDDCupDataset**: Contains the KDD Cup dataset files used for training and evaluation.
  - **kddcup.csv**: The main dataset in CSV format.
  - **kddcup.data_10_percent.gz**: A compressed version of a subset of the dataset.
- **Models**: Stores the trained models for anomaly detection.
  - **autoencoder_model.tflite**: TensorFlow Lite model of the autoencoder used for anomaly detection.
  - **random_forest_model.tflite**: TensorFlow Lite model of the random forest classifier.

### EmotionDetection

This module is designed to detect emotions from textual data.

- **bert_tokenizer.pkl**: Serialized BERT tokenizer for preprocessing text data.
- **CreateDataset.py**: Script for creating and preprocessing the emotion detection dataset.
- **EmotionClassifier.ipynb**: Jupyter notebook detailing the process of training and evaluating the emotion classification model.
- **EmotionDataset**: Contains scripts and data related to different emotional states.
  - **angry.py, anxious.py, ashamed.py, etc.**: Scripts associated with specific emotions.
  - **emotions_dataset.csv**: Consolidated dataset containing text samples labeled with corresponding emotions.
- **Models**: Stores the trained models for emotion detection.
  - **lstm_emotion_model.tflite**: TensorFlow Lite model of the LSTM-based emotion classifier.
  - **lstm_emotion_model_enhanced.tflite**: An enhanced version of the LSTM model with improved accuracy.

### ImageClassification

This module handles image classification tasks using various neural network architectures.

- **MobilenetV2-Cifar10.ipynb**: Jupyter notebook demonstrating the training and evaluation of a MobileNetV2 model on the CIFAR-10 dataset.
- **TinyML-Cifar10.py**: Python script for training a lightweight model suitable for deployment on embedded devices.
- **Models**: Contains the trained models for image classification.
  - **cifar10_mobilenetv2.tflite**: TensorFlow Lite model of MobileNetV2 trained on CIFAR-10.
  - **cifar10_mobilenetv2_finetuned.tflite**: Fine-tuned version of the MobileNetV2 model.
  - **cifar10_tinyml.tflite**: A compact model optimized for TinyML applications.

### KeywordSpottingTrainer

This module is dedicated to training models for keyword spotting tasks.

- **.gitignore**: Specifies files and directories to be ignored by Git.
- **build_ref.sh**: Shell script for building reference models.
- **ConvertedModel**: Contains the converted TensorFlow Lite models.
  - **speech_commands_model.tflite**: Model trained for recognizing specific speech commands.
  - **speech_commands_model_float32.tflite**: Floating-point version of the speech commands model.
- **Dataset**: Includes scripts for dataset management.
  - **DatasetChecker.py**: Script to verify the integrity of the dataset.
  - **DownloadSpeechCommands.py**: Script to download the speech commands dataset.
- **evaluate.py**: Script for evaluating the performance of trained models.
- **eval_functions_eembc.py**: Contains evaluation functions adhering to EEMBC standards.
- **eval_quantized_model.py**: Script for evaluating quantized models.
- **get_dataset.py**: Script to retrieve and preprocess datasets.
- **keras_model.py**: Defines the Keras model architecture for keyword spotting.
- **kws_util.py**: Utility functions for keyword spotting tasks.
- **make_all_bin_files.sh**: Shell script to generate binary files for deployment.
- **make_bin_files.py**: Python script to create binary files from models.
- **make_model_c_file**: Script to convert models into C source files for embedded deployment.
- **mk_cal_set.py**: Script to create calibration sets for quantization.
- **ModelConverter.py**: Script to convert trained models into TensorFlow Lite format.
- **quantize.py**: Script for model quantization to reduce model size and increase inference speed.