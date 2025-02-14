# Code

This repository contains the codebase for the **EmbeddedML-Benchmark** project, which benchmarks various machine learning models across different tasks such as image classification, keyword spotting, emotion detection, and anomaly detection. The project is designed to assess the performance of machine learning models on embedded devices, focusing on resource efficiency, model size, inference speed, and power consumption. Below is an overview of the code structure and its components.

## Main Directory

- **main.py**: The entry point of the project, responsible for orchestrating the execution of various tasks and models, as well as collecting benchmarking results.
- **README.md**: Provides an overview of the project, including detailed instructions for users on setup and usage.
- **requirements.txt**: Lists all Python dependencies required to run the project.
- **Runner.c**: Contains C code for executing models on embedded devices, specifically for use with the STM32F103 microcontroller.

## Subdirectories

### **Anomaly Detection**

This module focuses on detecting anomalies within datasets using machine learning models such as autoencoders and random forests.

- **CoreDetection.ipynb**: A Jupyter notebook outlining the anomaly detection process, including data preprocessing, model training, and evaluation.
- **KDDCupDataset**: Contains the KDD Cup dataset files used for training and evaluation.
  - **kddcup.csv**: The main dataset in CSV format.
  - **kddcup.data_10_percent.gz**: A compressed version of a subset of the dataset.
- **Models**: Stores trained models for anomaly detection.
  - **autoencoder_model.tflite**: TensorFlow Lite model of the autoencoder used for anomaly detection.
  - **random_forest_model.tflite**: TensorFlow Lite model of the random forest classifier.

### **Emotion Detection**

This module is designed for detecting emotions from textual data using deep learning models such as LSTMs and BERT.

- **bert_tokenizer.pkl**: A serialized BERT tokenizer used for preprocessing text data.
- **CreateDataset.py**: A script for creating and preprocessing the emotion detection dataset.
- **EmotionClassifier.ipynb**: A Jupyter notebook detailing the process of training and evaluating the emotion classification model.
- **EmotionDataset**: Contains scripts and data related to different emotional states.
  - **angry.py, anxious.py, ashamed.py, etc.**: Scripts associated with specific emotions.
  - **emotions_dataset.csv**: Consolidated dataset containing text samples labeled with corresponding emotions.
- **Models**: Stores the trained models for emotion detection.
  - **lstm_emotion_model.tflite**: TensorFlow Lite model of the LSTM-based emotion classifier.
  - **lstm_emotion_model_enhanced.tflite**: An enhanced version of the LSTM model with improved accuracy.

### **Image Classification**

This module handles image classification tasks using various neural network architectures, such as MobileNetV2 and TinyML models.

- **MobilenetV2-Cifar10.ipynb**: A Jupyter notebook demonstrating the training and evaluation of a MobileNetV2 model on the CIFAR-10 dataset.
- **TinyML-Cifar10.py**: Python script for training a lightweight model optimized for deployment on embedded devices.
- **Models**: Contains trained models for image classification.
  - **cifar10_mobilenetv2.tflite**: TensorFlow Lite model of MobileNetV2 trained on CIFAR-10.
  - **cifar10_mobilenetv2_finetuned.tflite**: Fine-tuned version of the MobileNetV2 model.
  - **cifar10_tinyml.tflite**: A compact model optimized for TinyML applications.

### **Keyword Spotting**

This module is dedicated to training models for keyword spotting tasks using audio datasets like Google's Speech Commands.

- **.gitignore**: Specifies files and directories to be ignored by Git.
- **build_ref.sh**: A shell script for building reference models.
- **ConvertedModel**: Contains the converted TensorFlow Lite models.
  - **speech_commands_model.tflite**: Model trained for recognizing specific speech commands.
  - **speech_commands_model_float32.tflite**: Floating-point version of the speech commands model.
- **Dataset**: Includes scripts for dataset management.
  - **DatasetChecker.py**: Script to verify the integrity of the dataset.
  - **DownloadSpeechCommands.py**: Script to download the speech commands dataset.
- **evaluate.py**: Script for evaluating the performance of trained models.
- **eval_functions_eembc.py**: Contains evaluation functions that adhere to EEMBC standards.
- **eval_quantized_model.py**: Script for evaluating quantized models.
- **get_dataset.py**: Script to retrieve and preprocess datasets for keyword spotting tasks.
- **keras_model.py**: Defines the Keras model architecture for keyword spotting.
- **kws_util.py**: Utility functions for keyword spotting tasks.
- **make_all_bin_files.sh**: Shell script to generate binary files for deployment.
- **make_bin_files.py**: Python script to create binary files from models.
- **make_model_c_file**: Script to convert models into C source files for embedded deployment.
- **mk_cal_set.py**: Script to create calibration sets for quantization.
- **ModelConverter.py**: Script to convert trained models into TensorFlow Lite format.
- **quantize.py**: Script for model quantization to reduce model size and increase inference speed.

## Usage

### Install Dependencies
Ensure all Python dependencies are installed by running:

```bash
pip install -r requirements.txt
```

### Running the Benchmark
To run the benchmarking process:

```bash
python main.py
```

After execution, results will be saved in the `benchmark_results.txt` file.

### Running Individual Scripts
For specific tasks, you can execute individual scripts as follows:

#### Anomaly Detection

To run the anomaly detection process:
```bash
python CoreDetection.ipynb
```

#### Emotion Detection

To train the emotion detection model:
```bash
python EmotionClassifier.ipynb
```

#### Image Classification

To train the image classification model:
```bash
python MobilenetV2-Cifar10.ipynb
```

#### Keyword Spotting

To download the Speech Commands dataset:
```bash
python Code/DownloadSpeechCommands.py --data_dir=<path_to_data>
```

To train the keyword spotting model:
```bash
python Code/train.py --data_dir=<path_to_data> --epochs=<num_epochs> --saved_model_path=trained_models/kws_model.h5
```

To evaluate the trained keyword spotting model:
```bash
python Code/evaluate.py --model_init_path=trained_models/kws_model.h5 --target_set=<train|test|val>
```

### Convert and Quantize the Model
To convert and quantize the model to TensorFlow Lite format:
```bash
python Code/quantize.py --saved_model_path=trained_models/kws_model.h5 --tfl_file_name=trained_models/kws_model.tflite
```

## Acknowledgments

- TensorFlow Lite for providing the framework for deploying models on embedded devices.
- The open-source community for their contributions to machine learning and embedded systems.