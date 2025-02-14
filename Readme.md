![Logo](https://github.com/Sharif-University-ESRLab/fall2024-embeddedml-benchmark/blob/main/Logo.jpg)

# A Benchmarking Framework for Machine Learning Algorithms on Embedded Devices

## Table of Contents
- [Overview](#overview)
- [Tools](#tools)
- [Implementation Details](#implementation-details)
  - [Deployment with Keil uVision5](#deployment-with-keil-uvision5)
  - [Simulation Environment for Python Scripts](#simulation-environment-for-python-scripts)
- [Results](#results)
  - [Emotion Detection](#emotion-detection)
  - [Anomaly Detection](#anomaly-detection)
- [Dataset Gathering](#dataset-gathering)
- [Simulation Process](#simulation-process)
- [Additional Notes](#additional-notes)
- [Related Links](#related-links)
- [Authors](#authors)

## Overview
This project benchmarks the performance of the STM32F103 microcontroller in running machine learning models for keyword spotting, image classification, anomaly detection, and emotion recognition. We convert models into optimized formats and analyze inference speed, memory usage, and power consumption to assess their feasibility on embedded systems. We used Keil uVision5 for firmware development and provided a Python-based simulation environment for validating models before deployment. This work helps assess the practical applications of machine learning on low-power embedded devices.

## Tools

### Hardware
- **STM32 Development Board:** A STM32F103C8 Board
-  **ST-Link programmer** for flashing the firmware

### Software
- **Development Tools:**
  - **Keil uVision5** (ARM uVision 5, version 5.x)
- **Libraries:**
  - **TensorFlow Lite for Microcontrollers:** A lightweight version of TensorFlow designed for microcontroller environments.
- **Other Tools:**
  - **Python 3.x:** For scripting and model preparation.
  - **Git:** For version control.

## Implementation Details

### Deployment with Keil uVision5

#### 1. Model Preparation and Conversion
- Convert trained model to TensorFlow Lite format using `ModelConverter.py`.
- Validate the TensorFlow Lite model with `ValidateTFModel.py`.
- Convert the `.tflite` model into a C header file using the command:
  ```bash
  xxd -i speech_commands_model_float32.tflite model_data.h
  ```
  This embeds the model data directly into firmware.

#### 2. Firmware Development in Keil uVision5
- Open the Keil project file (e.g., `RUN.uvprojx`) located in the `Code/STM32` folder.
- Configure the target device as an STM32F103 microcontroller, ensuring clock and memory settings match the board.
- In the main source file (e.g., `Runner.c`), implement code to:
  - Capture audio data.
  - Run inference using the embedded model from `model_data.h`.
  - Measure and print inference time and predicted labels to a serial terminal.

<div align="center">
  <img src="https://github.com/Sharif-University-ESRLab/fall2024-embeddedml-benchmark/blob/main/Code/STM32/Pics/Pic1.jpg" width="1000">
  <p><b>Keil uVision5 Board Menu</b></p>
</div>

<div align="center">
  <img src="https://github.com/Sharif-University-ESRLab/fall2024-embeddedml-benchmark/blob/main/Code/STM32/Pics/Pic2.jpg" width="1000">
  <p><b>Keil uVision5 Manage Run-Time Environment Menu</b></p>
</div>

<div align="center">
  <img src="https://github.com/Sharif-University-ESRLab/fall2024-embeddedml-benchmark/blob/main/Code/STM32/Pics/Pic3.jpg" width="1000">
  <p><b>Run-Time Environment Config</b></p>
</div>

<div align="center">
  <img src="https://github.com/Sharif-University-ESRLab/fall2024-embeddedml-benchmark/blob/main/Code/STM32/Pics/Pic4.jpg" width="1000">
  <p><b>Project Structure</b></p>
</div>

<div align="center">
  <img src="https://github.com/Sharif-University-ESRLab/fall2024-embeddedml-benchmark/blob/main/Code/STM32/Pics/Pic5.jpg" width="1000">
  <p><b>Cortex-M Target Driver Setup, Debug Menu</b></p>
</div>

<div align="center">
  <img src="https://github.com/Sharif-University-ESRLab/fall2024-embeddedml-benchmark/blob/main/Code/STM32/Pics/Pic6.jpg" width="1000">
  <p><b>Cortex-M Target Driver Setup, Flash Download Menu</b></p>
</div>

#### 3. Building and Flashing the Firmware
- **Build the Project:** In Keil uVision5, navigate to **Project > Open Project...** to load your project, then click the **Build** button to compile the firmware.
- **Flash the Firmware:** Connect the STM32F103 board via ST-Link, and use the **Download** option to flash the firmware. After flashing, reset or power cycle the board to start the benchmark application.

#### 4. Running the Benchmark
- Open a serial terminal (e.g., PuTTY, Tera Term) to monitor the output from the STM32F103 board.
- The device will print inference times and predicted labels, this allows analyzing performance metrics such as inference latency, memory usage, and power consumption.

### Simulation Environment for Python Scripts

1. **Install Python and Dependencies:**
   - Ensure that Python 3.x is installed on your system.
   - Open a command prompt in the repository’s root directory.
   - Install required Python packages using:
     ```bash
     pip install -r Code/requirements.txt
     ```

2. **Run Simulation Scripts:**
   - To validate the model conversion or simulate inference on your computer, navigate to the appropriate folder (e.g., `Code`).
   - Execute simulation scripts such as:
     ```bash
     python ValidateTFModel.py
     ```
   - These scripts will run the TensorFlow Lite model in a simulated environment and output performance metrics and inference results.

## Results

As we can interpret from the table below, in this work, we assess five different scenarios, targeting different tasks in ML, like speech recognition, anomaly detection, image classification, and lastly, emotion detection in the domain of natural language processing.

<div align="center">
  <img 
    style="width: 1000px;"
    src="https://github.com/user-attachments/assets/4ce2b044-fa08-41f4-ad6d-e81e4e9fe802">
</div>

From the table, it can be understood that many developed TFLite models are well reached in the limited target of 64 KB memory, and only the LSTM structure exceeds this value because of its tokenizer.

### Emotion Detection

Regarding detecting whether a sentence in the domain of text has what type of emotion, we developed two different models: BERT and LSTM.

For the BERT model, we utilized the ParsBERT embedding space and defined a Dense-CNN classifier on top of it to be trained using our novel dataset. Below, the process of training and the value of accuracy and loss can be seen:

<div align="center">
  <img src="https://github.com/user-attachments/assets/1c2ca0ca-fbb0-4278-8d85-df3baa9e6d42" width="1000">
  <p><b>Accuracy over each iteration of training</b></p>
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/71506042-5d7a-4927-abd1-a2a34936bc99" width="1000">
  <p><b>The value of loss for each iteration</b></p>
</div>

With the aid of the TFlite converter, we were able to reduce the size of the model to one-fourth of the initial size or, in other words, 162 MB, but this is beyond our limitation of the embedded system, so we developed an LSTM architecture in its place.

#### LSTM Model to Detect Emotion

For this task, we developed two distinct models, one regular one using a simple tokenizer by defining a dictionary of 2000 maximum vocab, and another using dynamic-learning rate and `sparse_categorical_crossentropy` loss. This model resulted in 82% and 89% accuracy, respectively, which is a promising result for 640 KB and 161 KB models.

<div align="center">
  <img src="https://github.com/user-attachments/assets/fc1725a9-e641-4f85-9458-27b708a142d7" width="1000">
  <p><b>Confusion Matrix of LSTM Architecture</b></p>
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/cc5c4c41-f421-43e2-b0c4-ca71192d8a28" width="1000">
  <p><b>Result for Enhanced Architecture</b></p>
</div>

Furthermore, more detailed information regarding classification is included below:

```txt
 precision    recall  f1-score   support

           0       0.88      0.76      0.82       110
           1       0.81      0.84      0.83       102
           2       0.95      0.86      0.90        97
           3       0.90      0.97      0.94       104
           4       0.97      0.96      0.96        95
           5       0.88      0.98      0.93       100
           6       0.99      0.94      0.96       113
           7       0.98      0.92      0.95       104
           8       0.80      0.73      0.76       101
           9       0.99      1.00      1.00       103
          10       0.91      0.92      0.92       100
          11       0.71      0.85      0.77       105

    accuracy                           0.89      1234
   macro avg       0.90      0.89      0.89      1234
weighted avg       0.90      0.89      0.89      1234
```

#### Dataset Gathering

One of our works' novel contributions is emotion detection dataset creation. For this dataset, we utilized GPT to generate sentences for each emotion, resulting in more than 6000 Farsi-labeled sentences across 12 different classes. For this reason, we used prompt engineering techniques to make sure that the generated sentences were valid and also unique.

### Anomaly Detection

To understand anomaly behaviors, we developed and trained two distinct architectures, one using a random forest tree and another using FC-AutoEncoder. The random forest tree classifier showed a promising result of 99% in this task; however, its model size exceeds 16 MB, which is beyond the limit to be run on the STM32 chipset. The result of anomaly detection is shown below:

<div align="center">
  <img src="https://github.com/user-attachments/assets/933df4c2-7e85-4a94-9549-7eadf43da01d" width="1000">
  <p><b>Result for Anomaly Detection</b></p>
</div>

Using FC-AutoEncoder, we achieved acceptable results, which is shown below:

<div align="center">
  <img src="https://github.com/user-attachments/assets/eb27c997-1546-4a13-927b-5e560b81868d" width="1000">
  <p><b>Result for Anomaly Detection Using FC-AutoEncoder</b></p>
</div>

Nevertheless, as shown in the training process, due to the limited size of the model, it is not capable of understanding the meaning and relation between labels and dense feature space.

<div align="center">
  <img 
    style="width: 1000px;"
    src="https://github.com/user-attachments/assets/f236673e-6714-40e1-9e2d-fcd264d21c24">
</div>

But its 41 KB size makes it manageable to work with in embedded systems.

## Simulation Process

For the simulation process, we utilized the TFLite model to get the result. We analyzed and monitored the memory usage, CPU time, execution time, and outputs, and we were able to produce results iteratively in each area of measurement. the result is as follows.

The CPU usage during iterations:

<div align="center">
  <img 
    style="width: 1000px;"
    src="https://github.com/user-attachments/assets/34d538e4-39c3-417e-9588-a9f8f99f3812">
</div>

The Memory Usage:

<div align="center">
  <img 
    style="width: 1000px;"
    src="https://github.com/user-attachments/assets/2ee29c3c-012d-407d-9034-5e36fcb75bc7">
</div>

The measured outputs of the TFLite model:

<div align="center">
  <img 
    style="width: 1000px;"
    src="https://github.com/user-attachments/assets/c2cd8bac-cdd7-45da-ac96-c5886438e55a">
</div>

And finally, Execution time across iterations:

<div align="center">
  <img 
    style="width: 1000px;"
    src="https://github.com/user-attachments/assets/237c0567-a871-4cdb-ad58-842ac8bb1eb1">
</div>

## Additional Notes

- **Firmware Development:**
  - In this project, we used a converted TFLite model (via `xxd`) embedded as a C header (`model_data.h`). No external AI libraries are imported in the firmware code.
  - Then in the project’s source files (`Runner.c`) we referenced `model_data.h`.

- **Simulation:**
  - The Python simulation helps in validating the TFLite model performance before deploying it on hardware.
  - To analyze memory usage, CPU time, and inference latency, we used a python script for each project.

## Related Links

- [STM32CubeIDE](https://www.st.com/en/development-tools/stm32cubeide.html)
- [STM32Cube.AI](https://www.st.com/en/development-tools/stm32cubeai.html)
- [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)
- [MLPerf Tiny Benchmark](https://arxiv.org/abs/2106.07597)

## Authors
- [Iman Mohammadi](https://github.com/Imanm02)
- [Shayan Salehi](https://github.com/ShayanSalehi81)
- [Armin Saghafian](https://github.com/ArminS03)

Special Thanks to [Ali Salesi](https://github.com/alisalc).
