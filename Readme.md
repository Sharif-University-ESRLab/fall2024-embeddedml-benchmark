![Logo](https://github.com/Sharif-University-ESRLab/fall2024-embeddedml-benchmark/blob/main/Logo.jpg)

# A Benchmarking Framework for Machine Learning Algorithms on Embedded Devices

This project measures how well an STM32 microcontroller can recognize spoken commands. It checks how fast the device processes information, how much memory it uses, and how much power it consumes. Finally, we offer a clear and simple benchmark that evaluates how different machine learning algorithms perform on an STM32 microcontroller, showing their processing speed, memory usage, and power consumption in a resource-limited environment.

## Tools

**Hardware:**

- **STM32 Development Board:** A STM32F103C8 Board
-  **ST-Link programmer** for flashing the firmware

**Software:**

- **Development Tools:**
  - **Keil uVision5** (ARM uVision 5, version 5.x)
- **Libraries:**
  - **TensorFlow Lite for Microcontrollers:** A lightweight version of TensorFlow designed for microcontroller environments.
- **Other Tools:**
  - **Python 3.x:** For scripting and model preparation.
  - **Git:** For version control.

## Implementation Details

**1. Model Preparation and Conversion**

- **Convert the Model to TensorFlow Lite Format:**
  - We use a script (`ModelConverter.py`) to convert our trained model into the TensorFlow Lite format (`.tflite`).

- **Validate the TensorFlow Lite Model:**
  - Use a validation script (e.g., `ValidateTFModel.py`) to ensure the converted model works correctly.

**2. Setting Up the STM32 Environment**

- **Install STM32CubeIDE:**
  - Download and install STM32CubeIDE from the [STMicroelectronics website](https://www.st.com/en/development-tools/stm32cubeide.html).

- **Install STM32Cube.AI:**
  - Within STM32CubeIDE, install the STM32Cube.AI plugin to enable model conversion and optimization features.

**3. Deploying the Model to STM32**

- **Convert the TensorFlow Lite Model to C Code:**
  - Open STM32CubeIDE and switch to the STM32Cube.AI perspective.
  - Select your `.tflite` model file.
  - STM32Cube.AI will generate optimized C code from the model.

- **Create a New STM32 Project:**
  - In STM32CubeIDE, create a new project for your specific STM32 board.
  - Add the generated C files from STM32Cube.AI to your project.
  - Configure necessary peripherals (e.g., ADC, I2S) for audio input.
  - Use the generated API to load the model and perform inference.

**4. Implementing the Benchmark Test**

- **Develop Inference Code:**
  - Write the main application code (e.g., in `runner.c`) to:
    - Capture audio data.
    - Preprocess the data as required by the model.
    - Run inference using the model.
    - Record performance metrics.

- **Measure Performance Metrics:**
  - **Inference Latency:** Measure the time taken to perform a single inference.
  - **Memory Usage:** Monitor RAM and Flash usage to ensure they are within the STM32's constraints.
  - **Power Consumption:** Use tools like ST-Link's power profiling feature or external hardware to assess power usage during inference.

**5. Running and Analyzing the Benchmark**

- **Flash the Firmware:**
  - Build the project in STM32CubeIDE to generate the firmware binary.
  - Connect the STM32 board and flash the firmware onto it.

- **Execute the Benchmark:**
  - Reset or power cycle the STM32 board to start running the benchmark.
  - Use a serial terminal (e.g., PuTTY, Tera Term) to view the printed inference times and predicted labels.

- **Analyze Data:**
  - **Inference Latency:** Calculate the average, minimum, and maximum inference times from the serial output.
  - **Memory Usage:** Check the compiled binary size and runtime memory usage.
  - **Power Consumption:** Review the power usage data collected during inference operations.

## Setting Up the Development Environment

#### Firmware Deployment with Keil uVision5

1. **Open the Keil Project:**
   - Launch Keil uVision5.
   - In Keil, navigate to **Project > Open Project...** and open the project file located in the `Code/STM32` folder (e.g., `RUN.uvprojx`).

2. **Configure the Target Device:**
   - Ensure the target device is set to an STM32F103 series microcontroller.
   - Verify that the project settings (clock configuration, memory size, etc.) match your STM32 board.

3. **Build the Project:**
   - Click on the **Build** button to compile the firmware.
   - Resolve any configuration issues that might arise during the build process.

4. **Flash the Firmware:**
   - Connect your STM32F103 board to your PC via the ST-Link programmer.
   - In Keil uVision5, use the **Download** option to flash the generated binary onto your board.
   - After flashing, reset or power cycle the board to start the application.

#### Simulation Environment for Python Scripts

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

![image](https://github.com/user-attachments/assets/4ce2b044-fa08-41f4-ad6d-e81e4e9fe802)

From the table, it can be understood that many developed TFLite models are well reached in the limited target of 64 KB memory, and only the LSTM structure exceeds this value because of its tokenizer.

### Emotion Detection

Regarding detecting whether a sentence in the domain of text has what type of emotion, we developed two different models: BERT and LSTM.

For the BERT model, we utilized the ParsBERT embedding space and defined a Dense-CNN classifier on top of it to be trained using our novel dataset. Below, the process of training and the value of accuracy and loss can be seen:

Accuracy over each iteration of training:

![image](https://github.com/user-attachments/assets/1c2ca0ca-fbb0-4278-8d85-df3baa9e6d42)

The value of loss for each iteration:

![image](https://github.com/user-attachments/assets/71506042-5d7a-4927-abd1-a2a34936bc99)

With the aid of the TFlite converter, we were able to reduce the size of the model to one-fourth of the initial size or, in other words, 162 MB, but this is beyond our limitation of the embedding system, so we developed an LSTM architecture in its place.

#### LSTM Model to Detect Emotion

For this task, we developed two distinct models, one regular one using a simple tokenizer by defining a dictionary of 2000 maximum vocab, and another using dynamic-learning rate and `sparse_categorical_crossentropy` loss. This model resulted in 82% and 89% accuracy, respectively, which is a promising result for 640 KB and 161 KB models.

Here, the confusion matrix of LSTM architecture can be seen:

![image](https://github.com/user-attachments/assets/fc1725a9-e641-4f85-9458-27b708a142d7)

And we have this result for enhanced architecture:

![image](https://github.com/user-attachments/assets/cc5c4c41-f421-43e2-b0c4-ca71192d8a28)

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

### Anomaly Detection

To understand anomaly behaviors, we developed and trained two distinct architectures, one using a random forest tree and another using FC-AutoEncoder. The random forest tree classifier showed a promising result of 99% in this task; however, its model size exceeds 16 MB, which is beyond the limit to be run on the STM32 chipset. The result of anomy detection is shown below:

![image](https://github.com/user-attachments/assets/933df4c2-7e85-4a94-9549-7eadf43da01d)

Using FC-AutoEncoder, we achieved acceptable results, which is shown here:

![image](https://github.com/user-attachments/assets/eb27c997-1546-4a13-927b-5e560b81868d)

Nevertheless, as shown in the training process, due to the limited size of the model, it is not capable of understanding the meaning and relation between labels and dense feature space.

![image](https://github.com/user-attachments/assets/f236673e-6714-40e1-9e2d-fcd264d21c24)

But its 41 KB size makes it manageable to work with in embedding systems.

## Dataset Gathering

One of our works' novel contributions is emotion detection dataset creation. For this dataset, we utilized GPT to generate sentences for each emotion, resulting in more than 6000 Farsi-labeled sentences across 12 different classes. For this reason, we used prompt engineering techniques to make sure that the generated sentences were valid and also unique.

## Simulation Process

For the simulation process, we utilized the TFLite model to get the result. We analyzed and monitored the memory usage, CPU time, execution time, and outputs, and we were able to produce results iteratively in each area of measurement. the result is as follows.

The CPU usage during iterations:

![image](https://github.com/user-attachments/assets/34d538e4-39c3-417e-9588-a9f8f99f3812)

The Memory Usage:

![image](https://github.com/user-attachments/assets/2ee29c3c-012d-407d-9034-5e36fcb75bc7)

The measured outputs of the TFLite model:

![image](https://github.com/user-attachments/assets/c2cd8bac-cdd7-45da-ac96-c5886438e55a)

And finally, the execution time over iterations:

![image](https://github.com/user-attachments/assets/237c0567-a871-4cdb-ad58-842ac8bb1eb1)

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
