![Logo](https://github.com/Sharif-University-ESRLab/fall2024-embeddedml-benchmark/blob/main/Logo.jpg)

# A Benchmarking Framework for Machine Learning Algorithms on Embedded Devices

This project measures how well an STM32 microcontroller can recognize spoken commands. It checks how fast the device processes information, how much memory it uses, and how much power it consumes.

## Tools

**Hardware:**

- **STM32 Development Board:** For example, the STM32H7 series, which offers higher performance.
- **Audio Input:** A microphone or pre-recorded audio samples for testing.

**Software:**

- **Development Tools:**
  - **STM32CubeIDE:** An integrated development environment for STM32 microcontrollers.
  - **STM32Cube.AI:** A tool to convert and optimize neural network models for STM32 devices.
- **Libraries:**
  - **TensorFlow Lite for Microcontrollers:** A lightweight version of TensorFlow designed for microcontroller environments.
- **Other Tools:**
  - **Python 3.x:** For scripting and model preparation.
  - **Git:** For version control.

## Implementation Details

**1. Model Preparation and Conversion**

- **Convert the Model to TensorFlow Lite Format:**
  - Use a script (e.g., `ModelConverter.py`) to convert your trained model into the TensorFlow Lite format (`.tflite`).

- **Validate the TensorFlow Lite Model:**
  - Use a validation script (e.g., `ValidatTFModel.py`) to ensure the converted model works correctly.

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
  - Connect your STM32 board and flash the firmware onto it.

- **Execute the Benchmark:**
  - Reset or power cycle the STM32 board to start running the benchmark.
  - Use a serial terminal (e.g., PuTTY, Tera Term) to view the printed inference times and predicted labels.

- **Analyze Data:**
  - **Inference Latency:** Calculate the average, minimum, and maximum inference times from the serial output.
  - **Memory Usage:** Check the compiled binary size and runtime memory usage.
  - **Power Consumption:** Review the power usage data collected during inference operations.

**Example Serial Output:**

```
Inference Time: 15 ms, Predicted Label: 3
Inference Time: 14 ms, Predicted Label: 7
Inference Time: 16 ms, Predicted Label: 2
...
```

## How to Run

**Build the Project:**

In STM32CubeIDE, build your project to generate the firmware binary.

**Flash the Firmware:**

Connect your STM32 board to your computer.

Use STM32CubeIDE to flash the generated firmware onto the board.

**Run the Benchmark:**

After flashing, reset or power cycle the STM32 board.

Open a serial terminal (e.g., PuTTY, Tera Term) and connect to the appropriate COM port to view the output.

## Results

After running the benchmark, you should observe output similar to the following in your serial terminal:

```
Inference Time: 15 ms, Predicted Label: 3
Inference Time: 14 ms, Predicted Label: 7
Inference Time: 16 ms, Predicted Label: 2
...
```

Analyze this data to assess the performance of your model on the STM32 microcontroller.

## Related Links

- [STM32CubeIDE](https://www.st.com/en/development-tools/stm32cubeide.html)
- [STM32Cube.AI](https://www.st.com/en/development-tools/stm32cubeai.html)
- [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)

## Authors
- [Iman Mohammadi](https://github.com/Imanm02)
- [Shayan Salehi](https://github.com/ShayanSalehi81)
- [Armin Saghafian](https://github.com/ArminS03)