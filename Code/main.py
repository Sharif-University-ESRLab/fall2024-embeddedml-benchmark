import time
import psutil
import os
import numpy as np
import tensorflow as tf
import json

def run_inference(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

def main():
    model_path = 'D:/Reposetories/fall2024-embeddedml-benchmark/Code/ConvertedModel/speech_commands_model.tflite'
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_data = np.random.rand(49, 10).astype(np.int8)  # Original 3D input: [height, width, channels]
    input_data = np.expand_dims(input_data, axis=0)    # Add batch dimension: [1, height, width, channels]

    # Measure execution time and memory usage
    num_iterations = 10
    execution_times = []
    memory_usages = []
    cpu_usages = []

    for _ in range(num_iterations):
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / (1024 ** 2)
        cpu_before = psutil.cpu_percent(interval=0.1)

        start_time = time.time()
        output = run_inference(interpreter, input_data)
        end_time = time.time()

        memory_after = process.memory_info().rss / (1024 ** 2)
        cpu_after = psutil.cpu_percent(interval=0.1)

        execution_times.append(end_time - start_time)
        memory_usages.append(memory_after - memory_before)
        cpu_usages.append(cpu_after - cpu_before)

    avg_execution_time = np.mean(execution_times)
    avg_memory_usage = np.mean(memory_usages)
    avg_cpu_usage = np.mean(cpu_usages)

    # Save results
    results = {
        'avg_execution_time': avg_execution_time,
        'avg_memory_usage': avg_memory_usage,
        'avg_cpu_usage': avg_cpu_usage,
        'output': output.tolist()
    }

    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Average Execution Time: {avg_execution_time} seconds")
    print(f"Average Memory Usage: {avg_memory_usage} MB")
    print(f"Average CPU Usage: {avg_cpu_usage}%")
    print(f"Output: {output}")

if __name__ == '__main__':
    main()