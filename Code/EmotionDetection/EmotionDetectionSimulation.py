import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import json
import time
import psutil
import os
import matplotlib.pyplot as plt


def run_inference(interpreter, input_data, attention_mask):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # print(f"Expected input shape: {input_details[0]['shape']}")
    # print(f"Input data shape: {input_data.shape}")

    if len(input_data.shape) == 3:
        input_data = np.squeeze(input_data, axis=-1)  # Remove extra dimension if necessary

    # print(f"input_details[0]['index']: {input_details[0]['index']}")
    # print(f"input_data: {input_data}")
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    if len(input_details) > 1:
        interpreter.set_tensor(input_details[1]['index'], attention_mask)

    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    return output, output_details


def run_simulation(model_path, X_test_input, X_test_attention_mask, num_iterations=100):
    interpreter = load_tflite_model(model_path)

    execution_times = []
    memory_usages = []
    cpu_usages = []
    all_outputs = []

    for i in range(num_iterations):
        print(f"Iteration: {i}")
        sample_input = X_test_input[i:i+1]  # Take one sample at a time
        attention_mask = X_test_attention_mask[i:i+1]

        if len(sample_input.shape) == 1:
            sample_input = np.expand_dims(sample_input, axis=0)  # Shape becomes [1, 28]

        if len(sample_input.shape) == 3:
            sample_input = np.squeeze(sample_input, axis=-1)  # Remove the last dimension if it exists

        # Track memory usage and CPU usage before inference
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / (1024 ** 2)
        cpu_before = psutil.cpu_percent(interval=0.1)

        start_time = time.time()
        output, output_details = run_inference(interpreter, sample_input, attention_mask)
        end_time = time.time()

        # Track memory usage and CPU usage after inference
        memory_after = process.memory_info().rss / (1024 ** 2)
        cpu_after = psutil.cpu_percent(interval=0.1)

        execution_times.append(end_time - start_time)
        memory_usages.append(memory_after - memory_before)
        cpu_usages.append(cpu_after - cpu_before)

        # Append output from the model
        all_outputs.append(output)

    # Calculate average values for output, execution time, memory usage, and CPU usage
    avg_execution_time = np.mean(execution_times)
    avg_memory_usage = np.mean(memory_usages)
    avg_cpu_usage = np.mean(cpu_usages)

    # Convert the list of outputs into a mean value for each output across iterations
    output = np.array(all_outputs).mean(axis=0)
    output_scale, output_zero_point = output_details[0]['quantization']
    dequantized_output = output_scale * (output - output_zero_point)

    # Prepare results dictionary
    results = {
        'execution_times': execution_times,
        'memory_usages': memory_usages,
        'cpu_usages': cpu_usages,
        'outputs': all_outputs,  # Store all output values
        'avg_execution_time': avg_execution_time,
        'avg_memory_usage': avg_memory_usage,
        'avg_cpu_usage': avg_cpu_usage,
        'quantized_output': output.tolist(),
        'dequantized_output': dequantized_output.tolist()
    }

    return results, output, dequantized_output

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def visualize_results(results):
    # Create directory to save figures if not exists
    if not os.path.exists('Results'):
        os.makedirs('Results')

    # 1. Execution Time Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(results['execution_times'], label='Execution Time', color='blue')
    plt.ylabel('Time (seconds)')
    plt.title('Execution Time Over Iterations')
    plt.xlabel('Iteration')
    plt.legend()
    plt.savefig('Results/execution_time_iterative.png')
    plt.show()

    # 2. Memory Usage Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(results['memory_usages'], label='Memory Usage', color='green')
    plt.ylabel('Memory (MB)')
    plt.title('Memory Usage Over Iterations')
    plt.xlabel('Iteration')
    plt.legend()
    plt.savefig('Results/memory_usage_iterative.png')
    plt.show()

    # 3. CPU Usage Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(results['cpu_usages'], label='CPU Usage', color='red')
    plt.ylabel('CPU Usage (%)')
    plt.title('CPU Usage Over Iterations')
    plt.xlabel('Iteration')
    plt.legend()
    plt.savefig('Results/cpu_usage_iterative.png')
    plt.show()

    # 4. Output Values (first 10 values from each iteration)
    plt.figure(figsize=(10, 6))
    for i, output in enumerate(results['outputs']):
        plt.plot(output[0][:10], label=f'Iteration {i+1}')
    plt.ylabel('Output Value')
    plt.title('Output Values for Each Iteration (First 10)')
    plt.xlabel('Output Index')
    plt.legend()
    plt.savefig('Results/outputs_iterative.png')
    plt.show()


# def save_benchmark_results(results):
#     with open('Results/benchmark_results.json', 'w') as f:
#         json.dump(results, f, indent=4)


def main():
    df = pd.read_csv("EmotionDataset/emotions_dataset.csv")
    
    MODEL_NAME = "HooshvareLab/bert-fa-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    sentences = df['sentence'].tolist()
    labels = pd.factorize(df['emotion'])[0]

    X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.2, random_state=42, stratify=labels)

    X_test_bert_tokens = tokenizer(X_test, padding="max_length", truncation=True, max_length=28, return_tensors="tf")
    X_test_input = X_test_bert_tokens['input_ids']

    X_test_attention_mask = X_test_bert_tokens['attention_mask']

    print("Shape of X_test_input:", X_test_input.shape)
    print("Shape of X_test_attention_mask:", X_test_attention_mask.shape)

    model_path = 'Models/parsbert_emotion_model.tflite'
    results, output, dequantized_output = run_simulation(model_path, X_test_input, X_test_attention_mask)

    # save_benchmark_results(results)

    # Print results
    print(f"Average Execution Time: {results['avg_execution_time']} seconds")
    print(f"Average Memory Usage: {results['avg_memory_usage']} MB")
    print(f"Average CPU Usage: {results['avg_cpu_usage']}%")
    print(f"Quantized Output (sample): {output[:10]}")
    print(f"Dequantized Output (sample): {dequantized_output[:10]}")

    visualize_results(results)


if __name__ == '__main__':
    main()