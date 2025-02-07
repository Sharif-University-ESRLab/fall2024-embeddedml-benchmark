import time
import psutil
import os
import numpy as np
import tensorflow as tf

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

    start_time = time.time()

    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / (1024 ** 2)

    output = run_inference(interpreter, input_data)

    memory_after = process.memory_info().rss / (1024 ** 2)

    end_time = time.time()

    execution_time = end_time - start_time
    inference_delay = execution_time

    with open('benchmark_results.txt', 'w') as f:
        f.write(f'Execution Time: {execution_time} seconds\n')
        f.write(f'Memory Before Inference: {memory_before} MB\n')
        f.write(f'Memory After Inference: {memory_after} MB\n')
        f.write(f'Inference Delay: {inference_delay} seconds\n')
        f.write(f'Output: {output}\n')

if __name__ == '__main__':
    main()
