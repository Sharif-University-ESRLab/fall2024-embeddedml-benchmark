import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path='Benchmark/ConvertedModel/speech_commands_model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

dummy_input = np.zeros(input_details[0]['shape'], dtype=np.int8)
interpreter.set_tensor(input_details[0]['index'], dummy_input)

interpreter.invoke()

output = interpreter.get_tensor(output_details[0]['index'])
print(output)