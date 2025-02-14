import tensorflow as tf

saved_model_dir = 'Benchmark/KeywordSpottingTrainer/trained_models/kws_ref_model'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int16  # Assuming int16 input
converter.inference_output_type = tf.int8  # For quantized models

tflite_model = converter.convert()

with open('Benchmark/ConvertedModel/speech_commands_model.tflite', 'wb') as f:
    f.write(tflite_model)
