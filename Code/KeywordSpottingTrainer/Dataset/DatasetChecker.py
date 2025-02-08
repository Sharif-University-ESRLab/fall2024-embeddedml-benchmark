import tensorflow_datasets as tfds

builder = tfds.builder('speech_commands')
print(builder.info)