import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

def preprocess_data(image, label):
    image = tf.image.resize(image, (32, 32))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, tf.one_hot(label, depth=10)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

(ds_train, ds_test), ds_info = tfds.load("cifar10", split=["train", "test"], as_supervised=True, with_info=True)

BATCH_SIZE = 64
AUTOTUNE = tf.data.experimental.AUTOTUNE

ds_train = ds_train.map(preprocess_data, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.map(lambda x, y: (data_augmentation(x, training=True), y))
ds_train = ds_train.shuffle(5000).batch(BATCH_SIZE).prefetch(AUTOTUNE)

ds_test = ds_test.map(preprocess_data, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE).prefetch(AUTOTUNE)

base_model = tf.keras.applications.MobileNetV2(input_shape=(32, 32, 3), include_top=False, weights="imagenet")

for layer in base_model.layers[-30:]:
    layer.trainable = True

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation="softmax")
])

initial_learning_rate = 0.0005
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=1000, decay_rate=0.9)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=["accuracy"])

EPOCHS = 20
model.fit(ds_train, validation_data=ds_test, epochs=EPOCHS)

model.save("cifar10_mobilenetv2_finetuned.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

tflite_model_path = "cifar10_mobilenetv2_finetuned.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved at {tflite_model_path}")