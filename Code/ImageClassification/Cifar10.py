import tensorflow as tf
import tensorflow_datasets as tfds

(ds_train, ds_test), ds_info = tfds.load("cifar10", split=["train", "test"], as_supervised=True, with_info=True)

def preprocess_data(image, label):
    image = tf.image.resize(image, (16, 16))
    image = tf.cast(image, tf.float32) / 255.0
    return image, tf.one_hot(label, depth=10)

ds_train = ds_train.map(preprocess_data).batch(64).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(preprocess_data).batch(64).prefetch(tf.data.AUTOTUNE)

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(16, 16, 3)),
    tf.keras.layers.Conv2D(8, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(ds_train, validation_data=ds_test, epochs=10)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_data_gen():
    for image, _ in ds_train.take(100):
        image = tf.cast(image, tf.float32)
        yield [image]

converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
with open("cifar10_tinyml.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved at cifar10_tinyml.tflite")
