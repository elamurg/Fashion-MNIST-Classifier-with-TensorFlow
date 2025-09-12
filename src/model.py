from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_shape=(28,28,1), num_classes=10):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32,3, activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64,3, activation = "relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation ="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
