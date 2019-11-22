import tensorflow as tf
import numpy as np
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt


# Hyper parameters
epochs = 20
batch_size = 128


fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
print(X_train_full.shape, y_train_full)

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

height, width = X_train[0].shape
channel = 1

# fix shape	for	CNN
X_train = X_train.reshape(-1, height, width, channel)
X_valid = X_valid.reshape(-1, height, width, channel)
X_test = X_test.reshape(-1, height, width, channel)

print(X_train.shape)

# 10 class names
class_names = [
    "t-shirt",
    "trousers",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "boot"
]

# ------------------------------------ LeNet-5 Model ---------------------------------------

model = keras.models.Sequential([
    # C1 Convolutional Layer
    keras.layers.Conv2D(6, 5, activation="relu", padding="same", input_shape=(28, 28, 1)),

    # S2 Pooling Layer
    keras.layers.MaxPooling2D(2),

    # C3 Convolutional Layer
    keras.layers.Conv2D(16, 3, activation="relu", padding="same"),

    # S4 Pooling Layer
    keras.layers.MaxPooling2D(2),

    # C5 Fully Connected Convolutional Layer
    keras.layers.Conv2D(120, 3, activation="relu", padding="same"),

    # Flatten the CNN output so that we can connect it with fully connected layers
    keras.layers.Flatten(),

    # FC6 Fully Connected Layer
    keras.layers.Dense(84, activation="relu"),
    keras.layers.Dropout(0.05),

    # Output Layer with softmax activation
    keras.layers.Dense(10, activation="softmax")
])

print(model.summary())
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

# history object that contains	all	information	collected during training
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, y_valid))
model.evaluate(X_test, y_test)

weights = model.get_weights()

# Plot the data in history objects
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.gca().set_xlim(0, epochs-1)
plt.savefig('lenet5i')
