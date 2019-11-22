from tensorflow import keras

model = keras.models.Sequential([

    # Block 1
    keras.layers.Conv2D(64, 3, activation="relu", padding="same", name='block1_conv1', input_shape=(224, 224, 3)),
    keras.layers.Conv2D(64, 3, activation="relu", padding="same", name='block1_conv2'),
    keras.layers.MaxPooling2D(strides=(2, 2)),

    # Block 2
    keras.layers.Conv2D(128, 3, activation="relu", padding="same", name='block2_conv1'),
    keras.layers.Conv2D(128, 3, activation="relu", padding="same", name='block2_conv2'),
    keras.layers.MaxPooling2D(2),

    # Block 3
    keras.layers.Conv2D(256, 3, activation="relu", padding="same", name='block3_conv1'),
    keras.layers.Conv2D(256, 3, activation="relu", padding="same", name='block3_conv2'),
    keras.layers.Conv2D(256, 3, activation="relu", padding="same", name='block3_conv3'),
    keras.layers.Conv2D(256, 3, activation="relu", padding="same", name='block3_conv4'),
    keras.layers.MaxPooling2D(2),

    # Block 4
    keras.layers.Conv2D(512, 3, activation="relu", padding="same", name='block4_conv1'),
    keras.layers.Conv2D(512, 3, activation="relu", padding="same", name='block4_conv2'),
    keras.layers.Conv2D(512, 3, activation="relu", padding="same", name='block4_conv3'),
    keras.layers.Conv2D(512, 3, activation="relu", padding="same", name='block4_conv4'),
    keras.layers.MaxPooling2D(2),

    # Block 5
    keras.layers.Conv2D(512, 3, activation="relu", padding="same", name='block5_conv1'),
    keras.layers.Conv2D(512, 3, activation="relu", padding="same", name='block5_conv2'),
    keras.layers.Conv2D(512, 3, activation="relu", padding="same", name='block5_conv3'),
    keras.layers.Conv2D(512, 3, activation="relu", padding="same", name='block5_conv4'),
    keras.layers.MaxPooling2D(2),

    # Flatten the CNN output so that we can connect
    # it with fully connected layers
    keras.layers.Flatten(),

    # FC6 Fully Connected Layer
    keras.layers.Dense(4096, activation="relu", name='fc6'),

    # FC7 Fully Connected Layer
    keras.layers.Dense(4096, activation="relu", name='fc7'),

    # Output Layer with softmax activation
    keras.layers.Dense(1000, activation="softmax", name='output')
])

print(model.summary())

# Link : https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/
feature_map_num = ((2*64) + (2*128) + (4*256) + (8*512)) + 4096 + 4096

print("\n\n Number of feature maps : ", feature_map_num)