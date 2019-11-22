from tensorflow import keras
feature_map_num = 0

model = keras.models.Sequential([

    # Block 1 : {7x7 kernel,	stride	2,	64	output	maps}
    keras.layers.Conv2D(64, 7, strides=(2, 2), activation="relu", padding="same", name='block1_conv1',
                        input_shape=(224, 224, 3)),

    # MAX Pooling
    keras.layers.MaxPooling2D(strides=(2, 2)),

    # Block 2 : 6x	{3x3 kernel,	64	output	maps}
    keras.layers.Conv2D(64, 3, activation="relu", padding="same", name='block2_conv1'),
    keras.layers.Conv2D(64, 3, activation="relu", padding="same", name='block2_conv2'),
    keras.layers.Conv2D(64, 3, activation="relu", padding="same", name='block2_conv3'),
    keras.layers.Conv2D(64, 3, activation="relu", padding="same", name='block2_conv4'),
    keras.layers.Conv2D(64, 3, activation="relu", padding="same", name='block2_conv5'),
    keras.layers.Conv2D(64, 3, activation="relu", padding="same", name='block2_conv6'),

    # Block 3 : {3x3	kernel,	stride	2,	128	output	maps}
    keras.layers.Conv2D(128, 3, strides=(2, 2), activation="relu", padding="same", name='block3_conv1'),

    # Block 4 : 7x	{3x3 kernel,	128	output	maps}
    keras.layers.Conv2D(128, 3, activation="relu", padding="same", name='block4_conv1'),
    keras.layers.Conv2D(128, 3, activation="relu", padding="same", name='block4_conv2'),
    keras.layers.Conv2D(128, 3, activation="relu", padding="same", name='block4_conv3'),
    keras.layers.Conv2D(128, 3, activation="relu", padding="same", name='block4_conv4'),
    keras.layers.Conv2D(128, 3, activation="relu", padding="same", name='block4_conv5'),
    keras.layers.Conv2D(128, 3, activation="relu", padding="same", name='block4_conv6'),
    keras.layers.Conv2D(128, 3, activation="relu", padding="same", name='block4_conv7'),

    # Block 5 : {3x3 kernel,	stride	2,	256	output	maps}
    keras.layers.Conv2D(256, 3, strides=(2, 2), activation="relu", padding="same", name='block5_conv1'),

    # block 6 : 11x	{3x3 kernel,	256	output	maps}
    keras.layers.Conv2D(256, 3, activation="relu", padding="same", name='block6_conv1'),
    keras.layers.Conv2D(256, 3, activation="relu", padding="same", name='block6_conv2'),
    keras.layers.Conv2D(256, 3, activation="relu", padding="same", name='block6_conv3'),
    keras.layers.Conv2D(256, 3, activation="relu", padding="same", name='block6_conv4'),
    keras.layers.Conv2D(256, 3, activation="relu", padding="same", name='block6_conv5'),
    keras.layers.Conv2D(256, 3, activation="relu", padding="same", name='block6_conv6'),
    keras.layers.Conv2D(256, 3, activation="relu", padding="same", name='block6_conv7'),
    keras.layers.Conv2D(256, 3, activation="relu", padding="same", name='block6_conv8'),
    keras.layers.Conv2D(256, 3, activation="relu", padding="same", name='block6_conv9'),
    keras.layers.Conv2D(256, 3, activation="relu", padding="same", name='block6_conv10'),
    keras.layers.Conv2D(256, 3, activation="relu", padding="same", name='block6_conv11'),

    # Block 7 : {3x3 kernel, stride	2,	512	output	maps}
    keras.layers.Conv2D(512, 3, strides=(2, 2), activation="relu", padding="same", name='block7_conv1'),

    # Block 8 : 5x	{3x3 kernel,	512	output	maps}
    keras.layers.Conv2D(512, 3, activation="relu", padding="same", name='block8_conv1'),
    keras.layers.Conv2D(512, 3, activation="relu", padding="same", name='block8_conv2'),
    keras.layers.Conv2D(512, 3, activation="relu", padding="same", name='block8_conv3'),
    keras.layers.Conv2D(512, 3, activation="relu", padding="same", name='block8_conv4'),
    keras.layers.Conv2D(512, 3, activation="relu", padding="same", name='block8_conv5'),

    # Average	pooling
    keras.layers.AveragePooling2D(strides=(2, 2)),

    keras.layers.Flatten(),

    # Output Layer with softmax activation
    keras.layers.Dense(1000, activation="softmax", name='output')
])

print(model.summary())


feature_map_num = ((64+(6*64)) + (128+(7*128)) + (256+(11*256)) + (512+(5*512)))

print("\n\n Number of feature maps : ", feature_map_num)


