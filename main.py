import isort

isort.file("main.py")
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras import Model, layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Train and Test directories
train_dir = "/Users/shubhamporiya/Documents/Sem3/ML/project/train"
validation_dir = "/Users/shubhamporiya/Documents/Sem3/ML/project/test"

# Adding one folder forward
train_normal_dir = os.path.join(train_dir, "NORMAL")
train_pneumonia_dir = os.path.join(train_dir, "PNEUMONIA")
validation_normal_dir = os.path.join(validation_dir, "NORMAL")
validation_pneumonia_dir = os.path.join(validation_dir, "PNEMONIA")

# Storing file names
train_normal_fnames = os.listdir(train_normal_dir)
# print(train_normal_fnames[:10])

train_pneumonia_fnames = os.listdir(train_pneumonia_dir)
train_pneumonia_fnames.sort()
# print(train_pneumonia_fnames[:10])


# Model Building
# ==============

# Model 1: Conv-Pool-Conv-Pool-Conv-Pool-Flatten-Dense-Dense-Softmax
# ========
# Our input feature map is 150x150x3: 150x150 for the image pixels
# And 3 for the three color channels: R, G, and B
img_input = layers.Input(shape=(150, 150, 3))

# First convolution extracts 16 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(filters=16, kernel_size=3, activation="relu", name="convolution1")(
    img_input
)
x = layers.MaxPooling2D(pool_size=2, name="maxpool1")(x)

# Second convolution extracts 32 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", name="convolution2")(x)
x = layers.MaxPooling2D(pool_size=2)(x)

# Third convolution extracts 64 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu", name="convolution3")(x)
x = layers.MaxPooling2D(pool_size=2)(x)

# Flatten feature map to a 1-dim tensor so we can add fully connected layers
x = layers.Flatten(name="Flatten")(x)

# Create a fully connected layer with ReLU activation and 512 hidden units
x = layers.Dense(units=512, activation="relu", name="fullyconnected1")(x)

# Crete a fully connected layer 2 with ReLU activation and 84 hidden units
x = layers.Dense(units=64, activation="relu", name="fullyconnected2")(x)

# Create output layer with a single node and sigmoid activation
output = layers.Dense(1, activation="sigmoid")(x)

# Create model:
# input = input feature map
# output = input feature map + stacked convolution/maxpooling layers + fully
# connected layer + sigmoid output layer
model1 = Model(img_input, output)

model1.summary()

# compiling the model
model1.compile(loss="binary_crossentropy", optimizer=RMSprop(lr=0.001), metrics=["acc"])


# Model 2: Conv-Pool-Conv-Pool-Flatten-Flatten-Dense-Softmax
# ========
# First convolution extracts 16 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
y = layers.Conv2D(filters=16, kernel_size=3, activation="relu", name="convolution1")(
    img_input
)
y = layers.MaxPooling2D(pool_size=2, name="maxpool1")(y)

# Second convolution extracts 32 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
y = layers.Conv2D(filters=32, kernel_size=3, activation="relu", name="convolution2")(y)
y = layers.MaxPooling2D(pool_size=2, name="maxpool2")(y)

y = layers.Flatten(name="Flatten1")(y)
y = layers.Flatten(name="Flatten2")(y)

y = layers.Dense(units=512, activation="relu", name="fullyconnected1")(y)

output = layers.Dense(1, activation="sigmoid")(y)

model2 = Model(img_input, output)

model2.summary()

# compiling the model
model2.compile(loss="binary_crossentropy", optimizer=RMSprop(lr=0.001), metrics=["acc"])

# Model 3: VGG Conv-Conv-Pool-Conv-Conv-Pool-Flatten-Dense-Softmax
# =======
x = layers.Conv2D(filters=16, kernel_size=3, activation="relu", name="convolution1")(
    img_input
)
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", name="convolution2")(x)

x = layers.MaxPooling2D(pool_size=2, name="maxpool1")(x)

x = layers.Conv2D(filters=64, kernel_size=3, activation="relu", name="convolution3")(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu", name="convolution4")(x)

x = layers.MaxPooling2D(pool_size=2)(x)

x = layers.Flatten(name="Flatten1")(x)

x = layers.Dense(units=512, activation="relu", name="fullyconnected1")(x)

output = layers.Dense(1, activation="sigmoid")(x)

model3 = Model(img_input, output)

model3.summary()

# compiling the model
model3.compile(loss="binary_crossentropy", optimizer=RMSprop(lr=0.001), metrics=["acc"])

# ==================================================================================

# All images will be rescaled by 1./255
train_data = ImageDataGenerator(rescale=1.0 / 255)
val_data = ImageDataGenerator(rescale=1.0 / 255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_data.flow_from_directory(
    train_dir,  # This is the source directory for training images
    target_size=(150, 150),  # All images will be resized to 150x150
    batch_size=20,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode="binary",
)

# Flow validation images in batches of 20 using val_datagen generator
validation_generator = val_data.flow_from_directory(
    validation_dir, target_size=(150, 150), batch_size=20, class_mode="binary"
)

history1 = model1.fit(train_generator, epochs=15, validation_data=validation_generator)
history2 = model2.fit(train_generator, epochs=15, validation_data=validation_generator)
history3 = model3.fit(train_generator, epochs=15, validation_data=validation_generator)


def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(150, 150))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 150, 150, 3)
    # center pixel data
    img = img.astype("float32")
    img = img - [123.68, 116.779, 103.939]
    return img


# load an image and predict the class
# load the image
img = load_image(
    "/Users/shubhamporiya/Documents/Sem3/ML/project/test/PNEUMONIA/SARS-10.1148rg.242035193-g04mr34g0-Fig8a-day0.jpeg"
)
# predict the class
result = model1.predict(img)
if result[0] == 1:  # 1 is for Pneumonia
    print("PNEUMONIA")
else:  # 0 is for Normal
    print("NORMAL")


# Plots for the models loss and accuracy

plt.plot(history1.history["loss"])
plt.plot(history2.history["loss"])
plt.plot(history3.history["loss"])

plt.title("Comparision of loss")
plt.legend(["LeNet-5", "My Model", "VGG"])

# plt.plot(history1.history["acc"])
# plt.plot(history2.history["acc"])
# plt.plot(history3.history["acc"])

# plt.title("Comparision of Accuracy")
# plt.legend(["LeNet-5", "My Model", "VGG"])


# Mean of 3 model accuracies 
np.mean(history1.history["acc"])
np.mean(history2.history["acc"])
np.mean(history3.history["acc"])