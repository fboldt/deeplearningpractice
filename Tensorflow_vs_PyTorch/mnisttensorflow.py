### Imports and definitions ###
# pip install tensorflow 

import tensorflow as tf
## import tensorflow_datasets as tfds
import numpy as np

# In Google Colab, can change runtime to GPU if desired. If a TensorFlow 
# operation has both CPU and GPU implementations, by default the GPU device 
# is prioritized when the operation is assigned.
device = tf.config.get_visible_devices()
print(device)

# Random seed for reproducibility
seed = 42
tf.keras.utils.set_random_seed(seed)

# Save the model at the end?
save_model = False

# Batch sizes for training and testing
batch_size = 64
test_batch_size = 14

# Training epochs
n_epochs = 10

# Learning rate
learning_rate = 1.0

# Decay rate for adjusting the learning rate
gamma = 0.7

# Number of target classes in the MNIST data
num_classes = 10

# Data input shape
input_shape = (28, 28, 1)

####################
### Loading data ###

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# The scaled mean and standard deviation of the MNIST dataset (precalculated)
data_mean = 0.1307
data_std = 0.3081

# Reshape the input data
x_train = x_train.reshape(x_train.shape[0], 
                          x_train.shape[1], 
                          x_train.shape[2], 1)

x_test = x_test.reshape(x_test.shape[0], 
                        x_test.shape[1], 
                        x_test.shape[2], 1)

# Normalize the data
x_train = (x_train/255.0 - data_mean) / data_std
x_test = (x_test/255.0 - data_mean) / data_std

# Convert labels to one-hot vectors
y_train = tf.one_hot(y_train.astype(np.int32), depth=num_classes)
y_test = tf.one_hot(y_test.astype(np.int32), depth=num_classes)

################################################
### Defining the neural network architecture ###

# Define the architecture of the neural network
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), strides=(1,1),
                                      padding='valid', 
                                      activation='relu',
                                      input_shape=input_shape),
    tf.keras.layers.Conv2D(64, (3,3), strides=(1,1),
                                      padding='valid',
                                      activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

############################################
### Choosing the optimizer and scheduler ###

# Decay the learning rate at a base rate of gamma roughly every epoch, which
# is len(x_train) steps
scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
    learning_rate,
    decay_steps=len(x_train),
    decay_rate=gamma)

# Define the optimizer to user for gradient descent
optimizer = tf.keras.optimizers.Adadelta(scheduler)

# Compile the model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

# Display a model summary
model.summary()

##########################
### Training the model ###

# Train the model
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=n_epochs,
          validation_data=(x_test, y_test),
          validation_batch_size=test_batch_size)

########################
### Saving the model ###

if save_model:
    model.save_weights("mnist_cnn_tf.ckpt")
