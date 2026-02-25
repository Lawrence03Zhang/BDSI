import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers, initializers

from multi_detection.config import Config


class CNN(keras.Model):

    def __init__(self):
        super(CNN, self).__init__()
        self.cnn1 = None
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.cnn2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = layers.Dense(512, name="Attribute Normal", kernel_initializer=initializers.GlorotUniform())
        self.fc2 = layers.Dense(512, name="Attribute Normal", kernel_initializer=initializers.GlorotUniform())
        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        self.fc3 = layers.Dense(Config.out_size, name="Attribute Normal",
                                kernel_initializer=initializers.GlorotUniform())
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs, training=None):
        image_mat = inputs
        if self.cnn1 == None:
            B, X, Y, R = tf.shape(image_mat)
            self.cnn1 = layers.Conv2D(32, (3, 3), activation='relu', input_shape=(X, Y, R), padding='same')
        out = self.cnn1(image_mat)
        out = self.pool1(out)
        out = self.cnn2(out)
        out = self.pool2(out)
        out = self.flatten(out)

        # out = self.fc1(out)
        # out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out
