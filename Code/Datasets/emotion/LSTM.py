import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers, initializers


class LSTM(keras.Model):

    def __init__(self):
        super(LSTM, self).__init__()
        self.rnn1 = layers.LSTM(64, activation='tanh', dropout=0.5, input_shape=(29, 100),
                                return_sequences=True)
        self.rnn2 = layers.LSTM(64, activation='tanh', dropout=0.5)
        self.fc1 = layers.Dense(512, name="Attribute Normal",
                                kernel_initializer=initializers.GlorotUniform())
        self.fc2 = layers.Dense(64, name="Attribute Normal", kernel_initializer=initializers.GlorotUniform())
        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        self.fc3 = layers.Dense(3, name="Attribute Normal",
                                kernel_initializer=initializers.GlorotUniform())
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs, training=None):
        out = self.rnn1(inputs)
        out = self.rnn2(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        # out = self.dropout(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
