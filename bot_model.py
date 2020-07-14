import os
import time

import tensorflow as tf
import numpy as np


class tetris_game_config:
    WIDTH = HEIGHT =80
    NAME = "Tetai"


class tetai_model(tf.keras.Model, tetris_game_config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._name = tetris_game_config.NAME
        self._width = tetris_game_config.WIDTH
        self._height = tetris_game_config.HEIGHT
        self.conv1 = tf.keras.layers.Conv2D(
            16,
            kernel_size=(8,8),
            input_shape =(80,80,4),
            strides=(3, 3),
            kernel_initializer="he_uniform",
            kernel_regularizer="l1",
            activation=tf.nn.leaky_relu,
        )
        self.conv2 = tf.keras.layers.Conv2D(
            32,
            kernel_size=(4, 4),
            strides=(2, 2),
            kernel_initializer="he_uniform",
            kernel_regularizer="l1",
            activation=tf.nn.leaky_relu,
        )
        self.conv3 = tf.keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_initializer="he_uniform",
            kernel_regularizer="l1",
            activation=tf.nn.leaky_relu,
        )
        self.conv4 = tf.keras.layers.Conv2D(
            32,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_initializer="he_uniform",
            kernel_regularizer="l1",
            activation=tf.nn.leaky_relu,
        )
        self.dense = tf.keras.layers.Dense(10)
        self.out = tf.keras.layers.Dense(4)
        self.flatten = tf.keras.layers.Flatten()

    @property
    def name(self):
        return self._name

    def __new__(cls, *args, **kwargs):
        print(
            "-+-+-+-+-+-+-+-+-+-Say , hi to new born bot!! named> Tetai-+-+-+-+-+-+-+-+-+-"
        )
        return super().__new__(cls, *args, **kwargs)

    def call(self, x, training=None):
        x = self.conv1(x , training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        x = self.flatten(x)
        x = self.dense(x, training=training)
        x = self.out(x, training=training)
        return x


if __name__ == "__main__":
    tm = tetai_model()
    for i in range(100):
        inputs = tf.random.uniform((1, 20, 10))
        x = tm(inputs)
        print(x)
    tm.summary()
