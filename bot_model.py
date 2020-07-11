import os
import time

import tensorflow as tf
import numpy as np


class tetris_game_config:
    WIDTH = 10
    HEIGHT = 20
    NAME = "Tetai"


class tetai_model(tf.keras.Model, tetris_game_config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._name = tetris_game_config.NAME
        self._width = tetris_game_config.WIDTH
        self._height = tetris_game_config.HEIGHT
        self.dense1 = tf.keras.layers.Dense(
            300,
            kernel_initializer="he_uniform",
            kernel_regularizer="l1",
            activation=tf.nn.leaky_relu,
        )
        self.dense2 = tf.keras.layers.Dense(
            100,
            kernel_initializer="he_uniform",
            kernel_regularizer="l1",
            activation=tf.nn.leaky_relu,
        )
        self.dense3 = tf.keras.layers.Dense(
            50,
            kernel_initializer="he_uniform",
            kernel_regularizer="l1",
            activation=tf.nn.leaky_relu,
        )
        self.dense4 = tf.keras.layers.Dense(
            10,
            kernel_initializer="he_uniform",
            kernel_regularizer="l1",
            activation=tf.nn.leaky_relu,
        )
        self.dense5 = tf.keras.layers.Dense(
            4,
            kernel_initializer="he_uniform",
            kernel_regularizer="l1",
            activation=tf.nn.leaky_relu,
        )
        self.softmax = tf.keras.layers.Activation("softmax", dtype=tf.float32)
        self.flatten = tf.keras.layers.Flatten()

    @property
    def name(self):
        return self._name

    def __new__(cls, *args, **kwargs):
        print("-+-+-+-+-+-+-+-+-+-Say , hi to new born bot!! named> Tetai-+-+-+-+-+-+-+-+-+-")
        return super().__new__(cls, *args, **kwargs)

    def call(self, x, training=None):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        _out_prob = self.softmax(x)
        _stack_prob = tf.concat(_out_prob, axis=0)
        action = tf.compat.v1.multinomial(tf.math.log(_stack_prob), num_samples=1)
        # bug tf squeeze if not provided axis
        action = tf.squeeze(action, axis=0)
        return action


if __name__ == "__main__":
    inputs = tf.random.uniform((1, 20, 10))
    tm = tetai_model()
    x = tm(inputs)
    tm.summary()
