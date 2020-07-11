import os
import time

import tensorflow  as tf
import numpy as np

class tetris_game_config:
    WIDTH = 10
    HEIGHT = 20
    NAME = "Tetai"


class tetai_model(tf.keras.Model , tetris_game_config):
    def __init__(self , **kwargs):
        self._name = tetris_game_config.NAME
        self._width  = tetris_game_config.WIDTH
        self._height = tetris_game_config.HEIGHT
        self._input = tf.keras.layers.Input((self._height,self._width))
        self.dense1 = tf.keras.layers.Dense(100 , activation=tf.nn.leaky_relu)
        self.dense2 = tf.keras.layers.Dense(1000 , activation=tf.nn.leaky_relu)
        self.dense3 = tf.keras.layers.Dense(500 , activation=tf.nn.leaky_relu)
        self.dense4 = tf.keras.layers.Dense(100 , activation=tf.nn.leaky_relu)
        self.dense5 = tf.keras.layers.Dense(50 , activation=tf.nn.leaky_relu)
        self.dense6 = tf.keras.layers.Dense(4 , activation=tf.nn.leaky_relu)
        self.softmax = tf.keras.layers.Activation('softmax' , dtype=tf.float32)
        super().__init__( **kwargs)

    @property
    def name(self):
        return self._name


    def __new__(cls , *args , **kwargs):
        print("Say , hi to new born bot!! named> Tetai")
        return super().__new__(cls,*args , **kwargs)

    def __call__(self, x, training=None):
        inputs = self._input(x)
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.dense6(x)
        _out_prob = self.softmax(x)
        _stack_prob = tf.concat(_out_prob , axis=0)
        action = tf.multinomial(tf.math.log(_stack_prob) , num_samples=1)
        return action



