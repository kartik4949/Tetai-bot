import os
import time
import tensorflow as tf

class Rewards:
    def __init__(self , name):
        self_name = name

    @staticmethod
    def _num_blocks_left(matrix):
        num_block = tf.math.count_nonzero(matrix)
        return num_block

    def reward(self, matrix):
        num_block = Rewards._num_blocks_left(matrix)
        return num_block

