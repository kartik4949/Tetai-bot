import numpy as np
class Rewards:
    def __init__(self , name):
        self_name = name
        self._MAXLINES= 4

    @property
    def max_line(self,):
        return self._MAXLINES

    @staticmethod
    def max_height(field , reward):
        field = np.asarray(field)
        _reward =  (field!=0).argmax(axis=0)
        #Adding Height Reward Calculator
        _reward = max([20-i if (i > 0) else i for i in _reward])
        _reward /= -20
        if reward != 0:
            reward *= _reward
        else:
            reward = _reward
        return reward


    @staticmethod
    def _line_scored(lines):
        reward = (lines/4)**2
        return reward

    def reward(self , field , _lines_popped):
        reward = self._line_scored(_lines_popped)
        reward = self.max_height(field , reward)
        return reward

