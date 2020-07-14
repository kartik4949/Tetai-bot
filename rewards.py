
class Rewards:
    def __init__(self , name):
        self_name = name
        self._MAXLINES= 4

    @property
    def max_line(self,):
        return self._MAXLINES

    @staticmethod
    def _line_scored(lines):
        reward = (lines/4)**2
        return reward

