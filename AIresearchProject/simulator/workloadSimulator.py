import numpy as np

class workloadSimulator:
    def __init__(self):
        self.t = 0

    def nextLoad(self):
        load = 0.5 + 0.4 * np.sin(self.t / 10)
        self.t += 1
        return np.clip(load, 0, 1)