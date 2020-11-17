import matplotlib.pyplot as plt


class Environment:
    def __init__(self, fname):
        self.resolution = 0.2
        img = plt.imread(fname)[:, :, 0]
        img = img < 0.5
        self.height = img.shape[0]
        self.width = img.shape[1]
        self.map = img
