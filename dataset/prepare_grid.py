import numpy as np
from matplotlib import pyplot as plt

#a = 8
a = 88
#a = 128
x = np.linspace(1., 24.0, a)
b = 88
#b = 128
#b = 8
y = np.linspace(-12.8, 12.8, b)
c = 46
th = np.linspace(- 90 * np.pi / 180, 90 * np.pi / 180, c)
X, Y, TH = np.meshgrid(x, y, th)
X = X.flatten()
Y = Y.flatten()
TH = TH.flatten()
pk = np.stack([X, Y, TH, np.zeros_like(TH)], 1).astype(np.float32)
n = a * b * c

p0 = np.ones_like(pk) * np.array([0.4, 0., 0., 0.])[np.newaxis]
ddy0 = np.zeros_like(X)
lines = [str(ddy0[i]) + " " + " ".join(["{:.5f}".format(x) for x in p0[i].tolist()]) + " "
         + " ".join(["{:.5f}".format(x) for x in pk[i].tolist()]) + "\n" for i in range(p0.shape[0])]
print(lines)

with open("./data/grid.path", 'w') as fh:
    fh.writelines(lines)
