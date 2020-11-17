import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.size": 12,
    "font.sans-serif": ["Helvetica"]})


RRTSTAR = [15.97, 26.51, 82.51, 98.41]
BITSTAR = [84.19, 85.97, 89.49, 91.54]
SST = [3.95, 9.85, 90.7, 98.79]
AITSTAR = [84.12, 84.37, 87.72, 88.52]
ABITSTAR = [84.15, 86.08, 89.95, 91.18]
OURS = [93.06, 93.06, 93.06, 93.06]

T = [0.05, 0.1, 1., 10.]

lw = 3.
plt.plot(T, RRTSTAR, linewidth=lw)
plt.plot(T, BITSTAR, linewidth=lw)
plt.plot(T, SST, linewidth=lw)
plt.plot(T, AITSTAR, linewidth=lw)
plt.plot(T, ABITSTAR, linewidth=lw)
plt.plot(T, OURS, '--', linewidth=lw)
plt.grid(which="both")

plt.legend(["RRT$^*$", "BIT$^*$", "SST", "AIT$^*$", "ABIT$^*$", "ours"], fontsize=12)

#plt.yscale("log")
plt.xscale("log")
plt.show()
