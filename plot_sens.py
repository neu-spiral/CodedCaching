import matplotlib.pylab as plt
import pickle
import numpy as np


def lineplot(x_ax, y_ax, xaxis_label, yaxis_label):
    fig, ax = plt.subplots()
    fig.set_size_inches(7,7)
    for i in range(len(y_ax)):
        ax.plot(x_ax, y_ax[i])
    plt.xlabel(xaxis_label)
    plt.ylabel('Cost')
    plt.legend(labels=yaxis_label)
    plt.tight_layout()

    plt.show()

x_ax = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
y_ax = []
for request in x_ax:
    fname = 'Maddah-Ali_' + str(request)
    with open(fname, 'rb') as f:
        [x, z, objE, objV] = pickle.load(f)
    y_ax.append([objV, objE, objV+objE])
    #     x = pickle.load(f)

y_ax = np.array(y_ax).transpose()

lineplot(x_ax,y_ax,'difference', ['Cache', 'Trans', 'Total'])