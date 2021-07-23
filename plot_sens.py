import matplotlib.pylab as plt
import pickle
import numpy as np

style = ['-.', '-', '--', ':']


def lineplot(filename, x_ax, y_ax, xaxis_label, yaxis_label):
    fig, ax = plt.subplots()
    fig.set_size_inches(7,7)
    # for i in range(len(y_ax)):
    for i in [2,1,0]:
        ax.plot(x_ax, y_ax[i], style[i], linewidth=2)
    plt.xlabel(xaxis_label, fontsize=20)
    plt.ylabel('Cost', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(labels=yaxis_label, fontsize=20)
    plt.tight_layout()

    plt.show()
    # fig.savefig('figure/'+filename+'.pdf')

# x_ax = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
x_ax = [1.0, 1.5, 2.0, 2.5, 3.0]
# x_ax = [0.5, 1.0, 1.5, 2.0, 2.5]
filename = 'penalty_0.6'

y_ax = []
for request in x_ax:
    fname = filename + '/Maddah-Ali_' + str(request)
    with open(fname, 'rb') as f:
        [x, z1, objE1, objV] = pickle.load(f)
    objE1 = 0
    for e in z1:
        ze = 0
        for ii in z1[e]:
            ze += z1[e][ii]
        objE1 += ze
    fname = filename + '_ma/Maddah-Ali_' + str(request)
    with open(fname, 'rb') as f:
        [x, z2, objE2, objV] = pickle.load(f)
    objE2 = 0
    for e in z2:
        ze = 0
        for ii in z2[e]:
            ze += z2[e][ii]
        objE2 += ze
    fname = filename + '_nocc/Maddah-Ali_' + str(request)
    with open(fname, 'rb') as f:
        [x, z3, objE3, objV] = pickle.load(f)
    objE3 = 0
    for e in z3:
        ze = 0
        for ii in z3[e]:
            ze += z3[e][ii]
        objE3 += ze
    y_ax.append([objE1, objE2, objE3])

y_ax = np.array(y_ax).transpose()

lineplot(filename, x_ax,y_ax,'Power of Transmission', ['No Cross Coding', 'Maddah-Ali','Coded Cache'])
# lineplot(x_ax,y_ax,'power of load', ['Cache', 'Trans'])