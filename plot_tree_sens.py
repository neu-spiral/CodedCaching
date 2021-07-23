import matplotlib.pylab as plt
import pickle
import numpy as np
import copy

style = ['-.', '-', '--', ':']
x_layer = [[0], [1], [2,3,4], [5,6,7,8,9,10,11,12,13]]
z_layer = [[(0,1)], [(1,2), (1,3), (1,4)], [(2,5), (2,6), (2,7), (3,8), (3,9), (3,10), (4,11), (4,12), (4,13)]]

def lineplot(filename, x_ax, y_ax, xaxis_label, yaxis_label):
    P = len(y_ax) #number of plot
    A = len(y_ax[0]) # number of algorithm
    S = len(y_ax[0][0]) # number of parameter
    fig, ax = plt.subplots(nrows=P, ncols=1)
    fig.set_size_inches(5,P*2)

    for i in range(P):
        title = 'layer '+str(i)
        ax[i].set_title(title, fontsize=10)
        for j in [2,1,0]:
            # ax[i].set_yscale('log')
            ax[i].plot(x_ax, y_ax[i][j], style[j], linewidth=2)
            ax[i].set_ylabel('Cost', fontsize=10)
            ax[i].tick_params(labelsize=10)
    plt.xlabel(xaxis_label, fontsize=10)
    plt.subplots_adjust(hspace=0.4)

    lgd = plt.legend(labels=yaxis_label, fontsize=10, loc='upper center', bbox_to_anchor=(0.5, 4.3), ncol=A,)

    plt.show()
    fig.savefig('figure/'+filename+'.pdf', bbox_extra_artists=(lgd,), bbox_inches = 'tight')


def lineplot2(filename, x_ax, y_ax1, y_ax2, xaxis_label, yaxis_label):
    P = len(y_ax1) #number of plot
    A = len(y_ax1[0]) # number of algorithm
    S = len(y_ax1[0][0]) # number of parameter
    fig, ax = plt.subplots(nrows=P, ncols=2)
    fig.set_size_inches(7,P*3)

    for i in range(P):
        title = 'layer '+str(i)
        ax[i, 0].set_title(title, fontsize=10)
        ax[i, 1].set_title(title, fontsize=10)

        for j in [2,1,0]:
            # ax[i].set_yscale('log')
            ax[i,0].plot(x_ax, y_ax1[i][j], style[j], linewidth=2)
            ax[i,0].set_ylabel('Cost', fontsize=10)
            ax[i,0].tick_params(labelsize=10)

            ax[i,1].plot(x_ax, y_ax2[i][j], style[j], linewidth=2)
            ax[i,1].set_ylabel('Load', fontsize=10)
            ax[i,1].tick_params(labelsize=10)
        if i == P-1:
            ax[i,0].set_xlabel(xaxis_label, fontsize=10)
            ax[i,1].set_xlabel(xaxis_label, fontsize=10)

    plt.subplots_adjust(hspace=0.3)
    plt.subplots_adjust(wspace=0.3)


    lgd = plt.legend(labels=yaxis_label, fontsize=10, loc='upper center', bbox_to_anchor=(-0.2, 3.9), ncol=A,)

    plt.show()
    fig.savefig('figure/'+filename+'.pdf', bbox_extra_artists=(lgd,), bbox_inches = 'tight')

filename = 'penalty_all'
scale = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
y_ax = []

for mul in scale:
    y = []
    fname = 'tree_penalty_asymmetric/tree_up_' + str(mul)
    with open(fname, 'rb') as f:
        [x, z, objE, objV] = pickle.load(f)
    z_group = []
    for e_group in z_layer:
        objE = 0
        for e in e_group:
            ze = 0
            for ii in z[e]:
                ze += z[e][ii]
            objE += pow(ze, mul)
        if objE < 0:
            objE = 0
        z_group.append(objE)
    y.append(z_group)

    fname = 'tree_penalty_asymmetric/tree_up_ma_' + str(mul)
    with open(fname, 'rb') as f:
        [x, z, objE, objV] = pickle.load(f)
    z_group = []
    for e_group in z_layer:
        objE = 0
        for e in e_group:
            ze = 0
            for ii in z[e]:
                ze += z[e][ii]
            objE += pow(ze, mul)
        if objE < 0:
            objE = 0
        z_group.append(objE)
    y.append(z_group)

    fname = 'tree_penalty_asymmetric/tree_up_nocc_' + str(mul)
    with open(fname, 'rb') as f:
        [x, z, objE, objV] = pickle.load(f)
    z_group = []
    for e_group in z_layer:
        objE = 0
        for e in e_group:
            ze = 0
            for ii in z[e]:
                ze += z[e][ii]
            objE += pow(ze, mul)
        if objE < 0:
            objE = 0
        z_group.append(objE)
    y.append(z_group)

    y_ax.append(y)

y_ax1 = np.array(y_ax).transpose()

y_ax = []

for mul in scale:
    y = []
    fname = 'tree_penalty_asymmetric/tree_up_' + str(mul)
    with open(fname, 'rb') as f:
        [x, z, objE, objV] = pickle.load(f)
    z_group = []
    for e_group in z_layer:
        objE = 0
        for e in e_group:
            ze = 0
            for ii in z[e]:
                ze += z[e][ii]
            objE += pow(ze, 1)
        if objE < 0:
            objE = 0
        z_group.append(objE)
    y.append(z_group)

    fname = 'tree_penalty_asymmetric/tree_up_ma_' + str(mul)
    with open(fname, 'rb') as f:
        [x, z, objE, objV] = pickle.load(f)
    z_group = []
    for e_group in z_layer:
        objE = 0
        for e in e_group:
            ze = 0
            for ii in z[e]:
                ze += z[e][ii]
            objE += pow(ze, 1)
        if objE < 0:
            objE = 0
        z_group.append(objE)
    y.append(z_group)

    fname = 'tree_penalty_asymmetric/tree_up_nocc_' + str(mul)
    with open(fname, 'rb') as f:
        [x, z, objE, objV] = pickle.load(f)
    z_group = []
    for e_group in z_layer:
        objE = 0
        for e in e_group:
            ze = 0
            for ii in z[e]:
                ze += z[e][ii]
            objE += pow(ze, 1)
        if objE < 0:
            objE = 0
        z_group.append(objE)
    y.append(z_group)

    y_ax.append(y)

y_ax2 = np.array(y_ax).transpose()

lineplot2(filename, scale, y_ax1, y_ax2, 'Penalty Multiplier', ['No Cross Coding', 'Maddah-Ali','Coded Cache'])
