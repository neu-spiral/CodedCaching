import matplotlib.pylab as plt
import pickle
import copy
import numpy as np

def barplot(plotname, x, ylabel):
    N = len(x[0][0]) # number of item
    M = len(x[0]) # number of layer
    Algs = ['Coded Cache', 'Maddah-Ali', 'No Cross Coding']
    hatches = [ '/', '\\\\', '+', '--', '']
    fig, ax = plt.subplots(nrows=M, ncols=1)
    fig.set_size_inches(7, M*2)
    numb_bars = len(x) + 1

    ind = np.arange(0, numb_bars * N, numb_bars)
    for i in range(M):
        title = 'layer '+str(i)
        ax[i].set_title(title, fontsize=10)
        x_ax = ['l', 'h', '(l,h)', '(l,l)', '(h,h)']
        y_ax = x[0][i].values()
        ax[i].set_yscale('log')
        width = 1
        ax[i].bar(ind, y_ax, width = width, hatch = hatches[0], label=Algs[0])
        y_ax = x[1][i].values()
        ax[i].bar(ind+width, y_ax, width = width, hatch = hatches[1], label=Algs[1])
        y_ax = x[2][i].values()
        ax[i].bar(ind+2*width, y_ax, width = width, hatch = hatches[2], label=Algs[2])
        ax[i].tick_params(labelsize=10)
        ax[i].set_ylabel(ylabel, fontsize=10)
        ax[i].set_xticks(ind + width)
        ax[i].set_xticklabels(x_ax, fontsize=10)
    plt.xlabel('Item', fontsize=10)
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    lgd = fig.legend(labels = Algs, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=len(Algs), fontsize=10)
    plt.show()
    fig.savefig(plotname, bbox_extra_artists=(lgd,), bbox_inches = 'tight')

fname1 = 'tree_asymmetric/tree_up'
fname2 = 'tree_asymmetric/tree_up_ma'
fname3 = 'tree_up_nocc'

with open(fname1, 'rb') as f:
    [x1, z1, objE, objV] = pickle.load(f)
with open(fname2, 'rb') as f:
    [x2, z2, objE, objV] = pickle.load(f)
with open(fname3, 'rb') as f:
    [x3, z3, objE, objV] = pickle.load(f)

x_layer = [[0], [1], [2,3,4], [5,6,7,8,9,10,11,12,13]]
z_layer = [[(0,1)], [(1,2), (1,3), (1,4)], [(2,5), (2,6), (2,7), (3,8), (3,9), (3,10), (4,11), (4,12), (4,13)]]

def layer_result(x_layer, x):
    x_group = []
    for v_group in x_layer:
        storage = {'l': 0, 'h': 0, ('l', 'h'): 0, ('l', 'l'): 0, ('h', 'h'): 0}
        for v in v_group:
            for ii in x[v]:
                if isinstance(ii, tuple):
                    if ii[0]<10 and ii[1]<10:
                        storage[('l', 'l')] += x[v][ii]
                    if ii[0]>=10 and ii[1]>=10:
                        storage[('h', 'h')] += x[v][ii]
                    else:
                        storage[('l', 'h')] += x[v][ii]
                else:
                    if ii<10:
                        storage['l'] += x[v][ii]
                    else:
                        storage['h'] += x[v][ii]
        storage_temp = copy.deepcopy(storage)
        x_group.append(storage_temp)
    return x_group

x_group1 = layer_result(x_layer, x1)
z_group1 = layer_result(z_layer, z1)
x_group2 = layer_result(x_layer, x2)
z_group2 = layer_result(z_layer, z2)
x_group3 = layer_result(x_layer, x3)
z_group3 = layer_result(z_layer, z3)

barplot('figure/x_asymmetric.pdf', [x_group1, x_group2, x_group3], 'cache')
# barplot('figure/z_asymmetric.pdf', [z_group1, z_group2, z_group3], 'load')


