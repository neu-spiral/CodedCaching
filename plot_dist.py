import matplotlib.pylab as plt
import pickle

def barplot(x, f_name=''):
    x_ax = x.keys()
    x_ax = [str(x) for x in x_ax]
    y_ax = x.values()
    # y_ax = [y for y in y_ax]
    plt.bar(x_ax, y_ax)
    plt.show()
    # plt.savefig(f_name)

def barplot2(plotname, x):
    fig, ax = plt.subplots(nrows=len(x), ncols=1)
    fig.set_size_inches(7, 8)
    j = 0
    for i in x:
        title = str(i)
        ax[j].set_title(title, fontsize=20)
        x_ax = [str(x) for x in x[i].keys()]
        y_ax = x[i].values()
        ax[j].bar(x_ax, y_ax)
        # ax[j].tick_params(labelsize=15)
        j += 1
    plt.xlabel('Item', fontsize=20)
    plt.tight_layout()
    plt.show()
    # fig.savefig(plotname)

def barplot3(x):
    # plot result of each layer
    fig, ax = plt.subplots()
    fig.set_size_inches(7, 4)
    ax.bar(range(1,len(x)+1), x)
    plt.xlabel('layer', fontsize=20)
    plt.ylabel('cost', fontsize=20)
    plt.tight_layout()

    plt.show()
    fig.savefig('figure/plot')



# barplot(x[0])
# barplot(x[1])
# barplot(x[2])
# barplot(z[(0,1)])
# barplot(z[(0,2)])
# fname = 'penalty_0.6_h/Maddah-Ali_3.0'
plotname1 = 'figure/cache_ma.pdf'
plotname2 = 'figure/load_ma.pdf'
fname = 'penalty_0.6/Maddah-Ali_2.5'
with open(fname, 'rb') as f:
    [x, z, objE, objV] = pickle.load(f)
barplot2(plotname1, x)
barplot2(plotname2, z)

# x_layer = [[0], [1], [2,3,4], [5,6,7,8,9,10,11,12,13]]
# z_layer = [[(0,1)], [(1,2), (1,3), (1,4)], [(2,5), (2,6), (2,7), (3,8), (3,9), (3,10), (4,11), (4,12), (4,13)]]
# x_group = []
# z_group = []
# for e_group in z_layer:
#     objE = 0
#     for e in e_group:
#         ze = 0
#         for ii in z[e]:
#             ze += z[e][ii]
#         objE += pow(ze, 2)
#     z_group.append(objE)
# for v_group in x_layer:
#     objV = 0
#     for v in v_group:
#         xv = 0
#         for ii in x[v]:
#             xv += x[v][ii]
#         objV += pow(xv, 2)
#     x_group.append(objV)
#
# barplot3(x_group)
# barplot3(z_group)