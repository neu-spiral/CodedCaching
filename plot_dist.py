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

def barplot2(x):
    fig, ax = plt.subplots(nrows=len(x), ncols=1)
    fig.set_size_inches(7, 8)
    j = 0
    for i in x:
        title = str(i)
        ax[j].set_title(title)
        x_ax = [str(x) for x in x[i].keys()]
        y_ax = x[i].values()
        ax[j].bar(x_ax, y_ax)
        j += 1
    plt.tight_layout()
    plt.show()


# barplot(x[0])
# barplot(x[1])
# barplot(x[2])
# barplot(z[(0,1)])
# barplot(z[(0,2)])
fname = 'Maddah-Ali_1.0'
with open(fname, 'rb') as f:
    [x, z, objE, objV] = pickle.load(f)
barplot2(x)
barplot2(z)