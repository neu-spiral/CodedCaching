import matplotlib.pylab as plt
import argparse, pickle
import numpy as np

# topology_map = {'expander': 'expander',}
topology_map = {'erdos_renyi': 'ER', 'grid_2d': 'grid', 'hypercube': 'HC',
                'small_world': 'SW', 'barabasi_albert': 'BA', 'watts_strogatz': 'WS',
                'geant': 'geant', 'abilene': 'abilene',}
algorithm = ['Coded Cache', 'MAN', 'No Cross Coding', 'Random_CC', 'Random_MA']
colors = ['r', 'sandybrown', 'gold', 'darkseagreen', 'c', 'dodgerblue', 'm']
hatches = ['/', '\\\\', '|', 'o', '--', '', '////', 'x', '+', '.', '\\']
# Dir = ['cc/', 'ma/', 'nocc/', 'random_cc/', 'random_ma/']
Dir = ['cc_small/', 'ma_small/', 'nocc_small/', 'random_cc_small/', 'random_ma_small/']


def readresult(fname):
    with open(fname, 'rb') as f:
        result = pickle.load(f)
    return result


def barplot(x):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(18, 3)
    N = len(topology_map)
    numb_bars = len(algorithm) + 1
    ind = np.arange(0, numb_bars * N, numb_bars)
    width = 1
    for i in range(len(algorithm)):
        y_ax = x[algorithm[i]].values()
        ax.bar(ind + i * width, y_ax, width=width, hatch=hatches[i], color=colors[i], label=algorithm[i],
                  edgecolor='k', linewidth=1.5)
    ax.tick_params(labelsize=10)
    ax.set_ylabel('Norm. Cost', fontsize=15)
    ax.set_xlabel('Topology', fontsize=15)
    ax.set_xticks(ind + width * (len(algorithm) - 1) / 2)
    ax.set_xticklabels(x[algorithm[i]].keys(), fontsize=13)
    ax.grid(axis='y', linestyle='--')

    plt.ylim(0, max(x['Random_MA'].values())+1)
    plt.tight_layout()
    lgd = fig.legend(labels=algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=len(algorithm), fontsize=13)
    plt.show()
    fig.savefig('figure/topologies.pdf', bbox_extra_artists=(lgd,),bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot bar',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--graph_type', type=str, help='Graph type', choices=['Maddah-Ali', 'tree',
                                                                              'erdos_renyi', 'balanced_tree',
                                                                              'hypercube', "cicular_ladder", "cycle",
                                                                              "grid_2d",
                                                                              'lollipop', 'expander', 'star',
                                                                              'barabasi_albert', 'watts_strogatz',
                                                                              'regular', 'powerlaw_tree', 'small_world',
                                                                              'geant', 'abilene', 'dtelekom',
                                                                              'servicenetwork'])

    args = parser.parse_args()

    obj = {}
    for alg in algorithm:
        obj[alg] = {}
        for top in topology_map.values():
            obj[alg][top] = 0

    for top in topology_map:
        fname = Dir[0] + top
        result = readresult(fname)
        compare = result[-2]
        print(top, compare)
        obj[algorithm[0]][topology_map[top]] = 1
        for j in range(1, len(algorithm)):
            fname = Dir[j] + top
            result = readresult(fname)
            obj[algorithm[j]][topology_map[top]] = result[-2] / compare
    barplot(obj)
