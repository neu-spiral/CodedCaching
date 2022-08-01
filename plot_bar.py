import matplotlib.pylab as plt
import argparse, pickle
import numpy as np
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# topology_map = {}
topology_map = {'erdos_renyi': 'ER', 'grid_2d': 'grid', 'hypercube': 'HC', 'expander': 'expander',
                'small_world': 'SW', 'barabasi_albert': 'BA', 'watts_strogatz': 'WS',
                'geant': 'geant', 'abilene': 'abilene', 'dtelekom': 'dtelekom'}
algorithm = ['SCC', 'RC-SMANT', 'RC-CT', 'SMAN', 'CC']
colors = ['r', 'sandybrown', 'gold', 'darkseagreen', 'c', 'dodgerblue', 'm']
hatches = ['/', '\\\\', '|', 'o', '--', '', '////', 'x', '+', '.', '\\']
Dir = ['nocc/', 'random_ma/', 'random_cc/', 'ma/', 'cc/']


def readresult(fname):
    with open(fname, 'rb') as f:
        result = pickle.load(f)
    return result


def barplot(x):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(18, 4)
    N = len(topology_map)
    numb_bars = len(algorithm) + 1
    ind = np.arange(0, numb_bars * N, numb_bars)
    width = 1
    for i in range(len(algorithm)):
        y_ax = x[algorithm[i]].values()
        ax.bar(ind + i * width, y_ax, width=width, hatch=hatches[i], color=colors[i], label=algorithm[i],
                  edgecolor='k', linewidth=1.5, log=True)
    ax.tick_params(labelsize=10)
    ax.set_ylabel('Norm. Cost', fontsize=15)
    ax.set_xlabel('Topology', fontsize=15)
    ax.set_xticks(ind + width * (len(algorithm) - 1) / 2)
    ax.set_xticklabels(x[algorithm[i]].keys(), fontsize=13)
    ax.grid(axis='y', linestyle='--')

    # plt.ylim(0, max(x['Random_MAN'].values())+1)
    plt.tight_layout()
    lgd = fig.legend(labels=algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=len(algorithm), fontsize=13)
    plt.show()
    fig.savefig('figure/topologies2.pdf', bbox_extra_artists=(lgd,),bbox_inches='tight')


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
        fname = Dir[-1] + top
        result = readresult(fname)
        compare = result[-2]
        print(top, compare)
        obj[algorithm[-1]][topology_map[top]] = 1
        for j in range(len(algorithm)-1):
            fname = Dir[j] + top
            result = readresult(fname)
            obj[algorithm[j]][topology_map[top]] = result[-2] / compare
    barplot(obj)
