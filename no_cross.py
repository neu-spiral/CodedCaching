import networkx as nx
import cvxpy as cp
import numpy as np
import logging, argparse
from collections import Counter

from cvxpy.atoms.elementwise.minimum import minimum
from cvxpy.atoms.elementwise.power import power
import pickle

# questions:
# Do we need upper bound of 1 on any variables?
# Can we incorporate capacity constraints?
# How to x_{v(i,j)} contribute to z/capacity constraints/costs etc.?

def minhalf(a, b):
    """ Return min(a,b) + 0.5*[a-b]_+ = 0.5 *(a + min(a,b)) """
    return 0.5 * (a + minimum(a, b))


class CacheNetwork:
    """ A class modeling a cache network. """

    def __init__(self, G, we, wv, dem, cat, hyperE, c={}):
        """ Instantiate a new network. Inputs are:
            - G : a networkx graph
            - we : dictionary containing edge weight/cost functions; must be constructed from cvxpy atoms
            - wv : dictionary containing node storage weight/cost functions; must be constructed from cvxpy atoms
            - dem : dictionary containing demand; d[t][i] contains demand for item i in target node t
            - cat :  content catalog
            - c :  dictionary containing storage capacity constraints (optional)"""
        logging.debug("Initializing object...")
        self.G = G
        self.we = we
        self.wv = wv
        self.demand = dem
        self.catalog = sorted(cat)
        self.targets = sorted(dem.keys())
        self.c = c
        self.hyperE = hyperE

    def is_target(self, t, i):
        """Detect whether node t is a target for item i."""
        return t in self.demand and i in self.demand[t]

    def cvx_init(self):
        """Constuct cvxpy problem instance"""

        self.vars = {}  # used to access variables outside cvx program

        # cache variables
        logging.debug("Creating cache variables...")
        x = {}
        for v in self.G.nodes():
            x[v] = {}
            for i in self.catalog:
                x[v][i] = cp.Variable()

        self.vars['x'] = x

        # xi book-keeping variables
        logging.debug("Creating xi book-keeping variables...")
        xi = {}
        for t in self.demand:
            xi[t] = {}
            for v in self.G.nodes():
                xi[t][v] = {}
                for i in self.catalog:  # not self.demand[t], because of cross coding you may want to use traffic not demanded by t
                    xi[t][v][i] = cp.Variable()

        self.vars['xi'] = xi

        # z flow variables
        logging.debug("Creating z flow variables...")
        z = {}
        for e in self.G.edges():
            z[e] = {}
            for i in self.catalog:
                z[e][i] = cp.Variable()

        self.vars['z'] = z

        # rho book-keeping variables
        logging.debug("Creating rho book-keeping variables...")
        rho = {}
        for t in self.demand:
            rho[t] = {}
            for e in self.G.edges():
                rho[t][e] = {}
                for i in self.catalog:
                    rho[t][e][i] = cp.Variable()

        self.vars['rho'] = rho

        # mu decoding capability variables
        logging.debug("Creating mu decoding capability variables...")
        mu = {}
        for t in self.demand:
            mu[t] = {}
            for v in self.G.nodes():
                mu[t][v] = {}
                for i in self.catalog:
                    mu[t][v][i] = cp.Variable()

        self.vars['mu'] = mu

        # objective
        logging.debug("Creating objective...")
        obj = 0
        for e in self.we:
            ze = 0
            for ii in z[e]:
                ze += z[e][ii]
            obj += self.we[e](ze)
        capacities = list(self.c.values())
        if capacities[-1] == float('Inf'): # if no constraint
            for v in self.wv:
                xv = 0
                for ii in x[v]:
                    xv += x[v][ii]
                obj += self.wv[v](xv)

            # constraints
        logging.debug("Creating costraints...")
        constr = []
        logging.debug("Creating cache variable non-negativity constraints...")
        for v in x:
            for ii in x[v]:
                constr.append(x[v][ii] >= 0)
                # constr.append(x[v][ii] <= 1) no upper bounds on x
        logging.debug("Creating xi variable non-negativity constraints...")
        for t in xi:
            for v in xi[t]:
                for ii in xi[t][v]:
                    constr.append(xi[t][v][ii] >= 0)
        logging.debug("Creating rho variable non-negativity constraints...")
        for t in rho:
            for e in rho[t]:
                for ii in rho[t][e]:
                    constr.append(rho[t][e][ii] >= 0)
        logging.debug("Creating z variable non-negativity constraints...")
        for e in z:
            for ii in z[e]:
                constr.append(z[e][ii] >= 0)
        logging.debug("Creating mu variable non-negativity constraints...")
        for t in mu:
            for v in mu[t]:
                for ii in mu[t][v]:
                    constr.append(mu[t][v][ii] >= 0)

        # hyperedges for z flow
        logging.debug("Creating hyperedges for z flow variables...")
        for hyperedge in self.hyperE:
            for pos in range(len(hyperedge) - 1):
                for ii in z[hyperedge[pos]]:
                    constr.append(z[hyperedge[pos]][ii] == z[hyperedge[pos + 1]][ii])

        # book-keeping constraints
        logging.debug("Creating z book-keeping constraints...")
        for t in rho:
            for e in rho[t]:
                for ii in rho[t][e]:
                    constr.append(rho[t][e][ii] <= z[e][ii])

        logging.debug("Creating x book-keeping constraints...")
        for t in xi:
            for v in xi[t]:
                for ii in xi[t][v]:
                    constr.append(xi[t][v][ii] <= x[v][ii])

        # flow constraints
        logging.debug("Creating rho flow constraints...")

        # Decodability rate constraints

        logging.debug("Creating in flows...")

        in_flow = {}
        out_flow = {}
        for t in self.targets:
            in_flow[t] = {}
            out_flow[t] = {}

            for v in self.G.nodes():
                in_edges = self.G.in_edges(v)
                out_edges = self.G.out_edges(v)

                in_flow[t][v] = {}
                out_flow[t][v] = {}

                for i in self.catalog:
                    in_flow[t][v][i] = xi[t][v][i]
                    for e in in_edges:
                        in_flow[t][v][i] += rho[t][e][i]

                    out_flow[t][v][i] = 0
                    for e in out_edges:
                        out_flow[t][v][i] += rho[t][e][i]


        self.vars['in_flow'] = in_flow
        self.vars['out_flow'] = out_flow

        logging.debug("Creating decodability constraints...")
        for t in self.targets:
            for v in self.G.nodes():
                for i in self.catalog:
                    dec_rate = in_flow[t][v][i]
                    constr.append(dec_rate >= mu[t][v][i])

        # Outgoing flow is bounded by decodability

        logging.debug("Creating unitary outgoing flow constraints...")
        for t in self.targets:
            for v in self.G.nodes():
                for i in self.catalog:
                    constr.append(mu[t][v][i] >= out_flow[t][v][i])


        # Demand should be met

        logging.debug("Creating demand constraints...")
        for t in self.targets:
            for i in self.demand[t]:  # demand met should be restricted to i's in demand[t]
                constr.append(mu[t][t][i] >= self.demand[t][i])

        # Capacity constraints (optional)
        logging.debug("Creating cache variable capacity constraints...")
        for v in self.c:
            xv = 0
            for ii in x[v]:
                xv += x[v][ii]
            constr.append(xv <= self.c[v])

        self.problem = cp.Problem(cp.Minimize(obj), constr)
        logging.debug("Problem is DCP: " + str(self.problem.is_dcp()))

    def solve(self):
        """Solve cvxpy instance"""
        logging.debug("Running cvxpy solver...")
        self.problem.solve()

    def test_feasibility(self, tol=1e-5):
        """Confirm that the solution is feasible, upto tolerance tol."""

        x = self.vars['x']
        xi = self.vars['xi']
        z = self.vars['z']
        rho = self.vars['rho']
        mu = self.vars['mu']
        in_flow = self.vars['in_flow']
        out_flow = self.vars['out_flow']

        logging.debug('Asserting cache variable non-negativity...')
        for v in x:
            for ii in x[v]:
                assert (x[v][ii].value >= -tol), "Cache %s, item %s has negative x value: %f" % (v, ii, x[v][ii].value)

        logging.debug("Asserting xi variable non-negativity...")
        for t in xi:
            for v in xi[t]:
                for ii in xi[t][v]:
                    assert (xi[t][v][ii].value >= -tol), "Target %s cache %s item %s has negative xi value: %f" % (
                    t, v, ii, xi[t][v][ii].value)

        logging.debug("Asserting rho variable non-negativity")
        for t in rho:
            for e in rho[t]:
                for ii in rho[t][e]:
                    assert (rho[t][e][ii].value >= -tol), "Target %s edge %s item %s has negative rho value %f" % (
                    t, e, ii, rho[t][e][ii].value)

        logging.debug('Asserting flow variable non-negativity...')
        for e in z:
            for ii in z[e]:
                assert (z[e][ii].value >= -tol), "Edge %s, item %s has negative z value: %f" % (e, ii, z[v][ii].value)

        logging.debug("Asserting mu variable non-negativity")
        for t in mu:
            for v in mu[t]:
                for ii in mu[t][v]:
                    assert (mu[t][v][ii].value >= -tol), "Target %s cache %s item %s has negative mu value %f" % (
                    t, v, ii, mu[t][v][ii].value)

        logging.debug("Asserting hyperedges for z flow variables...")
        for hyperedge in self.hyperE:
            for pos in range(len(hyperedge) - 1):
                for ii in z[hyperedge[pos]]:
                    assert (abs(z[hyperedge[pos]][ii].value - z[hyperedge[pos + 1]][
                        ii].value) <= tol), "Edge %s and edge %s item %s do not have the same z with %f and %f" % (
                    hyperedge[pos], hyperedge[pos + 1], ii, z[hyperedge[pos]][ii].value,
                    z[hyperedge[pos + 1]][ii].value)
                    # assert(z[hyperedge[pos]][ii].value == z[hyperedge[pos+1]][ii].value ), "Edge %s and edge %s item %s do not have the same z with %f and %f" % (hyperedge[pos], hyperedge[pos+1], ii, z[hyperedge[pos]][ii].value, z[hyperedge[pos+1]][ii].value)

        logging.debug("Asserting z book-keeping constraints...")
        for t in rho:
            for e in rho[t]:
                for ii in rho[t][e]:
                    assert (z[e][ii].value - rho[t][e][
                        ii].value >= -tol), "Target %s edge %s item %s not satisfy z book-keeping with %f" % (
                    t, e, ii, z[e][ii].value - rho[t][e][ii].value)

        logging.debug("Asserting x book-keeping constraints...")
        for t in xi:
            for v in xi[t]:
                for ii in xi[t][v]:
                    assert (x[v][ii].value - xi[t][v][
                        ii].value >= -tol), "Target %s cache %s item %s not satisfy z book-keeping with %f" % (
                    t, v, ii, z[e][ii].value - rho[t][e][ii].value)

        logging.debug("Asserting decodability constraints...")
        for t in self.targets:
            for v in self.G.nodes():
                for i in self.catalog:
                    dec_rate = in_flow[t][v][i]
                    assert (dec_rate.value - mu[t][v][
                        i].value >= -tol), "Target %s cache %s item %s not satisfy decodability constraints with %f" % (
                    t, v, i, dec_rate - mu[t][v][i].value)

        logging.debug("Asserting unitary outgoing flow constraints...")
        for t in self.targets:
            for v in self.G.nodes():
                for i in self.catalog:
                    if type(out_flow[t][v][i]) is int:  # there is no out edges
                        assert (mu[t][v][
                                    i].value >= -tol), "Target %s cache %s item %s not satisfy unitary outgoing flow constraints with %f" % (
                        t, v, i, mu[t][v][i].value)
                    else:
                        assert (mu[t][v][i].value - out_flow[t][v][
                            i].value >= -tol), "Target %s cache %s item %s not satisfy unitary outgoing flow constraints with %f" % (
                        t, v, i, mu[t][v][i].value - out_flow[t][v][i].value)

        logging.debug("Asserting demand constraints...")
        for t in self.targets:
            for i in self.demand[t]:  # demand met should be restricted to i's in demand[t]
                assert (mu[t][t][i].value - self.demand[t][
                    i] >= -tol), "Target %s cache %s item %s not satisfy decodability constraints with %f" % (
                t, t, i, mu[t][t][i].value - self.demand[t][i].value)

        logging.debug("Asserting cache variable capacity constraints...")
        for v in self.c:
            xv = 0
            for ii in x[v]:
                xv += x[v][ii]
            assert (self.c[v] - xv.value >= -tol), "Cache %s not satisfy cache variable capacity costraints with %f" % (
            v, self.c[v] - xv)

    def solution(self):
        x = self.vars['x']
        z = self.vars['z']
        mu = self.vars['mu']
        for v in x:
            for ii in x[v]:
                x[v][ii] = x[v][ii].value
        for e in z:
            for ii in z[e]:
                z[e][ii] = z[e][ii].value
        for t in self.targets:
            for i in self.demand[t]:
                mu[t][t][i] = mu[t][t][i].value

        objE = 0
        for e in self.we:
            ze = 0
            for ii in z[e]:
                ze += z[e][ii]
            objE += self.we[e](ze).value
        objV = 0
        for v in self.wv:
            xv = 0
            for ii in x[v]:
                xv += x[v][ii]
            objV += self.wv[v](xv).value
        return x, z, mu, objE, objV


def main():
    parser = argparse.ArgumentParser(description='Simulate a Network of Coded Caches',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('outputfile', help='Output file')
    parser.add_argument('--debug_level', default='INFO', type=str, help='Debug Level',
                        choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'])
    parser.add_argument('--graph_type', type=str, help='Graph type', choices=['Maddah-Ali', 'tree'])
    parser.add_argument('--catalog_size', default=4, type=int, help='Catalog size')
    parser.add_argument('--random_seed', default=10, type=int, help='Random seed')
    parser.add_argument('--zipf_parameter', default=1.2, type=float,
                        help='parameter of Zipf, used in demand distribution')
    parser.add_argument('--capacity', default=float('Inf'), type=float, help='capacity of each node')
    parser.add_argument('--difference', default=0.0, type=float, help='difference of two items used in exploring')
    parser.add_argument('--penalty', default=1.0, type=float, help='penalty of edge')
    parser.add_argument('--penalty_mul', default=2.0, type=float, help='increase penalty')
    parser.add_argument('--capacity_mul', default=2.0, type=float, help='decrease capacity')

    args = parser.parse_args()

    args.debug_level = eval("logging." + args.debug_level)
    np.random.seed(args.random_seed)
    logging.basicConfig(level=args.debug_level)

    def graphGenerator():
        if args.graph_type == 'Maddah-Ali':
            return nx.balanced_tree(2, 1)
        if args.graph_type == 'tree':
            temp = nx.balanced_tree(3, 2)
            temp.add_node(-1)
            temp.add_edge(-1, 0)
            return temp

    def hyperedges():
        if args.graph_type == 'Maddah-Ali':
            return [[(0, 1), (0, 2)]]
        if args.graph_type == 'tree':
            hyperedges = []
            hyperedge_node = [2, 3, 4]
            for node_1 in hyperedge_node:
                hyperedge = []
                for node_2 in G.successors(node_1):
                    hyperedge.append((node_1, node_2))
            hyperedges.append(hyperedge)
            return hyperedges

    def targetGenerator():
        if args.graph_type == 'Maddah-Ali':
            return [1, 2]
        if args.graph_type == 'tree':
            return [5, 6, 7, 8, 9, 10, 11, 12, 13]

    logging.info('Generating graph...')

    temp_graph = graphGenerator()
    logging.debug('nodes: ' + str(temp_graph.nodes))  # list
    logging.debug('edges: ' + str(temp_graph.edges))  # list of node pair
    G = nx.DiGraph()  # generate a DiGraph
    temp_nodes = list(temp_graph.nodes)
    temp_nodes.sort()
    number_map = dict(zip(temp_nodes, range(len(temp_graph.nodes()))))
    G.add_nodes_from(number_map.values())  # add node from temp_graph to G

    for (x, y) in temp_graph.edges():  # add edge from temp_graph to G
        xx = number_map[x]
        yy = number_map[y]
        G.add_edge(min(xx, yy), max(xx, yy))

    logging.info('...done. Created graph with %d nodes and %d edges' % (G.number_of_nodes(), G.number_of_edges()))
    logging.debug('G is:' + str(G.nodes) + str(G.edges))
    # pos = nx.shell_layout(G)
    # nx.draw_networkx(G, pos)
    # plt.draw()
    # plt.show()
    we = {}
    wv = {}
    capacity = {}

    # for e in G.edges():
    #     we[e] = lambda x: power(x,args.penalty)

    z_layer = [[(0, 1)], [(1, 2), (1, 3), (1, 4)],
               [(2, 5), (2, 6), (2, 7), (3, 8), (3, 9), (3, 10), (4, 11), (4, 12), (4, 13)]]
    pen = args.penalty
    for layer in z_layer:
        for e in layer:
            we[e] = lambda x: power(x, pen)
        pen *= args.penalty_mul

    for v in G.nodes():
        wv[v] = lambda x: power(x, 1)

    x_layer = [[0], [1], [2, 3, 4], [5, 6, 7, 8, 9, 10, 11, 12, 13]]
    cap = args.capacity
    for layer in x_layer:
        for v in layer:
            capacity[v] = cap
        cap /= args.capacity_mul

    # for v in G.nodes():
    #     capacity[v] = args.capacity

    capacity[0] = args.catalog_size
    dem = {}
    targets = targetGenerator()
    catalog = range(args.catalog_size)
    scale = 100
    for t in targets:
        dem[t] = {}
        sample = np.random.zipf(args.zipf_parameter, 1000)
        demend = Counter(sample)
        for i in catalog:
            dem[t][i] = demend[args.catalog_size-i]/scale
            # dem[t][i] = demend[i+1] / scale

        scale -= 5
    print(dem)

    # dem[1] = {}
    # dem[2] = {}
    #
    # dem[1][0] = (1+args.difference)/2
    # dem[1][1] = dem[1][0]
    # dem[1][2] = (1-args.difference)/2
    # dem[1][3] = dem[1][2]
    #
    # dem[2][0] = (1-args.difference)/2
    # dem[2][1] = dem[2][0]
    # dem[2][2] = (1+args.difference)/2
    # dem[2][3] = dem[2][2]

    hyperE = hyperedges()
    CN = CacheNetwork(G, we, wv, dem, catalog, hyperE, capacity)
    CN.cvx_init()
    CN.solve()
    print("Status:", CN.problem.solution.status)
    print("Optimal Value:", CN.problem.solution.opt_val)
    # print('solve_time', CN.problem.solution.attr['solve_time'])
    # print('num_iters:', CN.problem.solution.attr['num_iters'])

    CN.test_feasibility(1e-2)
    x, z, mu, objE, objV = CN.solution()

    output = args.graph_type + '_' + args.outputfile
    with open(output, 'wb') as f:
        pickle.dump([x, z, objE, objV], f)


if __name__ == "__main__":
    main()

    # G = nx.DiGraph([(0, 1), (1, 2), (2, 3)])
    # we = {}
    # wv = {}
    # for e in G.edges():
    #     we[e] = square
    # for v in G.nodes():
    #     wv[v] = square
    #
    # dem = {}
    # targets = [0,2]
    # catalog = ['a','b','c','d']
    # for t in targets:
    #     dem[t] = {}
    #     for i in catalog:
    #         dem[t][i] = np.random.rand()
    #
    #
    # CN = CacheNetwork(G,we,wv,dem,catalog)
    # CN.cvx_init()
    # CN.solve()
    # print("Status:",CN.problem.solution.status)
    # print("Optimal Value:",CN.problem.solution.opt_val)
    # print('solve_time',CN.problem.solution.attr['solve_time'])
    # print('num_iters:',CN.problem.solution.attr['num_iters'])
    #
    # CN.test_feasibility(1e-2)
