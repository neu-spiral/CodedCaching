import networkx as nx
import cvxpy as cp
import numpy as np
import logging

from cvxpy.atoms.elementwise.minimum import minimum
from cvxpy.atoms.elementwise.square import square

#questions: 
# Do we need upper bound of 1 on any variables?
# Can we incorporate capacity constraints?
# How to x_{v(i,j)} contribute to z/capacity constraints/costs etc.?

def minhalf(a,b):
    """ Return min(a,b) + 0.5*[a-b]_+ = 0.5 *(a + min(a,b)) """
    return 0.5*(a+minimum(a,b))


class CacheNetwork:
    """ A class modeling a cache network. """
    
    def __init__(self,G,we,wv,dem,cat,c={}):
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
   
    def is_target(self,t,i):
        ''' Detect whether node t is a target for item i.'''
        return t in self.demand and i in self.demand[t]

    
    def cvxInit(self):
        """Constuct cvxpy problem instance"""
        


        self.vars = {} # used to access variables outside cvx program

        # cache variables
        logging.debug("Creating cache variables...")
        x = {}
        for v in self.G.nodes():
            x[v] = {}
            for i in self.catalog:
                x[v][i] = cp.Variable()
                for j in self.catalog:
                    if j>i:
                        x[v][(i,j)] = cp.Variable()

        self.vars['x'] = x

        # xi book-keeping variables
        logging.debug("Creating xi book-keeping variables...")
        xi = {}
        for t in self.demand:
            xi[t] = {}
            for v in self.G.nodes():
                xi[t][v] = {}
                for i in self.catalog:  #not self.demand[t], because of cross coding you may want to use traffic not demanded by t
                    xi[t][v][i] = cp.Variable()
                    for j in self.catalog:
                        if j>i:
                            xi[t][v][(i,j)] = cp.Variable()
        
        self.vars['xi'] = xi
        
        # z flow variables
        logging.debug("Creating z flow variables...")
        z = {}
        for e in self.G.edges():
            z[e] = {}
            for i in self.catalog:
                z[e][i] = cp.Variable()
                for j in self.catalog:
                    if j>i:
                        z[e][(i,j)] = cp.Variable()

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
                    for j in self.catalog:
                        if j>i:
                            rho[t][e][(i,j)] = cp.Variable()

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
        for v in self.wv:
            xv = 0
            for ii in x[v]:
                xv += x[v][ii]
            obj += self.wv[v](xv) 

        # constraints
        logging.debug("Creating costraints...")
        constr = []
        logging.debug("Creating cache variable non-negativity costraints...")
        for v in x:
            for ii in x[v]:
                constr.append(x[v][ii] >= 0)
                # constr.append(x[v][ii] <= 1) no upper bounds on x
        logging.debug("Creating xi variable non-negativity costraints...")
        for t in xi:
            for v in xi[t]:
                for ii in xi[t][v][ii]:
                    constr.append(xi[t][v][ii] >= 0)
        logging.debug("Creating rho variable non-negativity costraints...")
        for t in rho:
            for e in rho[t]:
                for ii in rho[t][e][ii]:
                    constr.append(rho[t][e][ii] >= 0)
        logging.debug("Creating z variable non-negativity costraints...")
        for e in z:
            for ii in z[e][ii]:
                constr.append(z[e][ii]>=0)
        logging.debug("Creating mu variable non-negativity costraints...")
        for t in mu:
            for v in mu[t]:
                for ii in mu[t][v][ii]:
                    constr.append(mu[t][v][ii] >= 0)
 


        # book-keeping constraints
        logging.debug("Creating z book-keeping constraints...")
        for t in rho:
            for e in rho[t]:
                for ii in rho[t][e]:
                    constr.append( rho[t][e][ii] <= z[e][ii] )

        logging.debug("Creating x book-keeping constraints...")
        for t in xi:
            for v in xi[t]:
                for ii in xi[t][v]:
                    constr.append( xi[t][v][ii] <= x[v][ii] )



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
                        in_flow[t][v][i] +=  rho[t][e][i]
                    for j in self.catalog:
                        if j>i:
                            in_flow[t][v][(i,j)] = xi[t][v][(i,j)]
                            for e in in_edges:
                                in_flow[t][v][(i,j)] += rho[t][e][(i,j)]

                    out_flow[t][v][i] = 0
                    for e in out_edges:
                        out_flow[t][v][i] += rho[t][e][i]
             
                    for j in self.catalog:
                        if j>i:
                            out_flow[t][v][(i,j)] = 0
                            for e in out_edges:
                                out_flow[t][v][(i,j)] += rho[t][e][(i,j)]
                            

                                
                        
        logging.debug("Creating decodability constraints...")
        for t in self.targets:
            for v in self.G.nodes():
                for i in self.catalog:
                    dec_rate = in_flow[t][v][i]
                    for j in self.catalog:
                        if j<i:
                            dec_rate += minimum( in_flow[t][v][(j,i)], mu[t][v][j] )
                        if j>i:
                            dec_rate += minhalf( in_flow[t][v][(i,j)], in_flow[t][v][j])
                    constr.append( dec_rate >= mu[t][v][i] )      

        # Outgoing flow is bounded by decodability
        
        logging.debug("Creating unitary outgoing flow constraints...")
        for t in self.targets:
            for v in self.G.nodes():
                for i in self.catalog:
                    constr.append( mu[t][v][i] >= out_flow[t][v][i])


        logging.debug("Creating pair outgoing flow constraints...")
        for t in self.targets:
            for v in self.G.nodes():
                for i in self.catalog:
                    for j in self.catalog:
                        if j>i:
                            constr.append( 2*minimum(mu[t][v][i],mu[t][v][j]) >= out_flow[t][v][(i,j)])

        # Demand should be met

        logging.debug("Creating demand constraints...")
        for t in self.targets:
            for i in self.demand[t]: #demand met should be restricted to i's in demand[t]
                constr.append( mu[t][t][i] >= self.demand[t][i] )

       
             
        # Capacity constraints (optional)
        logging.debug("Creating cache variable capacity costraints...")
        for v in self.c:
            xv = 0
            for ii in x[v]:
                xv += x[v][ii]
            constr.append(xv <= c[v])    

        self.problem = cp.Problem(cp.Minimize(obj),constr)              
        logging.debug("Problem is DCP: "+str(self.problem.is_dcp()))

    def solve(self):
        logging.info("Initializing problem parameters...")
        self.cvxInit()
        logging.info("Running cvxpy solver...")
        return self.problem.solve()
    

if __name__=="__main__":
    logging.basicConfig(level=logging.DEBUG)
    G = nx.DiGraph([(0, 1), (1, 2), (2, 3)])
    we = {}
    wv = {}
    for e in G.edges():
        we[e] = square
    for v in G.nodes():
        wv[v] = square

    dem = {}
    targets = [0,2]
    catalog = ['a','b','c','d']
    for t in targets:
        dem[t] = {}
        for i in catalog:
            dem[t][i] = np.random.rand()


    CN = CacheNetwork(G,we,wv,dem,catalog)
    CN.cvxInit()
    results = CN.solve()
