import networkx as nx
import cvxpy as cp
import numpy as np
import logging

from cvxpy.atoms.elementwise.minimum import minimum 

#questions: 
# Do we need upper bound of 1 on any variables?
# Can we incorporate capacity constraints?
# How to x_{v(i,j)} contribute to z/capacity constraints/costs etc.?

def minhalf(a,b):
    """ Return min(a,b) + 0.5*[a-b]_+ = 0.5 *(a + min(a,b)) """
    return 0.5*(a+minimum(a,b))


class CacheNetwork:
    """ A class modeling a cache network. """
    
    def __init__(self,G,we,wv,dem,C,c={}):
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
   
    def is_target(self,t,i):
        ''' Detect whether node t is a target for item i.'''
        return t in self.demand and i in self.demand[t]

    def cvxInit(self):
        """Constuct cvxpy problem instance"""
        



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

        # xi book-keeping variables
        logging.debug("Creating xi book-keeping variables...")
        xi = {}
        for t in self.demand:
            xi[t] = {}
            for v in self.G.nodes():
                xi[t][v] = {}
                for i in self.catalog:
                    xi[t][v][i] = cp.Variable()
                    for j in self.catalog:
                        if j>i:
                            xi[t][v][(i,j)] = cp.Variable()

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

        
        # mu decoding capability variables
        logging.debug("Creating mu decoding capability variables...")
        mu = {}
        for t in self.demand:
            mu[t] = {}
            for v in self.G.nodes():
                mu[t][v] = {}
                for i in self.catalog:
                    mu[t][v][i] = cp.Variable()



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
        logging.debug("Creating z book-keeping constraints")
        for t in rho:
            for e in rho[t]:
                for ii in rho[t][e]:
                    constr.append( rho[t][e][ii] <= z[e][ii] )

        logging.debug("Creating x book-keeping constraints")
        for t in xi:
            for v in xi[t]:
                for ii in xi[t][v]:
                    constr.append( xi[t][e][ii] <= x[v][ii] )



        # flow constraints
        logging.debug("Creating rho flow constraints")

        # Decodability rate constraints

        logging.debug("Creating in flows")

        in_flow={}
        for t in self.targets:
            in_flow[t] = {}

            for v in self.G.nodes():
                in_edges = self.G.in_edges(v)
                in_flow[t][v]={}

                for i in self.catalog:
                    in_flow[t][v][i] = xi[t][v][i] 
                    for e in in_edges:
                        in_flow[t][v][i] +=  rho[t][e][i]
                    for j in self.catalog:
                        if j>i:
                            in_flow[t][v][(i,j)] = xi[t][v][(i,j)]
                            for e in in_edges:
                                in_flow[t][v][(i,j)] += rho[t][v][(i,j)]
                        

        in_sum = {}
        out_sum = {}
        for v in self.G.nodes():
            in_edges = self.G.in_edges(v)
            out_edges = self.G.out_edges(v)
            
            in_sum[v] = {}
            out_sum[v] = {}

            for t in self.targets:
                in_sum[v][t] = {}
                out_sum[v][t] = {}

                for i in self.catalog:

                    in_sum[v][t][i] = 0
                    for e in in_edges:
                        in_sum[v][t][i] += rho[t][e][i]
                    in_sum[v][t][i] += x[v][i] 

                    out_sum[v][t][i] = 0
                    for e in out_edges:
                        out_sum[v][t][i] += rho[t][e][i]

                    constr.append( in_sum[v][t][i]  >= out_sum[v][t][i])
  
        # Pair-flow
        for v in self.G.nodes():
            in_edges = self.G.in_edges(v)
            out_edges = self.G.out_edges(v)
             
            for t in self.targets:
                for i in self.catalog:
                    for j in self.catalog:
                        if j>i:
                            in_sum[v][t][(i,j)] = 0
                            for e in in_edges:
                                in_sum[v][t][(i,j)] += rho[t][e][(i,j)]
                            in_sum[v][t][(i,j)] += x[v][(i,j)]

                            
                            out_sum[v][t][(i,j)] = 0
                            for e in out_edges:
                                out_sum[v][t][(i,j)] += rho[t][e][(i,j)]
                            
                            constr.append(minimum(in_sum[v][t][i], in_sum[v][t][j])+ in_sum[v][t][(i,j)] >=  out_sum[v][t][(i,j)]  )
                        
        # Meet target demand
        for t in self.targets:
            for  i in self.demand[t]:
                tot = in_sum[t][t]
                for j in self.catalog:
                    if i==j:
                        continue
                    elif i<j:
                        ii = (i,j)
                    else:
                        ii = (j,i)
                    
                    tot += minhalf(in_sum[t][t][ii],in_sum[t][t][j])

                constr.append(tot >= self.demand[t][i])

   
        # Capacity constraints (optional)
        logging.debug("Creating cache variable capacity costraints...")
        for v in self.c:
            xv = 0
            for ii in x[v]:
                xv += x[v][ii]
            constr.append(x[v][i] <= c[v])    

        self.problem = cp.Problem(cp.Minimize(obj),constr)              
        logging.debug("Problem is DCP: "+self.problem.is_dcp())

    def solve(self):
        logger.info("Initializing problem parameters...")
        self.cvxInit()
        logging.info("Running cvxpy solver...")
        return self.problem.solve()
