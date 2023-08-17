# Codes for "Joint Optimization of Storage and Transmission via Coding Traffic Flows for Content Distribution"

Codes of algorithm implementations and experiments for paper:
>D. Malak, Y. Li, S. Ioannidis, E. Yeh and M. Médard. Joint Optimization of Storage and Transmission via Coding Traffic Flows for Content Distribution. International Symposium on Modeling and Optimization in Mobile, Ad hoc, and Wireless Networks (WiOpt), 2023.

Please cite this paper if you intend to use this code for your research. 

This work was supported in part by the National Science Foundation under Grants CNS 2008639 and CNS 2107062. Co-funded by the European Union (ERC, SENSIBILIT\'E, 101077361). Views and opinions expressed are however those of the authors only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them.

* [Dependencies](#dependencies)
* [Algorithms](#algorithms)
* [Plots](#plots)

## Dependencies
The dependencies are specified in the [``requirements.txt``](requirements.txt) file. 

## Algorithms
In our paper, we implemetnted 5 algorithm:
* Coded Caching (CC): ccaching.py --method cc
* Self-Coded Caching (SCC): ccaching.py --method ma
* Simulated’ Maddah-Ali and Niesen (SMAN): no_cross.py 
* Random Caching and Coded Transmission (RC-CT): Random.py --method cc
* Random Caching and SMAN Transmission (RC-SMANT): Random.py --method ma

through [``ccaching.py``](ccaching.py), [``no_cross.py``](no_cross.py), and [``Random.py``](Random.py). 
We also implement how to
1. generate the topology: --graph_type
2. their transmission cost of an edge and cache capacity of a node

in these three files. Some execution examples are shown in [``run_top``](run_top).

We also conduct sensitivity study over ''Tree'' Topology.
Some execution examples are shwon in [``run_tree``](run_tree).

## Plots
We plot Fig. 5 by [``plot_bar.py``](plot_bar.py),
Fig. 6 by [``plot_layer.py``](plot_layer.py),
Fig. 7 by [``plot_sens.py``](plot_sens.py).


