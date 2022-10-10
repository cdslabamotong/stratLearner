# stratLearner
Code for StratLearner.

StratLearner: Learning a Strategy for Misinformation Prevention in Social Network, NeurIPS 2020

Please download data at http://udel.edu/~amotong/dataset/stratLearner/

The folder "data" contains three datasets: pro, power768 and ER512.

Each dataset contains the following files:

	"data_diffusionModel": each lines shows one edge: "node1 node2 alpha beta", where alpha and beta are the parameter of the Weibull distribution.

	"data_pair_2500": each four lines denotes one pair of attacker-protector:
		Line1: indexes of the nodes in attacker
		Line2: indexes of the nodes in protector
		Line3: the pre computed influence of attacker
		Line4: the pre computed influence of the influence when proctor is added
		*line3 and line4 are not necessary, but can speed up testing if used.

	folder "feature": contains the subgraphs used for generating features, the four distributions are:
		"uniform_structure0-005_2000": phi_0.005^1
		"uniform_structure0-01_2000": phi_0.01^1
		"uniform_structure1-0_1": phi_1^1
		"WC_Weibull_structure_800": phi_+^+
		Each subgraph is associated with two files: 
			"index_graph.txt" shows the graph structure: each line is one edge: "node1 node2"
			"index_distance.txt" show the distance matrix between each pair of nodes.
		


The three folders "StratLearner" "dspn" and "gcn" contain the codes used for experiments. 

Run train.py to reproduce the experiments. Change the default parameters to switch datasets, change training size, type of features, and other learning settings.




Except "cvxopt" for quadratic programming, other require packages such as numpy and torch are common.

cvxopt can be installed via pip. (Conda seems to do not have cvxopt)


Thank you.
