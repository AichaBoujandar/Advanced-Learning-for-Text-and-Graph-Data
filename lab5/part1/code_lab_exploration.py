"""
Graph Mining - ALTEGRAD - Dec 2019
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1

G = nx.read_edgelist("C:/Users/Aicha BOUJANDAR/Desktop/3A/ALTEGRAD/TP5/code/datasets/CA-HepTh.txt",comments = '#', delimiter='\t', create_using=nx.Graph())
print("Nodes:",G.number_of_nodes())
print("Edges:",G.number_of_edges())


############## Task 2

print("Number of connected components:",nx.number_connected_components(G))
largest_cc = max(nx.connected_components(G), key=len)
L_cc = G.subgraph(largest_cc)
print("Fraction of nodes of the largest connected components:",L_cc.number_of_nodes()/G.number_of_nodes())
print("Fraction of edges of the largest connected components:",L_cc.number_of_edges()/G.number_of_edges())



############## Task 3
# Degree
degree_sequence = [G.degree(node) for node in G.nodes()]
deg_seq_numpy = np.array(degree_sequence)
print("minimum of nodes degrees:", deg_seq_numpy.min())
print("maximum of nodes degrees:", deg_seq_numpy.max())
print("mean of nodes degrees:", deg_seq_numpy.mean())






############## Task 4

hist = nx.degree_histogram(G)
plt.plot(hist)
plt.title("Degree distribution")
plt.xlabel("Degree")
plt.ylabel("Frequency")
#plt.xscale('log')
#plt.yscale('log')
plt.show()