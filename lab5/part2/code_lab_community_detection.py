"""
Graph Mining - ALTEGRAD - Dec 2019
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from random import randint
from sklearn.cluster import KMeans


############## Task 5
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    
    n = G.number_of_nodes()
    A = nx.adjacency_matrix(G)
    D = np.zeros((n,n))
    for i,node in enumerate(G.nodes()) :
      D[i,i] = G.degree(node)
    L = D-A
    eigvals, eigvecs = eigs(L,k=100,which='SR')
    #eigvals, eigvecs = eigs(L,k=nx.number_connected_components(G),which='SR')
    eigvecs = eigvecs.real

    km = KMeans(n_clusters=k)
    km.fit(eigvecs)

    clustering = dict()
    for i, node in enumerate(G.nodes()):
      clustering[node] = km.labels_[i]

    
    return clustering
    




############## Task 6

G = nx.read_edgelist("C:/Users/Aicha BOUJANDAR/Desktop/3A/ALTEGRAD/TP5/code/datasets/CA-HepTh.txt",comments = '#', delimiter='\t', create_using=nx.Graph())
largest_cc = max(nx.connected_components(G), key=len)
L_cc = G.subgraph(largest_cc)



############## Task 7
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    
    modularity = 0
    clusters = set(clustering.values())
    m = G.number_of_edges()
    for cluster in clusters :
      nodes_in_cluster = [node for node in G.nodes() if clustering[node] == cluster]
      subG = G.subgraph(nodes_in_cluster)
      l_c = subG.number_of_edges()
      d_c = 0
      for node in nodes_in_cluster :
        d_c+=G.degree(node)
      modularity += (l_c/m) - (d_c/(2*m))**2
    
    return modularity
    




############## Task 8

print("Modularity spectral clustering:",modularity(L_cc,spectral_clustering(L_cc,k=50)))
random_clustering = dict()
for node in G.nodes():
  random_clustering[node] = randint(0,49)
print("Modularity random clustering", modularity(L_cc,random_clustering))