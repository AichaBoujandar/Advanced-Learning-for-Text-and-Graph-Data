import string
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

#import os
#os.chdir() # to change working directory to where functions live
# import custom functions
from library import clean_text_simple, terms_to_graph, core_dec

stpwds = stopwords.words('english')
punct = string.punctuation.replace('-', '')

my_doc = 'A method for solution of systems of linear algebraic equations \
with m-dimensional lambda matrices. A system of linear algebraic \
equations with m-dimensional lambda matrices is considered. \
The proposed method of searching for the solution of this system \
lies in reducing it to a numerical system of a special kind.'

my_doc = my_doc.replace('\n', '')

# pre-process document
my_tokens = clean_text_simple(my_doc,my_stopwords=stpwds,punct=punct)

g = terms_to_graph(my_tokens, 4)

# number of edges
print(len(g.es))


# the number of nodes should be equal to the number of unique terms
print(len(g.vs) == len(set(my_tokens)))

edge_weights = []
for edge in g.es:
    source = g.vs[edge.source]['name']
    target = g.vs[edge.target]['name']
    weight = edge['weight']
    edge_weights.append([source, target, weight])

print(edge_weights)
den = []
for w in range(2,10):
    g = terms_to_graph(my_tokens, w)
    ### fill the gap (print density of g) ###
    print("density of the graph for window size = ",w,' : ')
    print(g.density())
    den.append(g.density())
plt.plot([2,3,4,5,6,7,8,9],den,"o")
plt.xlabel('window size')
plt.ylabel('density of the graph')
# decompose g
core_numbers = core_dec(g,False)


### fill the gap (compare 'core_numbers' with the output of the .coreness() igraph method) ###
print("comparison between core_numbers and .coreness()")
print(list(core_numbers.values()) == g.coreness())


# retain main core as keywords
max_c_n = max(core_numbers.values())
keywords = [kwd for kwd, c_n in core_numbers.items() if c_n == max_c_n]
print(keywords)
layout = g.layout("kk")
pl = igraph.plot(g, layout = layout)
pl.show()