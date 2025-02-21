import re 
import itertools
import operator
import copy
import igraph
import heapq
import nltk
import numpy as np
nltk.download('maxent_treebank_pos_tagger')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
# requires nltk 3.2.1
from nltk import pos_tag # nltk.download('maxent_treebank_pos_tagger')

def clean_text_simple(text, my_stopwords, punct, remove_stopwords=True, pos_filtering=True, stemming=True):
    text = text.lower()
    text = ''.join(l for l in text if l not in punct) # remove punctuation (preserving intra-word dashes)
    text = re.sub(' +',' ',text) # strip extra white space
    text = text.strip() # strip leading and trailing white space
    tokens = text.split(' ') # tokenize (split based on whitespace)
    if pos_filtering == True:
        # POS-tag and retain only nouns and adjectives
        tagged_tokens = pos_tag(tokens)
        tokens_keep = []
        for item in tagged_tokens:
            if (
            item[1] == 'NN' or
            item[1] == 'NNS' or
            item[1] == 'NNP' or
            item[1] == 'NNPS' or
            item[1] == 'JJ' or
            item[1] == 'JJS' or
            item[1] == 'JJR'
            ):
                tokens_keep.append(item[0])
        tokens = tokens_keep
    if remove_stopwords:
        tokens = [token for token in tokens if token not in my_stopwords]
    if stemming:
        stemmer = nltk.stem.PorterStemmer()
        tokens_stemmed = list()
        for token in tokens:
            tokens_stemmed.append(stemmer.stem(token))
        tokens = tokens_stemmed
    
    return(tokens)


def terms_to_graph(terms, window_size):
    '''This function returns a directed, weighted igraph from lists of list of terms (the tokens from the pre-processed text)
    e.g., ['quick','brown','fox']
    Edges are weighted based on term co-occurence within a sliding window of fixed size 'w'
    '''
    
    from_to = {}

    w = min(window_size, len(terms))
    # create initial complete graph (first w terms)
    terms_temp = terms[0:w]
    indexes = list(itertools.combinations(range(w), r=2))

    new_edges = []

    for my_tuple in indexes:
        new_edges.append(tuple([terms_temp[i] for i in my_tuple]))
    for new_edge in new_edges:
        if new_edge in from_to:
            from_to[new_edge] += 1
        else:
            from_to[new_edge] = 1

    # then iterate over the remaining terms
    for i in range(w, len(terms)):
        # term to consider
        considered_term = terms[i]
        # all terms within sliding window
        terms_temp = terms[(i - w + 1):(i + 1)]

        # edges to try
        candidate_edges = []
        for p in range(w - 1):
            candidate_edges.append((terms_temp[p], considered_term))

        for try_edge in candidate_edges:

            # if not self-edge
            if try_edge[1] != try_edge[0]:

                # if edge has already been seen, update its weight
                if try_edge in from_to:
                    from_to[try_edge] += 1

                # if edge has never been seen, create it and assign it a unit weight
                else:
                    from_to[try_edge] = 1

    # create empty graph
    g = igraph.Graph(directed=True)

    # add vertices
    g.add_vertices(sorted(set(terms)))

    # add edges, direction is preserved since the graph is directed
    g.add_edges(list(from_to.keys()))

    # set edge and vertice weights
    g.es['weight'] = list(from_to.values()) # based on co-occurence within sliding window
    g.vs['weight'] = g.strength(weights=list(from_to.values())) # weighted degree

    return (g)


def core_dec(g,weighted):
    '''(un)weighted k-core decomposition'''
    # work on clone of g to preserve g 
    gg = copy.deepcopy(g)
    if not weighted:
        gg.vs['weight'] = gg.strength() # overwrite the 'weight' vertex attribute with the unweighted degrees
    # initialize dictionary that will contain the core numbers
    cores_g = dict(zip(gg.vs['name'],[0]*len(gg.vs)))
    
    while len(gg.vs) > 0:
        # find index of lowest degree vertex
        min_degree = min(gg.vs['weight'])
        index_top = gg.vs['weight'].index(min_degree)
        name_top = gg.vs[index_top]['name'] #the name of the vertix that has the lowest weight
        # get names of its neighbors
        neighbors = gg.vs[gg.neighbors(index_top)]['name']
        # exclude self-edges
        neighbors = [elt for elt in neighbors if elt!=name_top]
        # set core number of lowest degree vertex as its degree
        cores_g[name_top] = min_degree
        ### fill the gap (delete top vertex and its incident edges) ###
        gg.delete_vertices(gg.vs[index_top,name_top]) #
        new_degrees = copy.deepcopy(gg.vs['weight'])
        
        
        
        if neighbors:
            if weighted: 
                ### fill the gap (compute the new weighted degrees, save results as 'new_degrees')
                
                for neigh in neighbors :
                    index_n = gg.vs['name'].index(neigh) #l'ordre du neighbour dans le graphe principal
                    new_degrees[index_n] = gg.strength(neigh, weights=gg.es['weight'])
                    
            else:
                ### fill the gap (same as above but for the basic degree) ###
            # iterate over neighbors of top element
                for neigh in neighbors :
                    index_n = gg.vs['name'].index(neigh)
                    new_degrees[index_n] = gg.strength(neigh)
                    
            
            for neigh in neighbors:
                index_n = gg.vs['name'].index(neigh)
                gg.vs[index_n]['weight'] = max(min_degree,new_degrees[index_n]) 
                
        
    return(cores_g)


def accuracy_metrics(candidate, truth):
    # true positives ('hits') are both in candidate and in truth
    tp = len(set(candidate).intersection(truth))
    
    # false positives a.k.a. false alarms are in candidate but not in truth
    fp = len([element for element in candidate if element not in truth])
    
    ### fill the gap (compute false negatives a.k.a. misses, save results as 'fn')
    fn = len([element for element in truth if element not in candidate])
    
    ### fill the gaps (compute precision and recall as a function of 'tp', 'fp' and 'fn', save results as 'prec' and 'rec') ###
    prec = tp/(tp+fp)
    rec = tp/(tp+fn)
    if prec+rec != 0:
        f1 = 2*prec*rec/(prec+rec)
    else:
        f1 = 0
    
    return (prec, rec, f1)
