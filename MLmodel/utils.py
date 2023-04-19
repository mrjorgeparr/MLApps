import networkx as nx
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import pairwise_distances
import pandas as pd
from sklearn.utils import Bunch



def reduce(data, n, asdf=True, sparse=False, centrality_measure='closeness'):

    """
    Inputs:
        - data: dataset comprised of vectors and their class label, sklearn.utils.Bunch format
        - n: number of top samples according to criteria 
        - asdf: flag for returning a dataframe
        - sparse: if the input is a sparse matrix regular PCA does not work
        - centrality measures (of node i):
            + degree: measures number of conexions connecting node i
            + closeness: 1/sum(d(i,j)) where j 
            + betweenneess: measures number of closes paths between two pairs of nodes go through i
    
    Outputs: 
        - top n sample according to criteria, format dependent upon flag
    """


    if sparse:
        svd = TruncatedSVD(n_components=2)
        X_reduced = svd.fit_transform(data.data)
    else:
        pca =  PCA(n_components=2)
        X_reduced = pca.fit_transform(data.data)

    distances = pairwise_distances(data.data)

    cutoff = np.mean(distances)
    G = nx.Graph()

    for i in range(len(X_reduced)):
        G.add_node(i)

    for i in range(len(X_reduced)):
        for j in range(i+1, len(X_reduced)):
            if distances[i, j] <= cutoff:
                G.add_edge(i, j, weight=distances[i,j])

    if centrality_measure == 'closeness':
        centrality = nx.closeness_centrality(G)
    elif centrality_measure == 'eigenvector':
        centrality = nx.eigenvector_centrality(G)
    elif centrality_measure == 'degree':
        centrality = nx.degree_centrality(G)
    elif centrality_measure == "betweenneess":
        centrality = nx.betweenness_centrality(G)
    else:
        raise ValueError('Centrality measure not supported.')

    topn = np.argsort(list(centrality.values()))[-n:]

    # create new dataset object with only top n examples
    if asdf:
        target = data.target[topn]
        data = data.data[topn]
        newdata = pd.DataFrame(data)
        newdata['target'] = target
        return newdata
    else:
        new_data = Bunch(
            data = data.data[topn],
            target = data.target[topn],
            target_names = data.target_names,
            feature_names = data.feature_names,
            DESCR = data.DESCR
        )
        return new_data