import numpy as np
import functools
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def partition_coefficient(u):
    '''
    Calculates the partition coefficient of membership function

    u is a membership function over N pixels and c classes 
    '''

    N, c = u.shape

    return (u**2).sum()/N

def partition_entropy(u):
    '''
    Calculates the partition entropy of membership function

    u is a membership function over N pixels and c classes
    '''

    N, c = u.shape

    return -(u * np.ma.log(u).filled(0)).sum()/N

def visualize_matrix(mat, title=None, save=False, fname=None,**kwargs):
    plt.figure(figsize = (13, 10))

    sn.heatmap(mat, annot=True,**kwargs)
    if title:
        plt.title(title)
    if save:
        plt.savefig(fname)
    else:
        plt.show()

class Cluster:
    def __init__(self, labels, consider = None):
        self.labels = labels
        self.consider = consider
        self.total_size = np.prod(self.labels.shape)
        self.__initialize_partition_sets()

    def __initialize_partition_sets(self):
        self.partition = {x : set() for x in np.unique(self.labels)}

        for i, r in enumerate(self.labels):
            for j, v in enumerate(r):
                if self.consider != None and v in self.consider:
                    self.partition[v].add((i,j))
                elif self.consider == None:
                    self.partition[v].add((i,j))

    def approximate_mapping(self, other, metric = lambda x, y : len(x & y)/len(x | y)):
        '''
        Approximate mapping between self and other clusters using maximum IOU

        Returns a 2-tuple consisting of 
            a) The most likely mapping according to IOU criteria, i.e., m(l) : self.labels -> (B subset of other.labels) * [0, 1] (where * is the cartesian product)
            b) A subset 'A' from other.labels that satisfies: x in A iff (x not in B and x in other.labels) (complement of B)
        '''
        mp  = {x : [] for x in np.unique(self.labels)}
        #iou = lambda x, y : len(x & y)/len(x | y)

        #Create a mapping
        for p1 in self.partition:
            for p2 in other.partition:
                mp[p1].append((metric(self.partition[p1], other.partition[p2]), p2))


        A = set([x for x in np.unique(other.labels)]) - set([mp[x][1] for x in mp]) #other.labels - B

        return mp, A

    
    @staticmethod
    def iou_mapping(clusters_from, clusters_from_names, cluster_to):
        assert len(clusters_from) == len(clusters_from_names)
        
        res = {}
        for a, name in zip(clusters_from, clusters_from_names):
            mp, A = a.approximate_mapping(cluster_to)

            if 1 in mp:
                association = max([x for x in mp[1]], key = lambda y : y[0])

                res[name] = association

        return res

    @staticmethod
    def distribution(clusters_from, clusters_from_names, cluster_to, metric = 'iox'):
        assert len(clusters_from) == len(clusters_from_names)

        res = {}
        arr = []

        ioumap = Cluster.iou_mapping(clusters_from, clusters_from_names, cluster_to)
        invioumap = defaultdict(list)
        for key in ioumap:
            invioumap[ioumap[key][1]].append(key)
        #invioumap = {ioumap[key][1] : key for key in ioumap}


        names = [key for key in ioumap]
        idx   = np.array([True]*len(clusters_from_names))
        for i in range(len(idx)):
            if clusters_from_names[i] not in names:
                idx[i] = False

        clusters_from = [clusters_from[i] for i in range(len(clusters_from)) if idx[i]]
        clusters_from_names = [clusters_from_names[i] for i in range(len(clusters_from_names)) if idx[i]]

        for a, name in zip(clusters_from, clusters_from_names):
            if metric == 'iou':
                mp, A = a.approximate_mapping(cluster_to, metric = lambda x, y : len(x & y)/len(x | y) if len(y) > 0 else 0)
            elif metric == 'iox':
                mp, A = a.approximate_mapping(cluster_to, metric = lambda x, y : len(x & y)/len(x) if len(x) > 0 else 0)
            
            if 1 in mp:
                arr.append([mp[1][x][0] for x in range(len(mp[1]))])

        s = ['{0} ({1})'.format(x, invioumap[x]) for x in np.unique(cluster_to)]
        df_cm = pd.DataFrame(arr, index = clusters_from_names, columns = ['{0} ({1})'.format(x, invioumap[x]) for x in np.unique(cluster_to.labels)])

        return arr, df_cm
        


        