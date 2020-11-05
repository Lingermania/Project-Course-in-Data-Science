import numpy as np


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

    return -(u * np.log(u)).sum()/N

class Cluster:
    def __init__(self, labels):
        self.labels = labels


        self.__initialize_partition_sets()

    def __initialize_partition_sets(self):
        self.partition = {x : set() for x in np.unique(self.labels)}

        for i, r in enumerate(self.labels):
            for j, v in enumerate(r):
                self.partition[v].add((i,j))

    def approximate_mapping(self, other):
        '''
        Approximate mapping between self and other clusters using maximum IOU

        Returns a 2-tuple consisting of 
            a) The most likely mapping according to IOU criteria, i.e., m(l) : self.labels -> (B subset of other.labels) * [0, 1] (where * is the cartesian product)
            b) A subset 'A' from other.labels that satisfies: x in A iff (x not in B and x in other.labels) (complement of B)
        '''
        mp  = {x : (0, 0) for x in np.unique(self.labels)}
        iou = lambda x, y : len(x & y)/len(x | y)

        #Create a mapping
        for p1 in self.partition:
            for p2 in other.partition:
                mp[p1] = max(mp[p1], (iou(self.partition[p1], other.partition[p2]), p2), key = lambda x : x[0])


        A = set([x for x in np.unique(other.labels)]) - set([x[1] for x in mp]) #other.labels - B

        return mp, A

    


        