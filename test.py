from brainweb_utils.brainweb_loader import brainweb_images
from deepClustering.DFC import DFC
from sFCM.sFCM import FCM, sFCM, csFCM
from utils import Cluster
import brainweb, cv2
import numpy as np
import matplotlib.pyplot as plt

'''
s = brainweb_images()

s = brainweb_images()
s.fetch()
parameters = {'petNoise' : 1, 
                't1Noise' : 0.75, 
                't2Noise' : 0.75,
                'petSigma' : 1, 
                't1Sigma' : 1, 
                't2Sigma' : 1,
                'PetClass' : brainweb.FDG}

pet, uMap, T1, T2, ground_truth = s.sample_mMR_dataset(parameters, 1, 1337)


dfc = DFC()

idx = 60
im, label_probabilities = T1[0][idx], [ground_truth[0][key][idx,...] for key in ground_truth[0]]


#sample labels from probabilities
sample_labels = [np.random.binomial(1, x) for x in label_probabilities]


dfc.initialize_clustering(im)

for i in range(0, 100):
    im, labels, r_map, n_labels = dfc.step()

    if n_labels == 12:
        np.save('/home/kristmundur/Documents/KTH/Project Course/PCiDS/unsupervisedSegmentation/clustered_data/test.npy', {'im' : im, 'labels' : labels, 'label_probabilities' : label_probabilities}, allow_pickle=True)
    cv2.imshow('{}'.format(i), im)
    cv2.waitKey(10)
'''


test = np.load('/home/kristmundur/Documents/KTH/Project Course/PCiDS/deepClustering/clustered_data/test.npy', allow_pickle=True)

im, labels, label_probabilities = test[()]['im'], test[()]['labels'], test[()]['label_probabilities']

sample_labels = [np.random.binomial(1, x) for x in label_probabilities]

#adjust background

a, b = Cluster(labels), [Cluster(x, consider=[1]) for x in sample_labels]
#print([x.sum() for x in sample_labels])
#print(labels.shape, sample_labels[0].shape)
#mp, _ = b.approximate_mapping(a)
#print(mp)
#print(sum([x[0] for x in mp[1]]), sample_labels[1].sum()) 
Cluster.distribution(b, brainweb.Act.all_labels, a)
'''
for i in np.unique(labels):
    img = np.int8(labels == i)*255
    
    plt.imshow(img)
    #plt.show()

    plt.imshow(sample_labels[i]*255)
    #plt.show()
'''
'''
fig = plt.figure(figsize = (6, 10))
columns = 12
rows    = 1

for i, (lbl, name) in enumerate(zip(sample_labels, brainweb.Act.all_labels)):
    inferred, true = Cluster(labels), Cluster(lbl, consider = [1])

    print('Label name:', name)
    mp, A = true.approximate_mapping(inferred)

    association = max([mp[x] for x in mp], key = lambda y : y[0])

    print('associated {0} with cluster w. label {1} with IOU score of {2}'.format(name, association[1], association[0]))
    
    img = np.int8(labels == association[1])*255
    img = np.vstack((img, lbl*255))
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(name)

plt.show()
'''