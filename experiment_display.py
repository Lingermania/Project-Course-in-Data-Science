from experiment_util import *
from utils import visualize_matrix
import numpy as np
import seaborn as sns
import pandas as pd

def visualize_labeled_cm(cm,labels):
    truth_labels = [label for label in labels[:len(cm)]]
    cluster_labels = [[] for i in range(len(cm[0]))]
    for i,label in enumerate(truth_labels):
        win_ind = np.argmax(cm[i])
        cluster_labels[win_ind].append(label)
    cluster_labels = ["\n".join(cluster) for cluster in cluster_labels]
    visualize_matrix(cm, yticklabels=truth_labels,xticklabels=cluster_labels)

def compute_mIOU(IOU_maps):
    mIOU_imlist = []
    for im in IOU_maps:
        mIOU_list=[]
        for trial in im:
            mIOU = 0 
            for key in trial:
                mIOU+=trial[key][0]/len(trial)
            mIOU_list.append(mIOU)
        mIOU_imlist.append(mIOU_list)
    return np.array(mIOU_imlist)

def best_worst_mIOU_ind(miou_list):
    best_ind = np.unravel_index(np.argmax(mIOU_imlist), mIOU_imlist.shape)
    worst_ind = np.unravel_index(np.argmin(mIOU_imlist), mIOU_imlist.shape)
    return best_ind,worst_ind


sns.set(font_scale=0.8)

d = load_experiments_data(data_storage_path_compiled,"npy")[0]
metric_index ={metric:i for i,metric in enumerate(["partition_c","partition_e","iox_matrix", "iox_cm", "iou_matrix", "iou_cm", "iou_mapping"])}
labels_dict = load_experiments_data(experiments_storage_path_label_prob, "npy",item=True)
truth_labels = [key for key in labels_dict[0]]
iou_cm = np.array(d[metric_index["iou_cm"]])
iox_cm = np.array(d[metric_index["iox_cm"]])
mIOU_imlist=compute_mIOU(d[metric_index["iou_mapping"]])

#Histogram of mIOU
#sns.displot(pd.Series(mIOU_imlist.flatten(), name="mIOU"))
#plt.show()

best_ind,worst_ind = best_worst_mIOU_ind(mIOU_imlist)

visualize_matrix(iox_cm[best_ind], title="Best iox confusion matrix")
visualize_matrix(iou_cm[best_ind], title="Best iou confusion matrix")
visualize_matrix(iox_cm[worst_ind], title="Worst iox confusion matrix")
visualize_matrix(iou_cm[worst_ind], title="Worst iou confusion matrix")

print()