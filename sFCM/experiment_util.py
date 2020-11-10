from sFCM import *
from demo import *
import torch
use_cuda = torch.cuda.is_available()
def experiment_runs(model, true_labels,*args, **kwargs):
    """
    Runs the models run method on the data n times, calculates the average and standard
    deviation of the metric method.
    """
    metric_list = []
    for i in range(kwargs["n_trials"]):
        args
        for c, (stat, finish) in enumerate(model.run(*args,**kwargs)):
            #One could potentially save the images
            #or do some computation on the response to examine the convergence
            #during the runs
            continue
        metric_list.append(generate_metrics(stat,true_labels))
        if "save_trials" in kwargs:
            np.save("data_storage/image_"+str(kwargs["image_nr"])+"_"+type(model).__name__+"_trial_"+str(i),np.array(metric_list[-1]))


    stats = metric_stats(metric_list)
    if "save_stats" in kwargs:
        np.save("data_storage/image_"+str(kwargs["image_nr"])+"_"+type(model).__name__+"_trial_"+str(i),np.array(stats))

    return stats

def load_imgdir(dir_path,img_format):
    import glob
    import cv2

    images = [cv2.imread(file) for file in glob.glob(dir_path+"*."+img_format)]
    return images

def load_impathdir(dir_path,img_format):
    import glob
    import cv2

    images_paths = [file for file in glob.glob(dir_path+"*."+img_format)]
    return images_paths


def load_run(model,dir_path,img_format, true_labels,*args, paths=False, **kwargs):
    """args are used as the input parameters to the model run method 
        kwargs are used deterministically with "n_trials" required
    """
    if paths:
        images = load_impathdir(dir_path,img_format)
    else:
        images = load_imgdir(dir_path,img_format)
    metrics = []
    for i,im in enumerate(images):
        if "save_trials" in kwargs or "save_stats" in kwargs:
            metrics.append(experiment_runs(model,true_labels, im,*args, image_nr=i,**kwargs))
        else:
            metrics.append(experiment_runs(model,true_labels, im,*args,**kwargs))
    return metrics

def generate_metrics(stat,true_labels):
    """
    Generates metrics for clustering based on stat containing
    membership function and data labels, in that order.
    """
    partition_c = partition_coefficient(stat[0])
    partition_e = partition_entropy(stat[0])
    #output_cluster = Cluster(stat[1])
    #true_cluster = Cluster(true_labels)
    #IOU = output_cluster.iou(true_cluster)
    IOU=np.array([None])
    return partition_c,partition_e,IOU

def partition_coefficient(u):
    return 0

def partition_entropy(u):
    return 0

def metric_stats(metrics):
    """
    Computes the average and variance of the different metrics
    if not array or lists, also add the original metric list
    for histogram purposes
    """
    stats = []
    for m in metrics:
        if type(m)==list or type(m)==np.ndarray:
            stats.append([None,None,m])
        else:
            stats.append([np.average(m),np.var(m),m])
    return stats


if __name__=="__main__":
    images = load_imgdir('Project-Course-in-Data-Science/sFCM/Experiments_data/',"jpeg")
    #fcm = sFCM(2, 5, 1, 0.5, 3, images[0].shape)
    fcm = FCM(2, 10, images[0].shape)
    #load_run(fcm,metric_0,'Project-Course-in-Data-Science/sFCM/gifs/',"jpeg",0,15, n_trials=30)

    #IMPORTANT: When using the NN method we need to use paths to the run method
    # train
    data = torch.from_numpy( np.array([images[0].transpose( (2, 0, 1) ).astype('float32')/255.]) )
    if use_cuda:
        data = data.cuda()
    data = Variable(data)
    model = MyNet( data.size(1),2,100 )
    if use_cuda:
        model.cuda()
    model.train()
    load_run(fcm,'Project-Course-in-Data-Science/sFCM/Experiments_data/',"jpeg",[None], 0, n_iter=1,paths=False, n_trials=1) 