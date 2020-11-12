from sFCM.sFCM import *
from sFCM.demo import *
from utils import partition_coefficient,partition_entropy
import torch
use_cuda = torch.cuda.is_available()
def experiment_runs(model, true_labels,*args, **kwargs):
    """
    Runs the models run method on the data n times, calculates the average and standard
    deviation of the metric method.
    """
    metric_list = []
    for i in range(kwargs["n_trials"]):
        print("Trial {} \n----------------------------------".format(i+1))
        for c, (stat, finish) in enumerate(model.run(*args,**kwargs)):
            print("Iteration {}".format(c))
            #One could potentially save the images
            #or do some computation on the response to examine the convergence
            #during the runs
            continue
        metric_list.append(generate_metrics(stat,true_labels))
        if "save_trials" in kwargs:
            np.save(data_storage_path+str(kwargs["image_nr"])+"_"+type(model).__name__+"_trial"+str(i),np.array(metric_list[-1],dtype=object))


    stats = metric_stats(metric_list)
    if "save_stats" in kwargs:
        np.save(data_storage_path+str(kwargs["image_nr"])+"_"+type(model).__name__+"total"+str(i),np.array(stats,dtype=object))

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
    experiment_para_str = "Experiment parameters\n"+"".join([key+":"+str(kwargs[key])+"\n" for key in kwargs])+"----------------------------------"
    print("Begining experiment with {} model \n----------------------------------".format(type(model).__name__))
    print(experiment_para_str)
    for i,im in enumerate(images):
        print("Starting experiment trials with image {} \n----------------------------------".format(i))
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

    return 0

def metric_stats(metrics):
    """
    Computes the average and variance of the different metrics
    if not array or lists, also add the original metric list
    for histogram purposes
    """
    metric_array = np.array(metrics)

    stats = []
    
    for m in range(metric_array.shape[1]):
        if type(metric_array[0,m])==list or type(metric_array[0,m])==np.ndarray:
            stats.append([metric_array[:,m]])
        else:
            stats.append([np.average(metric_array[:,m].astype(float)),np.var(metric_array[:,m].astype(float)),metric_array[:,m]])
    return stats

def combine_im_metrics(im_metric_list):
    """
    Concatenates the metrics from the different images 
    and then computes the metrics on the whole image
    set. Based on the assumption that the third value
    in the list added by metric_stats is the original
    metric list
    """
    stats=[]
    imset_metrics = [[] for i in range(len(im_metric_list[0]))]
    for im in im_metric_list:
        for i,metric_list in enumerate(im):
            if type(metric_list[0])==list or type(metric_list[0])==np.ndarray:
                for metric_set in metric_list:
                    imset_metrics[i].append(metric_set)
            else:
                for metrics in metric_list[2]:
                    imset_metrics[i].append(metrics)
    for metric_list in imset_metrics:
        if type(metric_list[0])==np.ndarray:
            stats.append(metric_list)
        else:
            stats.append([np.average(metric_list),np.var(metric_list),metric_list])
    return stats

def load_experiments_data(dir_path, data_format):
    """
    Creates a list of all the numpy files of a certain
    format in a directory
    """
    import glob

    experiments_paths = [file for file in glob.glob(dir_path+"*."+data_format)]
    data = []
    for file in experiments_paths:
        data.append(np.load(file,allow_pickle=True))
    return data

data_storage_path = "Project-Course-in-Data-Science/data_storage/"
experiments_storage_path = 'Project-Course-in-Data-Science/Experiments_data/'


if __name__=="__main__":
    e=load_experiments_data(data_storage_path,"npy")
    d=combine_im_metrics([e[1], np.copy(e[1])])
    images = load_imgdir(experiments_storage_path,"jpeg")
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
    metric_stats = load_run(fcm,experiments_storage_path,"jpeg",[None], 0, n_iter=1,paths=False, n_trials=2, save_trials=True,save_stats=True) 
    combine_im_metrics(metric_stats)