from sFCM.sFCM import *
from deepClustering.DFC import DFC 
#from sFCM.demo import *
from utils import *
import cv2
from os import path
import os, tqdm

#import torch
#use_cuda = torch.cuda.is_available()
def experiment_runs(model, true_labels,label_names,*args, **kwargs):
    """
    Runs the models run method on the data n times, calculates the average and standard
    deviation of the metric method.
    """
    metric_list = []
    verbose=False
    if "verbose" in kwargs:
        if kwargs["verbose"]:
            verbose=True
    for i in range(kwargs["n_trials"]):
        if verbose:
            print("Trial {} \n----------------------------------".format(i+1))

        for c, (stat, finish) in tqdm.tqdm(enumerate(model.run(*args,**kwargs)), total = kwargs['n_iter']):
            #One could potentially save the images
            #or do some computation on the response to examine the convergence
            #during the runs
            if finish:
                break
            continue
        metric_list.append(generate_metrics(stat,true_labels,label_names))
        model.re_init()
        
        if "save_trials" in kwargs:
            if kwargs["save_trials"]:
                np.save(data_storage_path_trials+str(kwargs["image_nr"])+"_"+type(model).__name__+"_trial"+str(i),np.array(metric_list[-1],dtype=object))

    if "save_image" in kwargs:
        if kwargs["save_image"]:
            cv2.imwrite( data_storage_path_images+"output.png", stat["im"])

    stats = metric_stats(metric_list)
    if "save_stats" in kwargs:
        if kwargs["save_stats"]:
            np.save(data_storage_path_images+str(kwargs["image_nr"])+"_"+type(model).__name__+"_total"+str(i),np.array(stats,dtype=object))

    return stats

def load_imgdir(dir_path,img_format, grayscale=True):
    import glob
    import cv2
    if grayscale:
        images = [cv2.imread(file, 0) for file in glob.glob(dir_path+"*."+img_format)]
    else:
        images = [cv2.imread(file) for file in glob.glob(dir_path+"*."+img_format)]
    
    return images

def load_impathdir(dir_path,img_format):
    import glob
    import cv2

    images_paths = [file for file in glob.glob(dir_path+"*."+img_format)]
    return images_paths

def simple_cropping(im, cropp_args={"top":0,"bot":0,"left":0,"right":0}, **kwargs):
    """
    Cropps the image into a smaller rectangular shape

    cropps_args: dictionary of how many rows should be removed from the beginning "top",
    the end "bot" and how many columns should be removed from the same ("left","right")
    """
    cropped_im = im[cropp_args["top"]:-cropp_args["bot"],cropp_args["left"]:-cropp_args["right"]]
    return cropped_im


def load_run(model, true_labels,label_names,*args, paths=False,img_format=None, dir_path=None, cropping=False,cropping_method=simple_cropping, preloaded_images=None, **kwargs):
    """args are used as the input parameters to the model run method 
        kwargs are used deterministically with "n_trials" required
    """
    
    if paths:
        images = load_impathdir(dir_path,img_format)
    elif preloaded_images:
        images=preloaded_images
    else:
        images = load_imgdir(dir_path,img_format)
    if cropping:
        images = [cropping_method(im,**kwargs) for im in images]
        true_labels = [[cropping_method(label,**kwargs) for label in label_set] for label_set in true_labels]
    metrics = []

    verbose=False
    if "verbose" in kwargs:
        if kwargs["verbose"]:
            verbose=True
    if verbose:
        experiment_para_str = "Experiment parameters\n"+"".join([key+":"+str(kwargs[key])+"\n" if key!="verbose" else "" for key in kwargs ])+"----------------------------------"
        print("Begining experiment with {} model \n----------------------------------".format(type(model).__name__))
        print(experiment_para_str)
    for i,im in enumerate(images):
        if verbose:
            print("Starting experiment trials with image {} \n----------------------------------".format(i))
        if "save_trials" in kwargs or "save_stats" in kwargs:
            metrics.append(experiment_runs(model,true_labels[i],label_names[i], im,*args, image_nr=i,**kwargs))
        else:
            metrics.append(experiment_runs(model,true_labels[i],label_names[i], im,*args,**kwargs))
    return metrics

def generate_metrics(stat,true_labels,label_names):
    """
    Generates metrics for clustering based on stat containing
    membership function and data labels, in that order.
    """
    partition_c = partition_coefficient(stat["membership"])
    partition_e = partition_entropy(stat["membership"])
    output_cluster = Cluster(stat["labels"])
    true_cluster = [Cluster(x, consider=[1]) for x in true_labels]
    iox_matrix, iox_cm = Cluster.distribution(true_cluster, label_names, output_cluster, metric = 'iox')
    iou_matrix, iou_cm = Cluster.distribution(true_cluster, label_names, output_cluster, metric = 'iou')
    iou_mapping        = Cluster.iou_mapping(true_cluster, label_names, output_cluster)

    return partition_c,partition_e,iox_matrix, iox_cm, iou_matrix, iou_cm, iou_mapping, stat["labels"]#partition_c,partition_e,IOU, cm


def metric_stats(metrics):
    """
    Computes the average and variance of the different metrics
    if not array or lists, also add the original metric list
    for histogram purposes
    """
    metric_array = np.array(metrics)

    stats = []
    
    for m in range(metric_array.shape[1]):

        if type(metric_array[0,m])==float or type(metric_array[0,m])==np.float64:
            stats.append([np.average(metric_array[:,m].astype(float)),np.var(metric_array[:,m].astype(float)),metric_array[:,m]])
        else:
            stats.append([metric_array[:,m]])
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

def load_experiments_data(dir_path, data_format,item=False):
    """
    Creates a list of all the numpy files of a certain
    format in a directory
    """
    import glob

    experiments_paths = [file for file in glob.glob(dir_path+"*."+data_format)]
    data = []
    for file in experiments_paths:
        if item:
            data.append(np.load(file,allow_pickle=True).item())
        else:
            data.append(np.load(file,allow_pickle=True))
    return data


directory = path.dirname(path.abspath(path.abspath(__file__)))

data_storage_path_trials = path.join(directory, 'data_storage', 'trials')+ '/'#"Project-Course-in-Data-Science/data_storage/trials"
data_storage_path_images = path.join(directory, 'data_storage', 'images')+ '/'#"Project-Course-in-Data-Science/data_storage/images"
data_storage_path_compiled = path.join(directory, 'data_storage', 'compiled')+ '/'
experiments_storage_path_images = path.join(directory, 'Experiments_data', 'Images') + '/'#'Project-Course-in-Data-Science/Experiments_data/Images/'
experiments_storage_path_label_prob = path.join(directory, 'Experiments_data', 'ground_truths') + '/'#'Project-Course-in-Data-Science/Experiments_data/ground_truths/'
default_im_format = "jpeg"


if __name__=="__main__":
    experiments_name = "synthetic_brain"
    images = load_imgdir(experiments_storage_path_images,"png")
    cropping = {"top":100,"bot":100,"left":100,"right":100}
    cropped_image = simple_cropping(images[0],  cropp_args=cropping)

    fcm = sFCM(2, 9, 1, 1, 3, cropped_image.shape)
    #dfc = DFC(minLabels=9, max_iters=200)
    #dfc.initialize_clustering(cropped_image)

    labels_dict = load_experiments_data(experiments_storage_path_label_prob, "npy",item=True)
    labels_names = [[key for key in sample] for sample in labels_dict]
    labels_probs = [[sample[key] for key in sample] for sample in labels_dict]
    sample_labels = [[np.random.binomial(1, x) for x in label_p] for label_p in labels_probs]


    metric_stats = load_run(fcm,sample_labels, labels_names, eps=0.02,preloaded_images=images,cropping=True, n_iter=sFCM.MAX_ITER, cropp_args=cropping,paths=False, n_trials=5, save_trials=True,save_stats=True, verbose=True)
    #metric_stats = load_run(dfc,sample_labels, labels_names,preloaded_images=images,cropping=True, n_iter=1, cropp_args=cropping,paths=False, n_trials=2, save_trials=True,save_stats=True, verbose=True)
    #np.save(data_storage_path_compiled+experiments_name+" "+type(model).__name__, np.array(combine_im_metrics(metric_stats), dtype=object))