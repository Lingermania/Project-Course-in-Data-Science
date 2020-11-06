from sFCM import *
from demo import *
import torch
use_cuda = torch.cuda.is_available()
def experiment_runs(model,metric_method,*args, **kwargs):
    """
    Runs the models run method on the data n times, calculates the average and standard
    deviation of the metric method.
    """
    metric_list = []
    for i in range(kwargs["n_trials"]):

        for c, (stat, finish) in enumerate(model.run(*args)):
            #One could potentially save the images
            #or do some computation on the response to examine the convergence
            #during the runs
            continue
        metric_list.append(metric_method(model))
    
    metric_array = np.array(metric_list)
    metric_avg = np.average(metric_array)
    metric_std = np.std(metric_array)

    return metric_avg,metric_std

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


def load_run(model,metric_method,dir_path,img_format,*args, paths=False, **kwargs):
    """args are used as the input parameters to the model run method 
        kwargs are used deterministically with "n_trials" required
    """
    if paths:
        images = load_impathdir(dir_path,img_format)
    else:
        images = load_imgdir(dir_path,img_format)
    metrics = []
    for im in images:

        metrics.append(experiment_runs(model,metric_method, im,*args,**kwargs))
    return metrics
def metric_0():
    return 0


if __name__=="__main__":
    images = load_imgdir('Project-Course-in-Data-Science/sFCM/gifs/',"jpeg")
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
    load_run(model,metric_0,'Project-Course-in-Data-Science/sFCM/gifs/',"jpeg",paths=True, n_trials=30) 