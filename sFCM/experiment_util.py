def experiment_runs(model,metric_method,*args, **kwargs):
    """
    Runs the models run method on the datase n times, calculates the average and standard
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