from sFCM.sFCM import *
from deepClustering.DFC import DFC 
from utils import *
from experiment_util import *
from os import path
import os, tqdm, cv2, argparse

directory = path.dirname(path.abspath(path.abspath(__file__)))

data_storage_path_trials = path.join(directory, 'data_storage', 'trials')+ '/'#"Project-Course-in-Data-Science/data_storage/trials"
data_storage_path_images = path.join(directory, 'data_storage', 'images')+ '/'#"Project-Course-in-Data-Science/data_storage/images"
data_storage_path_compiled = path.join(directory, 'data_storage', 'compiled')+ '/'
experiments_storage_path_images = path.join(directory, 'Experiments_data', 'Images') + '/'#'Project-Course-in-Data-Science/Experiments_data/Images/'
experiments_storage_path_label_prob = path.join(directory, 'Experiments_data', 'ground_truths') + '/'#'Project-Course-in-Data-Science/Experiments_data/ground_truths/'
default_im_format = "jpeg"

parser = argparse.ArgumentParser(description='Perform clustering')

parser.add_argument('--name', type=str, help='Name of experiment', default='demo')
parser.add_argument('--cropping', nargs='+', default=['100', '100', '100', '100'], help = 'Cropping paramters, e.g., --cropping <top> <bot> <left> <right>')
parser.add_argument('--model', type=str, default='sFCM', help='Type of model (DFC or sFCM). Note that this script runs with demo hyper-parameters.')

args = parser.parse_args()


experiments_name = args.name #"synthetic_brain"
images = load_imgdir(experiments_storage_path_images,"png")
cropping = {"top":int(args.cropping[0]),"bot":int(args.cropping[1]),"left":int(args.cropping[2]),"right":int(args.cropping[3])}#{"top":100,"bot":100,"left":100,"right":100}
cropped_image = simple_cropping(images[0],  cropp_args=cropping)

if args.model.lower() == 'dfc':
    model = DFC(minLabels=9, max_iters=100, nChannel=50, nConv=3)
    model.initialize_clustering(cropped_image)
elif args.model.lower() == 'sfcm':
    model = sFCM(2, 9, 1, 1, 3, cropped_image.shape)
    model.maxIters = model.MAX_ITER

#dfc = DFC(minLabels=9, max_iters=100, maxLabels=200,nChannel=50, nConv=3)
#dfc.initialize_clustering(cropped_image)

labels_dict = load_experiments_data(experiments_storage_path_label_prob, "npy",item=True)
labels_names = [[key for key in sample] for sample in labels_dict]
labels_probs = [[sample[key] for key in sample] for sample in labels_dict]
sample_labels = [[np.random.binomial(1, x) for x in label_p] for label_p in labels_probs]


#metric_stats = load_run(fcm,sample_labels, labels_names, eps=0.02,preloaded_images=images,cropping=True, n_iter=sFCM.MAX_ITER, cropp_args=cropping,paths=False, n_trials=5, save_trials=True,save_stats=True, verbose=True)
metric_stats = load_run(model,sample_labels, labels_names,preloaded_images=images,cropping=True, n_iter=model.maxIters, cropp_args=cropping,paths=False, n_trials=1, show_convergence=True,show_image=True, save_trials=False,save_stats=False, verbose=True)
#np.save(data_storage_path_compiled+experiments_name+" "+type(model).__name__, np.array(combine_im_metrics(metric_stats), dtype=object))