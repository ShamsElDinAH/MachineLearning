from ClassDataSet import *
from ClassModel import *
from Doplots import DoPlots_activations
import pickle
import time
import numpy as np
import seaborn as sns

number_of_labels = 43
number_of_images = 2000

data_set_1 = DataSet(data_set='german', number_of_labels=number_of_labels, number_of_images=number_of_images, augment_dataset=True, grayscale=True, normalize=True, contrast=True)

test_name = str(number_of_labels)+'_lables_' + str(number_of_images) + '_images'

dataset_file = open(test_name + '_dataset.pickl', 'wb')
pickle.dump(data_set_1, dataset_file)
dataset_file.close()

# dataset_file = open(test_name+'_dataset.pickl', 'rb')
# data_set_reloaded = pickle.load(dataset_file)
# dataset_file.close()
#
# print(1)
