from ClassDataSet import *
from ClassModel import *
from Doplots import DoPlots_activations
import pickle
import time
import numpy as np
import seaborn as sns



start_time_total = time.time()

number_of_labels = 1
number_of_images = 3

data_set_default = DataSet(data_set='german', number_of_labels=number_of_labels, number_of_images=number_of_images, augment_dataset=False, grayscale=False, normalize=False, contrast=False)

data_set_1 = DataSet(data_set='german', number_of_labels=number_of_labels, number_of_images=number_of_images, augment_dataset=False, grayscale=False, normalize=True, contrast=False)

data_set_2 = DataSet(data_set='german', number_of_labels=number_of_labels, number_of_images=number_of_images, augment_dataset=False, grayscale=True, normalize=True, contrast=False)

data_set_3 = DataSet(data_set='german', number_of_labels=number_of_labels, number_of_images=number_of_images, augment_dataset=False, grayscale=True, normalize=True, contrast=True)

img4 = rotate_image_randomly(data_set_default.training_images[0])


img5 = rotate_image_randomly(data_set_default.training_images[0])


img6 = rotate_image_randomly(data_set_default.training_images[0])

# -------------- Color augmentation
img7 = cv2.normalize(img6, None, 0, 255, cv2.NORM_MINMAX)
# cv2.imshow('norm', norm)

img7 = cv2.cvtColor(img7, cv2.COLOR_RGB2GRAY)

clahe = cv2.createCLAHE(clipLimit=3.0)  # clipLimit=2.0, tileGridSize=(8, 8))
img7 = clahe.apply(img7)

img7 = cv2.cvtColor(img7, cv2.COLOR_BGR2RGB)

plt.figure(0)
plt.imshow(data_set_default.training_images[0])

plt.figure(1)
plt.imshow(data_set_1.training_images[0])

plt.figure(2)
plt.imshow(data_set_2.training_images[0])

plt.figure(3)
plt.imshow(data_set_3.training_images[0])

plt.figure(4)
plt.imshow(img4)

plt.figure(5)
plt.imshow(img5)

plt.figure(6)
plt.imshow(img6)

plt.figure(7)
plt.imshow(img7)

plt.show()