# from PIL import Image
import glob
import scipy.io as sio
from image_augmentation_function import *
import os
import pandas as pd
import numpy as np
import keras
from sklearn import preprocessing
import seaborn as sns

german_data_dir = os.path.abspath('GTSRB_Final_Training_Images/GTSRB/Final_Training/Images')
swedish_data_dir = 'code'


class DataSet:
    def __init__(self, data_set, number_of_labels=None, number_of_images=None, grayscale=False,
                 normalize=False, contrast=False, augment_dataset=False):
        self.data_set = data_set
        self.number_of_labels = number_of_labels
        self.number_of_images = number_of_images
        self.grayscale = grayscale
        self.normalize = normalize
        self.contrast = contrast
        self.augment_dataset = augment_dataset

        # Initiate stuff
        self.swedish_data_encoder = preprocessing.LabelEncoder()
        self.dataset_images = []
        self.dataset_labels_asarray = []
        self.dataset_labels =[]

        self.training_images = []
        self.training_labels = []

        self.validation_images = []
        self.validation_labels = []

        self.test_images = []
        self.test_labels = []

        # do stuff
        self.import_dataset()
        self.randomize_dataset()
        self.split_dataset()

    def import_dataset(self):
        if self.data_set == 'swedish':
            self.import_swedish_dataset()
        elif self.data_set == 'german':
            self.import_german_dataset()
        elif self.data_set == 'both':
            print('ok lol')

    def import_german_dataset(self):
        global german_data_dir

        os.path.exists(german_data_dir)
        list_images = []
        output = []
        list_dir = os.listdir(german_data_dir)


        if self.number_of_labels:
            labels_to_use = self.number_of_labels
            dir_labels_to_use = list_dir[0:labels_to_use]
        else:
            dir_labels_to_use = list_dir

        for dir in dir_labels_to_use:
            print(dir)
            if dir == '.DS_Store':
                continue

            inner_dir = os.path.join(german_data_dir, dir)
            csv_file = pd.read_csv(os.path.join(inner_dir, "GT-" + dir + '.csv'), sep=';')

            if self.number_of_images:
                cvs_file_to_use = csv_file[0:self.number_of_images]
            else:
                cvs_file_to_use = csv_file

            if self.augment_dataset:
                image_number = cvs_file_to_use.Filename.size
                ratio = self.number_of_images / image_number
                times_to_augemnt = int(math.ceil(self.number_of_images / image_number))
                total_augmented_images = image_number * (ratio-1)
            else:
                times_to_augemnt = 0
                total_augmented_images = 0

            augment_counter = 0

            for row in cvs_file_to_use.iterrows():
                img_path = os.path.join(inner_dir, row[1].Filename)
                img = imread(img_path)
                img = img[row[1]['Roi.X1']:row[1]['Roi.X2'], row[1]['Roi.Y1']:row[1]['Roi.Y2'], :]
                img = resize_cv(img)

                if self.normalize:
                    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                    # cv2.imshow('norm', norm)

                if self.grayscale:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    # cv2.imshow('gray', gray)
                    if self.contrast:
                        clahe = cv2.createCLAHE(clipLimit=3.0)  # clipLimit=2.0, tileGridSize=(8, 8))
                        img = clahe.apply(img)
                    # cv2.imshow('cont', cont)
                    img = cv2.merge((img, img, img))
                    # img = tf.expand_dims(img, 2)

                list_images.append(img)
                output.append(row[1].ClassId)

                if times_to_augemnt > 1 and augment_counter < total_augmented_images:
                    for augemnt_i in range(times_to_augemnt):
                        # augment image and resize
                        rotated_img = rotate_image_randomly(img)  # This should also resize the image

                        list_images.append(rotated_img)
                        output.append(row[1].ClassId)
                        print('augmenting:', row[1].ClassId)
                        augment_counter = augment_counter + 1

                self.dataset_images = np.stack(list_images)
                self. dataset_labels_asarray = output
                self.dataset_labels = keras.utils.np_utils.to_categorical(output)

        # fig = sns.distplot(output, kde=False, bins=43, hist=True, hist_kws=dict(edgecolor="black", linewidth=2))
        # fig.set(title="Traffic signs frequency graph",
        #         xlabel="ClassId",
        #         ylabel="Frequency")
        # plt.show()

    def import_swedish_dataset(self):
        global swedish_data_dir
        mat_contents = sio.loadmat(swedish_data_dir + '/Set1NewImages/useful.mat', struct_as_record=False, squeeze_me=True)
        useful = mat_contents['useful']

        labels = useful.label

        list_images = []
        for filename in glob.glob(swedish_data_dir + '/Set1NewImages/*.jpg'):  # assuming gif
            # img = Image.open(filename)
            img = imread(filename)
            if self.normalize:
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                # cv2.imshow('norm', norm)

            if self.grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                # cv2.imshow('gray', gray)
                if self.contrast:
                    clahe = cv2.createCLAHE(clipLimit=3.0)  # clipLimit=2.0, tileGridSize=(8, 8))
                    img = clahe.apply(img)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            list_images.append(img)

        self.dataset_images = np.stack(list_images)
        self.swedish_data_encoder.fit(labels)
        Y = self.swedish_data_encoder.transform(labels)
        self.dataset_labels_asarray = Y
        self.dataset_labels = keras.utils.to_categorical(Y)

    def randomize_dataset(self):
        randomize = np.arange(len(self.dataset_labels))
        np.random.shuffle(randomize)
        self.dataset_labels = self.dataset_labels[randomize]
        self.dataset_images = self.dataset_images[randomize]

    def split_dataset(self):
        x = self.dataset_images
        y = self.dataset_labels
        split_size = int(x.shape[0] * 0.6)
        train_x, val_x = x[:split_size], x[split_size:]
        train_y, val_y = y[:split_size], y[split_size:]

        split_size = int(val_x.shape[0] * 0.5)
        val_x, test_x = val_x[:split_size], val_x[split_size:]
        val_y, test_y = val_y[:split_size], val_y[split_size:]

        self.training_labels = train_y
        self.training_images = train_x

        self.validation_labels = val_y
        self.validation_images = val_x

        self.test_labels = test_y
        self.test_images = test_x


def resize_cv(img):
    return cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR)


