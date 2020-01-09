from skimage import io
from skimage import transform as tf
import cv2
from imageio import imread
from imageio import imsave
import matplotlib.pyplot as plt
import numpy as np
import math


def resize_cv(im):
    return cv2.resize(im, (64, 64), interpolation=cv2.INTER_LINEAR)


def rotate_image_randomly(img):
    random_power = (int(np.random.rand(1, 1)*10))
    sign_1 = (-1)**random_power
    sign_2 = (-1)**(int(np.random.rand(1, 1)*10))
    sign_3 = (-1)**(int(np.random.rand(1, 1)*10))

    theta = (np.random.rand(1, 3)*[sign_1, sign_2, sign_3]) / 200
    #
    theta = theta[0]

    # theta = [0, 0, 0.1]

    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    # r = np.random.rand(3, 3)
    #
    # r2 = r/50 #np.matmul(r, 1/10)
    #
    img_rotated = cv2.warpPerspective(img, R, (64, 64))

    return img_rotated


# img = imread('30kmh.jpg')
# img = resize_cv(img)
#
# for i in range(10):
#     rotated_image = rotate_image_randomly(img)
#
#     plt.imshow(rotated_image)
#     plt.show()