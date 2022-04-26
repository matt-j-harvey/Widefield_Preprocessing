import numpy as np
import matplotlib.pyplot as plt
import h5py
import tables
from scipy import signal, ndimage, stats
from sklearn.linear_model import LinearRegression
from skimage.morphology import white_tophat
from sklearn.preprocessing import StandardScaler
from skimage.transform import rescale
from PIL import Image
import os
import cv2
from datetime import datetime

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from time import sleep
import sys



def get_blue_file(base_directory):
    file_list = os.listdir(base_directory)
    for file in file_list:
        if "Blue" in file:
            return base_directory + "/" + file

def check_max_projection(home_directory):
    print("Getting Max Projection")

    # Load Data
    blue_file = get_blue_file(home_directory)
    blue_data_container = h5py.File(blue_file, 'r')
    blue_data = blue_data_container["Data"]

    sample = blue_data[:, 0:1000]
    max_projection = np.max(sample, axis=1)
    max_projection = np.reshape(max_projection, (600, 608))

    np.save(home_directory + "/max_projection", max_projection)



    plt.imshow(max_projection)
    plt.show(block=False)
    plt.pause(5)
    plt.close()
