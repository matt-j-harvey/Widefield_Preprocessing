import numpy as np
import matplotlib.pyplot as plt
import h5py
import tables
from scipy import signal, ndimage, stats
from sklearn.linear_model import LinearRegression
from skimage.morphology import white_tophat
from PIL import Image
import os
import cv2
print("cv2 version", cv2.__version__)
from datetime import datetime
import sys
from matplotlib.pyplot import cm
sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions


# Load Maskdef get_blue_file(base_directory):
def get_blue_file(base_directory):
    file_list = os.listdir(base_directory)
    for file in file_list:
        if "Blue" in file:
            return base_directory + "/" + file

def get_violet_file(base_directory):
    file_list = os.listdir(base_directory)
    for file in file_list:
        if "Violet" in file:
            return base_directory + "/" + file

def get_region_pixels(pixel_assignments, selected_regions):

    selected_pixels = []
    for region in selected_regions:
        region_mask = np.where(pixel_assignments == region, 1, 0)
        region_indicies = np.nonzero(region_mask)[0]
        for index in region_indicies:
            selected_pixels.append(index)
    selected_pixels.sort()
    return selected_pixels


def check_correct_region(blue_data, indicies, v1_pixels):

    colourmap = cm.get_cmap('binary_r')


    v1_pixels_in_brain_space = indicies[v1_pixels]

    frame_1 = blue_data[:, 0]
    frame_1 = np.divide(frame_1, np.max(frame_1))
    frame_1_rgba = colourmap(frame_1)
    frame_1_rgba[v1_pixels_in_brain_space] = (1, 1, 0, 1)

    frame_1_rgba = np.ndarray.reshape(frame_1_rgba, (600, 608, 4))

    plt.imshow(frame_1_rgba)
    plt.show()


def get_region_trace(blue_data, indicies, v1_pixels):

    v1_pixels_in_brain_space = indicies[v1_pixels]
    v1_trace = blue_data[v1_pixels_in_brain_space]

    print("v1 trace shape", np.shape(v1_trace))

    v1_mean_trace = np.mean(v1_trace, axis=0)
    return v1_mean_trace

#Load Data
base_directory = "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging"

blue_file = get_blue_file(base_directory)
blue_data_container = h5py.File(blue_file, 'r')
blue_data           = blue_data_container["Data"]

violet_file = get_violet_file(base_directory)
violet_data_container = h5py.File(violet_file, 'r')
violet_data           = violet_data_container["Data"]

indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)


# Load Region Assigments
pixel_assignments = np.load(os.path.join(base_directory, "Pixel_Assignmnets.npy"))

# Get V1 Pixels
v1 = [45, 46]
v1_pixels = get_region_pixels(pixel_assignments, v1)
#check_correct_region(blue_data, indicies, v1_pixels)



v1_mean_trace = get_region_trace(blue_data, indicies, v1_pixels)
np.save(base_directory + "/Raw_V1_Trace.npy", v1_mean_trace)
plt.plot(v1_mean_trace)
plt.show()