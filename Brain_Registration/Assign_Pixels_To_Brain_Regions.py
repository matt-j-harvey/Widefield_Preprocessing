import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
import os
import math
import scipy
import tables
from bisect import bisect_left
import cv2
from sklearn.decomposition import TruncatedSVD
from pathlib import Path
import joblib
from scipy import signal, ndimage, stats
from skimage.transform import resize
from scipy.interpolate import interp1d

# Mathmatical Functions




def get_average_response(session_list, onsets_file, save_directory):

    average_response = []

    trial_start = 0
    trial_stop = 40
    for base_directory in session_list:
        print(base_directory)

        # Load Delta F Matrix
        delta_f_matrix_filepath = os.path.join(base_directory, "Delta_F.h5")
        delta_f_matrix_container = tables.open_file(delta_f_matrix_filepath, mode='r')
        delta_f_matrix = delta_f_matrix_container.root['Data']

        # Load Onsets
        onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", onsets_file))

        # Create Trial Tensor
        activity_tensor = get_activity_tensor(delta_f_matrix, onsets, trial_start, trial_stop)

        # Get Mean Activity
        mean_activity = np.mean(activity_tensor, axis=0)

        # Add To List
        average_response.append(mean_activity)

    average_response = np.array(average_response)
    average_response = np.mean(average_response, axis=0)
    np.save(save_directory, average_response)


def load_mask(home_directory):

    # Loads the mask for a video, returns a list of which pixels are included, as well as the original image height and width
    mask = np.load(home_directory + "/mask.npy")

    image_height = np.shape(mask)[0]
    image_width = np.shape(mask)[1]

    mask = np.where(mask>0.1, 1, 0)
    mask = mask.astype(int)
    flat_mask = np.ndarray.flatten(mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    return indicies, image_height, image_width


def assign_pixels(base_directory):

    # Load Atlas Regions
    atlas_region_mapping = np.load(r"/home/matthew/Documents/Allen_Atlas_Templates/Allen_Atlas_Mapping.npy")
    number_of_regions = np.max(atlas_region_mapping)

    # Load Atlas Transformation Details
    atlas_alignment_dictionary = np.load(os.path.join(base_directory, "Atlas_Alignment_Dictionary.npy"), allow_pickle=True)
    atlas_alignment_dictionary = atlas_alignment_dictionary[()]
    atlas_rotation = atlas_alignment_dictionary['rotation']
    atlas_x_scale_factor = atlas_alignment_dictionary['x_scale_factor']
    atlas_y_scale_factor = atlas_alignment_dictionary['y_scale_factor']
    atlas_x_shift = atlas_alignment_dictionary['x_shift']
    atlas_y_shift = atlas_alignment_dictionary['y_shift']

    # Rotate Atlas
    atlas_region_mapping = ndimage.rotate(atlas_region_mapping, atlas_rotation, reshape=False)
    atlas_region_mapping = np.clip(atlas_region_mapping, a_min=0, a_max=number_of_regions)

    # Scale Atlas
    atlas_height = np.shape(atlas_region_mapping)[0]
    atlas_width = np.shape(atlas_region_mapping)[1]
    atlas_region_mapping = resize(atlas_region_mapping, (int(atlas_y_scale_factor * atlas_height), int(atlas_x_scale_factor * atlas_width)), preserve_range=True)

    # Load mask
    indicies, image_height, image_width = load_mask(base_directory)

    # Place Into Bounding Box
    bounding_array = np.zeros((800, 800))
    x_start = 100
    y_start = 100
    atlas_height = np.shape(atlas_region_mapping)[0]
    atlas_width = np.shape(atlas_region_mapping)[1]
    bounding_array[y_start + atlas_y_shift: y_start + atlas_y_shift + atlas_height,x_start + atlas_x_shift: x_start + atlas_x_shift + atlas_width] = atlas_region_mapping
    bounded_atlas = bounding_array[y_start:y_start + image_height, x_start:x_start + image_width]

    # Mask Atlas
    bounded_atlas = np.ndarray.flatten(bounded_atlas)
    masked_atlas = np.zeros((image_height * image_width))
    pixel_region_assigments = []
    for pixel_index in indicies:
        masked_atlas[pixel_index] = bounded_atlas[pixel_index]
        pixel_region_assigments.append(bounded_atlas[pixel_index])
    masked_atlas = np.ndarray.reshape(masked_atlas, (image_height, image_width))

    # Save Mapping
    np.save(os.path.join(base_directory, "Pixel_Assignmnets.npy"), pixel_region_assigments)

    plt.imshow(masked_atlas, cmap='jet')
    plt.savefig(os.path.join(base_directory, "Pixel_Region_Assignmnet.png"))
    plt.close()


