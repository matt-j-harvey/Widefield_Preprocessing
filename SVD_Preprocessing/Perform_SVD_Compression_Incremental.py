import numpy as np
import os
import h5py
import sys
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import IncrementalPCA
#from dask_ml.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy import ndimage

import datetime
sys.path.append(r"C:\Users\Creature\Documents\Matt Scripts\Widefield_Preprocessing")

import Widefield_General_Functions


def load_mask(home_directory):

    # Loads the mask for a video, returns a list of which pixels are included, as well as the original image height and width
    mask = np.load(home_directory + "/Generous_Mask.npy")

    image_height = np.shape(mask)[0]
    image_width = np.shape(mask)[1]

    mask = np.where(mask>0.1, 1, 0)
    mask = mask.astype(int)
    flat_mask = np.ndarray.flatten(mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    return indicies, image_height, image_width


def smooth_temporal(data, window_size=3):
    smoothed_data = convolve2d(data, np.ones((1, window_size)), 'same') / window_size
    return smoothed_data


def smooth_spatial(data, indicies, height, width, sigma=3):

    number_of_frames = np.shape(data)[0]
    for frame_index in range(number_of_frames):
        raw_data = data[frame_index]
        template = np.zeros(height * width)
        template[indicies] = raw_data
        template = np.reshape(template, (height, width))
        template = ndimage.gaussian_filter(template, sigma=sigma)
        template = np.reshape(template, (height * width))
        smoothed_data = template[indicies]
        data[frame_index] = smoothed_data

    return data


def plot_components(base_directory):

    # Load Sparial Components
    spatial_components = np.load(os.path.join(base_directory, "SVD_Components.npy"))
    print(np.shape(spatial_components))

    # Load Mask
    indicies, image_height, image_width = load_mask(base_directory)

    # Create Figure
    figure_1 = plt.figure()
    rows = 10
    columns = 10

    for x in range(100):

        component_data = spatial_components[x]
        component_image = np.zeros(image_height * image_width)
        component_image[indicies] = component_data
        component_image = np.reshape(component_image, (image_height, image_width))

        axis = figure_1.add_subplot(rows, columns, x+1)
        axis.imshow(component_image)
        axis.axis('off')

    plt.savefig(os.path.join(base_directory, "SVD_Components.png"))
    plt.close()


def perform_svd_compression(base_directory):

    # Set File Settings
    data_file = os.path.join(base_directory, "Delta_F.hdf5")

    # Load Mask
    indicies, image_height, image_width = load_mask(base_directory)

    # Load Delta F Data
    downsampled_file_object = h5py.File(data_file, 'r')
    data_matrix = downsampled_file_object["Data"]
    number_of_frames, number_of_pixels = np.shape(data_matrix)

    # Get Chunk Structure
    chunk_size = 3000
    number_of_components = 100
    number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = Widefield_General_Functions.get_chunk_structure(chunk_size, number_of_frames)
    print("Chunk sizes", chunk_sizes)

    # Ensure all Chunks Are Larger Than The Number of Components
    if chunk_sizes[-1] <= number_of_components * 5:
        final_chunk_size = chunk_sizes[-1]

        number_of_chunks = number_of_chunks - 1
        del chunk_sizes[-1]
        del chunk_starts[-1]
        del chunk_stops[-1]

        chunk_sizes[-1] = chunk_sizes[-1] + final_chunk_size
        chunk_stops[-1] = chunk_stops[-1] + final_chunk_size

    print("Chunk sizes", chunk_sizes)


    # Create Model
    model = IncrementalPCA()

    for chunk_index in range(number_of_chunks):
        print("Chunk Index", chunk_index, " of ", number_of_chunks, "Time: ", datetime.datetime.now())
        chunk_start = chunk_starts[chunk_index]
        chunk_stop = chunk_stops[chunk_index]
        chunk_data = data_matrix[chunk_start:chunk_stop]

        # Temporal Smoothing
        chunk_data = smooth_temporal(chunk_data)

        # Spatial Smoothing
        chunk_data = smooth_spatial(chunk_data, indicies, image_height, image_width)

        # Remove Nans
        chunk_data = np.nan_to_num(chunk_data)

        print("Chunk data min", np.min(chunk_data))
        print("Chunk data max", np.max(chunk_data))

        model.partial_fit(chunk_data)

    components = model.components_
    svd_mean = model.mean_

    np.save(os.path.join(base_directory, "SVD_Components.npy"), components)
    np.save(os.path.join(base_directory, "SVD_Mean.npy"), svd_mean)

    # Transform_Data
    transformed_data = []
    for chunk_index in range(number_of_chunks):
        chunk_start = chunk_starts[chunk_index]
        chunk_stop = chunk_stops[chunk_index]
        chunk_data = data_matrix[chunk_start:chunk_stop]
        transformed_chunk = model.transform(chunk_data)
        for datapoint in transformed_chunk:
            transformed_data.append(datapoint)

    transformed_data = np.array(transformed_data)
    print("Transformed Data Shape", np.shape(transformed_data))
    np.save(os.path.join(base_directory, "SVD_Transformed_data.npy"), transformed_data)

