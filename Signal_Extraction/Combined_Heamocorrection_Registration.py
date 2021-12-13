import sys
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
from datetime import datetime


sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Create_Sample_Video
import Reconstruct_Raw_Video
import Widefield_General_Functions
import Register_Delta_F


def check_led_colours(blue_array, violet_array):
    figure_1 = plt.figure()
    axes_1 = figure_1.subplots(1, 2)

    blue_image = blue_array[:, 0]
    blue_image = np.reshape(blue_image, (600,608))
    axes_1[0].set_title("Blue?")
    axes_1[0].imshow(blue_image)

    violet_image = violet_array[:, 0]
    violet_image = np.reshape(violet_image, (600,608))
    axes_1[1].set_title("Violet?")
    axes_1[1].imshow(violet_image)
    plt.show()


def get_max_projection(array, home_directory):

    print("Getting Max Projection")
    sample = array[:, 0:1000]

    max_projection = np.max(sample, axis=1)
    max_projection = np.reshape(max_projection, (600, 608))

    np.save(home_directory + "/max_projection", max_projection)


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



def get_filter_coefficients():

    # Create Butterwoth Bandpass Filter
    sampling_frequency = 25  # In Hertz
    cutoff_frequency = 8.5  # In Hertz
    w = cutoff_frequency / (sampling_frequency / 2)  # Normalised frequency
    low_cutoff_frequency = 0.01
    w_low = low_cutoff_frequency / (sampling_frequency / 2)
    b, a = signal.butter(2, [w_low, w], 'bandpass')

    return b, a




def load_transformation_details(base_directory):

    # Load Transformation Details
    transformation_details = np.load(os.path.join(base_directory, "Transformation_Dictionary.npy"), allow_pickle=True)
    transformation_details = transformation_details[()]
    return transformation_details



def load_mask(home_directory, transformation_details):

    # Load Mask:
    mask = np.load(home_directory + "/mask.npy")
    mask = np.where(mask > 0.1, 1, 0)
    mask = mask.astype(int)

    # Get Dimensions
    image_height = np.shape(mask)[0]
    image_width = np.shape

    # Load Variables From Dictionary
    rotation = transformation_details['rotation']
    x_shift = transformation_details['x_shift']
    y_shift = transformation_details['y_shift']

    # Invert These As We Are Mapping From Template To Our Data
    rotation = -1 * rotation
    x_shift = -1 * x_shift
    y_shift = -1 * y_shift

    # Rotate Mask
    transformed_mask = ndimage.rotate(mask, rotation, reshape=False)

    # Translate
    transformed_mask = np.roll(a=transformed_mask, axis=0, shift=y_shift)
    transformed_mask = np.roll(a=transformed_mask, axis=1, shift=x_shift)

    # Re-Binarise
    transformed_mask = np.where(transformed_mask > 0.01, 1, 0)
    transformed_mask = np.ndarray.astype(transformed_mask, int)

    # Flatten and Get Indicies
    flat_mask = np.ndarray.flatten(transformed_mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)
    indicies = list(indicies)

    return indicies, image_height, image_width


def perform_bandpass_filter(blue_data, violet_data, b, a):
    blue_data = signal.filtfilt(b, a, blue_data, axis=1)
    violet_data = signal.filtfilt(b, a, violet_data, axis=1)

    return blue_data, violet_data


def heamocorrection_regression(blue_data, violet_data):

    # Perform Regression
    chunk_size = np.shape(blue_data)[0]
    for pixel in range(chunk_size):

        violet_trace = violet_data[pixel]
        blue_trace = blue_data[pixel]

        slope, intercept, r, p, stdev = stats.linregress(violet_trace, blue_trace)
        violet_trace = np.multiply(violet_trace, slope)
        blue_trace = np.subtract(blue_trace, violet_trace)

        blue_min = np.min(blue_trace)
        if blue_min < 0:
            blue_trace = np.add(blue_trace, abs(blue_min))

        blue_data[pixel] = blue_trace

    return blue_data


def heamocorrection_ratio(blue_data, violet_data):
    print("Blue data", np.shape(blue_data))

    ratio = np.divide(blue_data, violet_data)
    mean_ratio = np.mean(ratio, axis=1)

    print("Mean ratio", np.shape(mean_ratio))
    corrected_data = ratio / mean_ratio[:, None]
    return corrected_data


def calculate_delta_f(activity_matrix, baseline_vector):

    # Transpose Baseline Vector so it can be used by numpy subtract
    baseline_vector = baseline_vector[:, np.newaxis]

    # Get Delta F
    delta_f = np.subtract(activity_matrix, baseline_vector)

    # Remove Negative Values
    delta_f = np.clip(delta_f, a_min=0, a_max=None)

    # Divide by baseline
    delta_f_over_f = np.divide(delta_f, baseline_vector)

    # Remove NANs
    delta_f_over_f = np.nan_to_num(delta_f_over_f)

    return delta_f_over_f


def normalise_delta_f(delta_f_matrix, max_pixel_values, min_pixel_values):

    # Transpose Baseline Vector so it can be used by numpy subtract
    max_pixel_values = max_pixel_values[:, np.newaxis]
    min_pixel_values = min_pixel_values[:, np.newaxis]

    # First Calculate Maximum Possible Delta F
    max_delta_f = np.subtract(max_pixel_values, min_pixel_values)
    max_delta_f_over_f = np.divide(max_delta_f, min_pixel_values)

    # Divide Pixels by Maximum Delta F
    delta_f_matrix = np.divide(delta_f_matrix, max_delta_f_over_f)

    print("Pre Clip Chunk_min", np.min(delta_f_matrix))
    print("Pre Clip Chunk_max", np.max(delta_f_matrix))
    delta_f_matrix = np.clip(delta_f_matrix, a_min=0, a_max=1)
    print("Post Clip Chunk_min", np.min(delta_f_matrix))
    print("Post Clip Chunk_max", np.max(delta_f_matrix))

    # Remove Nans
    delta_f_matrix = np.nan_to_num(delta_f_matrix)

    print("Chunk_min", np.min(delta_f_matrix)), "Chunk Max", np.max(delta_f_matrix)
    return delta_f_matrix


def process_pixels(blue_data, violet_data, output_file, home_directory, bandpass_filter=True, heamocorrection_type="Regression"):
    print("Processing Pixels")

    # Get Butterworth Filter Coefficients
    b, a, = get_filter_coefficients()

    # Load Data
    blue_matrix = h5py.File(blue_data, 'r')
    violet_matrix = h5py.File(violet_data, 'r')
    number_of_pixels = np.shape(blue_matrix["Data"])[0]
    number_of_images = np.shape(blue_matrix["Data"])[1]

    # Load Transformation Details
    transformation_details = load_transformation_details(home_directory)

    # Load Mask
    indicies, image_height, image_width = load_mask(home_directory, transformation_details)
    number_of_active_pixels = np.shape(indicies)[0]
    print("Number of active pixels", number_of_active_pixels)

    # Create Lists to Store Max and Min Values
    min_values = []
    max_values = []

    # Define Chunking Settings
    preferred_chunk_size = 20000
    number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = Widefield_General_Functions.get_chunk_structure(preferred_chunk_size, number_of_active_pixels)

    with h5py.File(output_file, "w") as f:
        dataset = f.create_dataset("Data", (number_of_images, number_of_active_pixels), dtype=np.float32, chunks=True, compression="gzip")

        for chunk_index in range(number_of_chunks):

            print("Chunk:", str(chunk_index).zfill(2), "of", number_of_chunks)
            chunk_start = int(chunk_starts[chunk_index])
            chunk_stop = int(chunk_stops[chunk_index])
            chunk_indicies = indicies[chunk_start:chunk_stop]

            # Load Data
            blue_data = blue_matrix["Data"][chunk_indicies]
            violet_data = violet_matrix["Data"][chunk_indicies]

            # Filter
            if bandpass_filter == True:
                blue_data, violet_data = perform_bandpass_filter(blue_data, violet_data, b, a)

            # Perform Heamocorrection
            if heamocorrection_type == "Regression":
                processed_data = heamocorrection_regression(blue_data, violet_data)

            elif heamocorrection_type == "Ratio":
                processed_data = heamocorrection_ratio(blue_data, violet_data)
            else:
                print("Invalid Heamocorrection Type!")

            # Free Up Ram
            blue_data = None
            violet_data = None

            # Add To Min and Max Vectors
            chunk_min_vector = np.percentile(processed_data, axis=1, q=1)
            chunk_max_vector = np.percentile(processed_data, axis=1, q=99)
            for value in chunk_min_vector:
                min_values.append(value)
            for value in chunk_max_vector:
                max_values.append(value)

            # Calculate Delta F
            processed_data = calculate_delta_f(processed_data, chunk_min_vector)
            processed_data = normalise_delta_f(processed_data, chunk_max_vector, chunk_min_vector)

            # Add Data To Dataset
            print("Procesed data shape", np.shape(processed_data))
            processed_data = np.transpose(processed_data)
            dataset[:, chunk_start:chunk_stop] = processed_data

    # Save Min and Max Vectors
    min_vector = np.array(min_values)
    max_vector = np.array(max_values)
    np.save(home_directory + "/Pixel_Baseline_Values.npy", min_vector)
    np.save(home_directory + "/Pixel_Max_Values.npy", max_vector)

    # Save Session Indicies
    np.save(os.path.join(home_directory, "session_indicies.npy"), indicies)

    print("Finished processing pixels")


def perform_heamocorrection(home_directory):

    # Preprocessing Variables
    bandpass_filter = True
    heamocorrection_type = "Regression"

    # Assign File Locations
    blue_file                   = get_blue_file(home_directory)
    violet_file                 = get_violet_file(home_directory)
    reconstructed_video_file    = os.path.join(home_directory, "Greyscale_Reconstruction.avi")
    delta_f_unregistered_file   = os.path.join(home_directory, "Delta_F_Unregistered.hdf5")
    delta_f_registered_file     = os.path.join(home_directory, "Delta_F_Registered.hdf5")

    #Load Data
    blue_data_container = h5py.File(blue_file, 'r')
    blue_data           = blue_data_container["Data"]

    violet_data_container = h5py.File(violet_file, 'r')
    violet_data           = violet_data_container["Data"]

    # Reconstruct Greyscale Video
    print("reconstructed video file", reconstructed_video_file)
    Reconstruct_Raw_Video.reconstruct_raw_video(blue_data, violet_data, reconstructed_video_file)

    # Extract Signal
    print("Processing Signal")
    process_pixels(blue_file, violet_file, delta_f_unregistered_file, home_directory, bandpass_filter=bandpass_filter, heamocorrection_type=heamocorrection_type)

    # Register Delta F
    print("Registering Delta F")
    Register_Delta_F.register_delta_f(base_directory, delta_f_registered_file)

    # Create Sample Video
    print("Creating Sample Video")
    Create_Sample_Video.create_sample_video(delta_f_registered_file, home_directory, blur_size=2)


base_directory = "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_12_Transition_Imaging"
perform_heamocorrection(base_directory)