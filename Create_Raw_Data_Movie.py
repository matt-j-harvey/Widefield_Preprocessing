import os
import h5py
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def get_blue_filename(base_directory):
    file_list = os.listdir(base_directory)
    for file_name in file_list:
        if "Blue_Data.hdf5" in file_name:
            return file_name

def get_violet_filename(base_directory):
    file_list = os.listdir(base_directory)
    for file_name in file_list:
        if "Violet_Data.hdf5" in file_name:
            return file_name

def visualise_raw_data(base_directory, sample_start, sample_stop):

    # Get Filenames
    blue_file = os.path.join(base_directory, get_blue_filename(base_directory))
    violet_file = os.path.join(base_directory, get_violet_filename(base_directory))

    # Open Files
    blue_df_file_container = h5py.File(blue_file, 'r')
    blue_matrix = blue_df_file_container["Data"]

    violet_df_file_container = h5py.File(violet_file, 'r')
    violet_matrix = violet_df_file_container["Data"]

    # Get Data Structure
    number_of_pixels, number_of_images = np.shape(blue_matrix)

    # Extract Data Sample
    blue_sample_data = blue_matrix[:, sample_start:sample_stop]
    violet_sample_data = violet_matrix[:, sample_start:sample_stop]

    # Create Video
    frame_width = 608
    frame_height = 600
    video_filename = os.path.join(base_directory, "Raw_Data_Video_" + str(sample_start) + "_to_" + str(sample_stop) + ".avi")
    video_codec = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(video_filename, video_codec, frameSize=(frame_width * 2, frame_height), fps=30)  # 0, 12

    sample_size = sample_stop - sample_start
    for frame in tqdm(range(sample_size)):

        blue_frame = blue_sample_data[:, frame]
        violet_frame = violet_sample_data[:, frame]

        # Convert To UInt8
        blue_frame = np.divide(blue_frame, 65535)
        blue_frame = np.multiply(blue_frame, 255)
        blue_frame = np.ndarray.astype(blue_frame, np.uint8)

        violet_frame = np.divide(violet_frame, 65535)
        violet_frame = np.multiply(violet_frame, 255)
        violet_frame = np.ndarray.astype(violet_frame, np.uint8)

        blue_frame = np.reshape(blue_frame, (600, 608))
        violet_frame = np.reshape(violet_frame, (600, 608))

        image = np.hstack((violet_frame, blue_frame))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        video.write(image)

    cv2.destroyAllWindows()
    video.release()

visualise_raw_data(r"/media/matthew/External_Harddrive_1/Thesis_mice/STR001" ,20000, 21000)
base_directory = r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Control_Data/NRXN78.1D/2020_12_05_Switching_Imaging/Downsampled_Data"