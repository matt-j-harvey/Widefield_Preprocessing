import os
import cv2
import numpy as np
from skimage.transform import AffineTransform, downscale_local_mean
import tables
from glob import glob
from os.path import join as pjoin
from datetime import datetime
from skimage.transform import warp
from multiprocessing import Pool, cpu_count
from functools import partial
from scipy.interpolate import interp1d
from scipy.sparse import load_npz, issparse,csr_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
import sys
import math
from skimage.transform import resize

import Preprocessing_Utils

cv2.setNumThreads(10)
def get_motion_corrected_data_filename(base_directory):

    file_list = os.listdir(base_directory)
    for file in file_list:
        if "Motion_Corrected_Mask_Data" in file:
            return file


def load_downsampled_mask(base_directory):

    mask = np.load(os.path.join(base_directory, "Generous_Mask.npy"))

    # Transform Mask
    mask = resize(mask, (300, 304), preserve_range=True, order=0, anti_aliasing=True)

    image_height = np.shape(mask)[0]
    image_width = np.shape(mask)[1]

    mask = np.where(mask > 0.1, 1, 0)
    mask = mask.astype(int)
    flat_mask = np.ndarray.flatten(mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    return indicies, image_height, image_width


def create_affine_matrix_from_transformations(x_shift, y_shift, rotation):
    affine = [[math.cos(rotation), -math.sin(rotation), x_shift],
              [math.sin(rotation),  math.cos(rotation), y_shift]]
    affine = np.array(affine)
    #affine = np.reshape(affine, (3,3))
    return affine


def reconstruct_chunk(data, full_indicies, full_image_height, full_image_width, downsample_indicies, downsample_image_height, downsample_image_width):

    data = np.transpose(data)

    downsampled_data = []
    number_of_frames = np.shape(data)[0]
    for frame_index in range(number_of_frames):

        # Extract Frame Data
        frame = data[frame_index]

        # Reconstrcut Frame
        frame = Preprocessing_Utils.create_image_from_data(frame, full_indicies, full_image_height, full_image_width)

        # Downsample
        frame = downscale_local_mean(image=frame, factors=(2,2))

        # Flatten
        frame = np.reshape(frame, (downsample_image_width * downsample_image_height))

        # Take Mased Portion
        frame = frame[downsample_indicies]

        # Add To Matrix
        downsampled_data.append(frame)

    downsampled_data = np.array(downsampled_data, dtype=np.uint16)
    downsampled_data = np.transpose(downsampled_data)
    return downsampled_data

def view_greyscale_sample(base_directory):

    print("Reconstructing Sample Video For Session", base_directory)

    # Load Data
    motion_corrected_data_file = "Motion_Corrected_Downsampled_Data.hdf5"
    data_file = os.path.join(base_directory, motion_corrected_data_file)
    data_container = h5py.File(data_file, 'r')
    blue_array = data_container["Blue_Data"]
    violet_array = data_container["Violet_Data"]

    # Take Sample of Data
    blue_array = blue_array[:, 1000:2000]
    violet_array = violet_array[:, 1000:2000]

    blue_array = np.transpose(blue_array)
    violet_array = np.transpose(violet_array)

    # Convert From 16 bit to 8 bit
    blue_array = np.divide(blue_array, 65536)
    violet_array = np.divide(violet_array, 65536)

    blue_array = np.multiply(blue_array, 255)
    violet_array = np.multiply(violet_array, 255)

    # Load Mask
    indicies, frame_height, frame_width = load_downsampled_mask(base_directory)

    # Create Video File
    reconstructed_video_file = os.path.join(base_directory, "Downsampled_Greyscale_Reconstruction.avi")
    video_name = reconstructed_video_file
    video_codec = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(video_name, video_codec, frameSize=(frame_width * 2, frame_height), fps=30)  # 0, 12

    number_of_frames = np.shape(blue_array)[0]

    for frame in range(number_of_frames):
        blue_template = np.zeros(frame_height * frame_width)
        violet_template = np.zeros(frame_height * frame_width)

        blue_frame = blue_array[frame]
        violet_frame = violet_array[frame]

        blue_template[indicies] = blue_frame
        violet_template[indicies] = violet_frame

        blue_template = np.ndarray.astype(blue_template, np.uint8)
        violet_template = np.ndarray.astype(violet_template, np.uint8)

        blue_frame = np.reshape(blue_template, (frame_height, frame_width))
        violet_frame = np.reshape(violet_template, (frame_height, frame_width))

        image = np.hstack((violet_frame, blue_frame))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        video.write(image)

    cv2.destroyAllWindows()
    video.release()

def downsample_session(base_directory, output_directory, output_file_name="Motion_Corrected_Downsampled_Data.hdf5"):

    # Load Data
    motion_corrected_filename = get_motion_corrected_data_filename(base_directory)
    motion_corrected_file = os.path.join(base_directory, motion_corrected_filename)
    motion_corrected_data_container = h5py.File(motion_corrected_file, 'r')
    blue_matrix = motion_corrected_data_container["Blue_Data"]
    violet_matrix = motion_corrected_data_container["Violet_Data"]

    # Load Downsampled Mask
    downsampled_indicies, downsampled_image_height, downsampled_image_width = load_downsampled_mask(base_directory)
    downsampled_pixels = len(downsampled_indicies)
    print("Downsampled Pixels", downsampled_pixels)

    # Load Full Mask
    full_indicies, full_image_height, full_image_width = Preprocessing_Utils.load_generous_mask(base_directory)

    # Define Chunking Settings
    preferred_chunk_size = 20000
    number_of_pixels, number_of_frames = np.shape(blue_matrix)
    print("Nuber Of Frames", number_of_frames, "Number Of Pixels", number_of_pixels)
    number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = Preprocessing_Utils.get_chunk_structure(preferred_chunk_size, number_of_frames)

    print("Heamocorrecting")
    with h5py.File(os.path.join(output_directory, output_file_name), "w") as f:
       downsampled_blue_data_container = f.create_dataset("Blue_Data", (downsampled_pixels, number_of_frames), dtype=np.uint16, chunks=True, compression=True)
       downsampled_violet_data_container = f.create_dataset("Violet_Data", (downsampled_pixels, number_of_frames), dtype=np.uint16, chunks=True, compression=True)

       for chunk_index in tqdm(range(number_of_chunks)):

           # Get Chunk Details
           chunk_start = int(chunk_starts[chunk_index])
           chunk_stop = int(chunk_stops[chunk_index])

            # Transform and Downsample Blue Chunk
           blue_chunk = blue_matrix[:, chunk_start:chunk_stop]
           blue_chunk = reconstruct_chunk(blue_chunk, full_indicies, full_image_height, full_image_width, downsampled_indicies, downsampled_image_height, downsampled_image_width)

            # Write Blue Chunk To Disk
           downsampled_blue_data_container[:, chunk_start:chunk_stop] = blue_chunk
           blue_chunk = None

            # Transform and Downsample Violet Chunk
           violet_chunk = violet_matrix[:, chunk_start:chunk_stop]
           violet_chunk = reconstruct_chunk(violet_chunk, full_indicies, full_image_height, full_image_width, downsampled_indicies, downsampled_image_height, downsampled_image_width)

            # Write Blue Chunk To Disk
           downsampled_violet_data_container[:, chunk_start:chunk_stop] = violet_chunk
           violet_chunk = None

    view_greyscale_sample(output_directory)

"""
session_list = [r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Retinotopy/NXAK16.1B/2021_07_26_Continous_Retinotopy_Left",
                r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Retinotopy/NXAK16.1B/2021_07_27_Continous_Retinotopy_Right"]
for base_directory in session_list:
    downsample_session(base_directory, base_directory)
"""