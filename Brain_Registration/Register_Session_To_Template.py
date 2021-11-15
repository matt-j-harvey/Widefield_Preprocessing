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

import sys



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

def get_chunk_structure(chunk_size, array_size):
    number_of_chunks = int(np.ceil(array_size / chunk_size))
    remainder = array_size % chunk_size

    # Get Chunk Sizes
    chunk_sizes = []
    if remainder == 0:
        for x in range(number_of_chunks):
            chunk_sizes.append(chunk_size)

    else:
        for x in range(number_of_chunks - 1):
            chunk_sizes.append(chunk_size)
        chunk_sizes.append(remainder)

    # Get Chunk Starts
    chunk_starts = []
    chunk_start = 0
    for chunk_index in range(number_of_chunks):
        chunk_starts.append(chunk_size * chunk_index)

    # Get Chunk Stops
    chunk_stops = []
    chunk_stop = 0
    for chunk_index in range(number_of_chunks):
        chunk_stop += chunk_sizes[chunk_index]
        chunk_stops.append(chunk_stop)

    return number_of_chunks, chunk_sizes, chunk_starts, chunk_stops





def tansform_data(data, transformation, save_directory):

    # Get Data Shape
    height = 600
    width = 608
    number_of_pixels = np.shape(data)[0]
    number_of_frames = np.shape(data)[1]

    # Load Transformation Details
    angle = transformation["rotation"]
    y_shift = int(transformation["y_shift"])
    x_shift = int(transformation["x_shift"])

    print("Angle", angle)
    print("Y Shift", y_shift)
    print("X shift", x_shift)

    # Get Chunking Structure
    preferred_chunk_size = 20000
    number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = get_chunk_structure(preferred_chunk_size, number_of_frames)
    print("Number of chunks", number_of_chunks)

    # Create Save File
    with h5py.File(save_directory, "w") as f:
        dataset = f.create_dataset("Data", (number_of_pixels, number_of_frames), dtype=np.uint16, chunks=True, compression="gzip")

        for chunk in range(number_of_chunks):
            print("Chunk: ", chunk, " of ", number_of_chunks)
            chunk_size = chunk_sizes[chunk]
            chunk_start = chunk_starts[chunk]
            chunk_stop = chunk_stops[chunk]

            # Reshape Data
            data_chunk = data[:, chunk_start:chunk_stop]
            data_chunk = np.ndarray.reshape(data_chunk, (height, width, chunk_size))

            # Rotate
            data_chunk = ndimage.rotate(data_chunk, angle, reshape=False)

            # Translate
            data_chunk = np.roll(a=data_chunk, axis=0, shift=y_shift)
            data_chunk = np.roll(a=data_chunk, axis=1, shift=x_shift)

            # Reshape
            data_chunk = np.ndarray.reshape(data_chunk, (height * width, chunk_size))

            # Add To Matrix
            dataset[:, chunk_start:chunk_stop] = data_chunk




def align_sessions(session_list):

    number_of_sessions = len(session_list)
    for session_index in range(number_of_sessions):

        # Get Session Name
        session_directory = session_list[session_index]
        session = session_directory.split('/')[-1]
        print(session)

        # Load Transformation
        transformation_file = session_directory + "/Transformation_Dictionary.npy"
        transformation_data = np.load(transformation_file, allow_pickle=True)
        transformation_data = transformation_data[()]

        # Align Blue Data
        blue_data_file_location = get_blue_file(session_directory)
        blue_data_file = h5py.File(blue_data_file_location, 'r')
        blue_data = blue_data_file["Data"]
        output_file = session_directory + "/Blue_Data_Registered.hdf5"
        tansform_data(blue_data, transformation_data, output_file)

        # Align Violet Data
        violet_data_file_location = get_violet_file(session_directory)
        violet_data_file = h5py.File(violet_data_file_location, 'r')
        violet_data = violet_data_file["Data"]
        output_file = session_directory + "/Violet_Data_Registered.hdf5"
        tansform_data(violet_data, transformation_data, output_file)



"""
Done
"/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging/",
"/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging/",
"/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK10.1A/2021_06_18_Transition_Imaging/"
"/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_04_02_Transition_Imaging"
"/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_12_Transition_Imaging/"
"/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK12.1F/2021_09_22_Transition_Imaging"
"""


session_list = ["/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NRXN78.1A/2020_12_09_Switching_Imaging"]

align_sessions(session_list)