import numpy as np
import os
import h5py
import sys
import datetime
import matplotlib.pyplot as plt
import tables

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions


def apply_mask(base_directory):

    # Load Generous Mask
    generous_mask = np.load(os.path.join(base_directory, "Generous_Mask.npy"))

    # Transpose Mask
    mask = np.transpose(generous_mask)

    # Load Data File
    data_matrix_file = tables.open_file(os.path.join(base_directory, "Interleaved_Tables_Motion_Correction.h5"), mode='r+')
    data_matrix = data_matrix_file.root.Data
    print("Data Matrix Shaoe", np.shape(data_matrix))

    # Get Data Structure
    number_of_frames, number_of_channels, image_width, image_height = np.shape(data_matrix)

    # Define Chunking Settings
    preferred_chunk_size = 15000
    number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = Widefield_General_Functions.get_chunk_structure(preferred_chunk_size, number_of_frames)

    for chunk_index in range(number_of_chunks):

        print("Chunk Index", chunk_index, " of ", number_of_chunks, "Time: ", datetime.datetime.now())
        chunk_start = int(chunk_starts[chunk_index])
        chunk_stop = int(chunk_stops[chunk_index])

        # Load Chunk Data
        chunk_data = data_matrix[chunk_start:chunk_stop]

        # Apply Mask
        chunk_data = np.multiply(chunk_data, mask)

        # Put Back
        data_matrix[chunk_start:chunk_stop] = chunk_data


base_directory = r"/media/matthew/29D46574463D2856/NXAK14.1A_2021_06_15_Transition_Imaging"
apply_mask(base_directory)