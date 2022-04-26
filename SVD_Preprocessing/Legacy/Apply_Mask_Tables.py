import numpy as np
import os
import h5py
import sys
import datetime

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions


def mask_file(data_file, mask_indicies, output_file_name):

    # Load Delta F Data
    downsampled_file_object = h5py.File(data_file, 'r')
    data_matrix = downsampled_file_object["Data"]

    # Create Output File
    number_of_masked_pixels = len(mask_indicies)
    number_of_frames = np.shape(data_matrix)[1]
    print("number of masked pixels", number_of_masked_pixels)
    print("number of timepoints", number_of_frames)

    # Define Chunking Settings
    preferred_chunk_size = 20000
    number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = Widefield_General_Functions.get_chunk_structure(preferred_chunk_size, number_of_masked_pixels)

    with h5py.File(output_file_name, "w") as f:
        dataset = f.create_dataset("Data", (number_of_frames, number_of_masked_pixels), dtype=np.float32, chunks=True, compression="gzip")

        for chunk_index in range(number_of_chunks):
            print("Chunk Index", chunk_index, " of ", number_of_chunks, "Time: ", datetime.datetime.now())
            chunk_start = int(chunk_starts[chunk_index])
            chunk_stop = int(chunk_stops[chunk_index])

            chunk_indicies = mask_indicies[chunk_start:chunk_stop]
            chunk_data = data_matrix[chunk_indicies]
            chunk_data = np.transpose(chunk_data)

            # Remove NaNs
            chunk_data = np.nan_to_num(chunk_data)

            # Ensure We Are Never dividing By Zero
            chunk_data = np.add(chunk_data, 0.000001)

            dataset[:, chunk_start:chunk_stop] = chunk_data



def apply_mask(base_directory):

    # Load Generous Mask
    generous_mask = np.load(os.path.join(base_directory, "Generous_Mask.npy"))

    # Mask File
    mask_file(violet_file,  mask_indicies, os.path.join(base_directory, "Masked_Violet_Data.hdf5"))


base_directory = r"/media/matthew/29D46574463D2856/NXAK14.1A_2021_06_15_Transition_Imaging"
apply_mask(base_directory)