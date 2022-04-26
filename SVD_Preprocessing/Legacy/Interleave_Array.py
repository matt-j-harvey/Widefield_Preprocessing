import h5py
import numpy as np
import os
import sys
import datetime
import matplotlib.pyplot as plt
import tables

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions



def interleave_array(base_directory, blue_filename, violet_filename):

    image_height = 600
    image_width = 608

    # Load Delta F Data
    blue_file_object = h5py.File(os.path.join(base_directory, blue_filename), 'r')
    violet_file_object = h5py.File(os.path.join(base_directory, violet_filename), 'r')

    blue_data_matrix = blue_file_object["Data"]
    violet_data_matrix = violet_file_object["Data"]

    number_of_pixels, number_of_frames = np.shape(blue_data_matrix)

    # Define Chunking Settings
    preferred_chunk_size = 5000
    number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = Widefield_General_Functions.get_chunk_structure(preferred_chunk_size, number_of_frames)

    #dat is a[nframes X nchannels X width X height] array

    output_file_name = os.path.join(base_directory, "Interleaved_Array.hdf5")
    with h5py.File(output_file_name, "w") as f:
        dataset = f.create_dataset("Data", (number_of_frames, 2, image_width, image_height), dtype=np.float32, chunks=True, compression="gzip")

        for chunk_index in range(number_of_chunks):
            print("Chunk Index", chunk_index, " of ", number_of_chunks, "Time: ", datetime.datetime.now())
            chunk_start = int(chunk_starts[chunk_index])
            chunk_stop = int(chunk_stops[chunk_index])
            chunk_size = int(chunk_sizes[chunk_index])

            blue_chunk = blue_data_matrix[:, chunk_start:chunk_stop]
            violet_chunk = violet_data_matrix[:, chunk_start:chunk_stop]

            blue_chunk = np.transpose(blue_chunk)
            violet_chunk = np.transpose(violet_chunk)

            blue_chunk = np.reshape(blue_chunk, (chunk_size, image_height, image_width))
            violet_chunk = np.reshape(violet_chunk, (chunk_size, image_height, image_width))

            blue_chunk = np.swapaxes(blue_chunk, 1, 2)
            violet_chunk = np.swapaxes(violet_chunk, 1, 2)

            dataset[chunk_start:chunk_stop, 0] = blue_chunk
            dataset[chunk_start:chunk_stop, 1] = violet_chunk


def get_average_frame(base_directory, blue_filename, violet_filename):

    # Load Delta F Data
    blue_file_object = h5py.File(os.path.join(base_directory, blue_filename), 'r')
    violet_file_object = h5py.File(os.path.join(base_directory, violet_filename), 'r')

    blue_data_matrix = blue_file_object["Data"]
    violet_data_matrix = violet_file_object["Data"]

    print(np.shape(blue_data_matrix))


    number_of_pixels, number_of_frames = np.shape(blue_data_matrix)

    # Define Chunking Settings
    preferred_chunk_size = 30000
    number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = Widefield_General_Functions.get_chunk_structure(preferred_chunk_size, number_of_pixels)

    blue_mean = []
    violet_mean = []

    for chunk_index in range(number_of_chunks):
        print("Chunk Index", chunk_index, " of ", number_of_chunks, "Time: ", datetime.datetime.now())
        chunk_start = int(chunk_starts[chunk_index])
        chunk_stop = int(chunk_stops[chunk_index])

        data_chunk = blue_data_matrix[chunk_start:chunk_stop]
        data_mean = np.mean(data_chunk, axis=1)
        for pixel in data_mean:
            blue_mean.append(pixel)

        data_chunk = violet_data_matrix[chunk_start:chunk_stop]
        data_mean = np.mean(data_chunk, axis=1)
        for pixel in data_mean:
            violet_mean.append(pixel)

    blue_mean = np.array(blue_mean)
    violet_mean = np.array(violet_mean)

    np.save(os.path.join(base_directory, "blue_mean.npy"), blue_mean)
    np.save(os.path.join(base_directory, "violet_mean.npy"), violet_mean)



def convert_to_tables(base_directory):


    #(number_of_frames, 2, image_width, image_height)

    # Load Delta F Data
    h5py_file_object = h5py.File(os.path.join(base_directory, "Interleaved_Array.hdf5"), 'r')
    h5py_data_matrix = h5py_file_object["Data"]
    number_of_frames, num_channels, image_width, image_height = np.shape(h5py_data_matrix)
    print("h5py matrix",  np.shape(h5py_data_matrix))

    # Create Tables File
    tables_file_path = os.path.join(base_directory, "Interleaved_Tables.h5")
    tables_file = tables.open_file(tables_file_path, mode='w')
    tables_storage = tables_file.create_earray(tables_file.root, 'Data', tables.Float32Atom(), shape=(0, num_channels, image_width, image_height), expectedrows=number_of_frames)

    # Define Chunking Settings
    preferred_chunk_size = 500
    number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = Widefield_General_Functions.get_chunk_structure(preferred_chunk_size, number_of_frames)

    for chunk_index in range(number_of_chunks):
        print("Chunk Index", chunk_index, " of ", number_of_chunks, "Time: ", datetime.datetime.now())
        chunk_start = int(chunk_starts[chunk_index])
        chunk_stop = int(chunk_stops[chunk_index])
        data_chunk = h5py_data_matrix[chunk_start:chunk_stop]

        for frame in data_chunk:
            tables_storage.append([frame])
        tables_storage.flush()



base_directory = "/media/matthew/29D46574463D2856/NXAK14.1A_2021_06_15_Transition_Imaging"
blue_file = "NXAK14.1A_20210615-135401_Blue_Data.hdf5"
violet_file = "NXAK14.1A_20210615-135401_Violet_Data.hdf5"
#interleave_array(base_directory, blue_file, violet_file)
#get_average_frame(base_directory, blue_file, violet_file)
convert_to_tables(base_directory)