import h5py
import numpy as np
import tables
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import Preprocessing_Utils


def convert_df_to_tables(base_directory, output_directory):

    # Load Processed Data
    delta_f_file_location = os.path.join(base_directory, "300_delta_f.hdf5")
    delta_f_file = h5py.File(delta_f_file_location, mode='r')
    processed_data = delta_f_file["Data"]
    number_of_frames, number_of_pixels = np.shape(processed_data)
    print("number_of_frames", number_of_frames)
    print("number_of_pixels", number_of_pixels)

    # Create Tables File
    output_file = os.path.join(output_directory, "Downsampled_Delta_F.h5")
    output_file_container = tables.open_file(output_file, mode="w")
    output_e_array = output_file_container.create_earray(output_file_container.root, 'Data', tables.Float32Atom(), shape=(0, number_of_pixels), expectedrows=number_of_frames)

    # Define Chunking Settings
    preferred_chunk_size = 30000
    number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = Preprocessing_Utils.get_chunk_structure(preferred_chunk_size, number_of_frames)

    for chunk_index in tqdm(range(number_of_chunks)):

        # Get Selected Indicies
        chunk_start = int(chunk_starts[chunk_index])
        chunk_stop = int(chunk_stops[chunk_index])

        chunk_data = processed_data[chunk_start:chunk_stop]

        for frame in chunk_data:
            output_e_array.append([frame])

        output_file_container.flush()

    delta_f_file.close()
    output_file_container.close()

session_list = [

    #r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_03_23_Transition_Imaging",
    #r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_03_31_Transition_Imaging",
    #r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_04_02_Transition_Imaging",

    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK4.1B/2021_04_02_Transition_Imaging",
    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK4.1B/2021_04_08_Transition_Imaging",
    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK4.1B/2021_04_10_Transition_Imaging",

    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK14.1A/2021_06_13_Transition_Imaging",
    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK14.1A/2021_06_15_Transition_Imaging",
    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK14.1A/2021_06_17_Transition_Imaging",

    #r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK22.1A/2021_10_29_Transition_Imaging",
    r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK22.1A/2021_11_03_Transition_Imaging",
    r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK22.1A/2021_11_05_Transition_Imaging"

]


session_list = [r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Retinotopy/NXAK16.1B/2021_07_26_Continous_Retinotopy_Left",
               r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Retinotopy/NXAK16.1B/2021_07_27_Continous_Retinotopy_Right"]

for base_directory in session_list:
    convert_df_to_tables(base_directory, base_directory)