import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import sys
sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions


# Load Original File

def get_output_directory(base_directory, output_stem):

    split_base_directory = base_directory.split("/")

    # Check Mouse Directory
    mouse_directory = os.path.join(output_stem, split_base_directory[-2])
    if not os.path.exists(mouse_directory):
        os.mkdir(mouse_directory)

    # Check Session Directory
    session_directory = os.path.join(mouse_directory, split_base_directory[-1])
    if not os.path.exists(session_directory):
        os.mkdir(session_directory)

    return session_directory



output_stem = "/media/matthew/Expansion/Widefield_Analysis"

# Get Mouse Name
def get_session_name(base_directory):

    # Split File Path By Forward Slash
    split_base_directory = base_directory.split("/")

    # Take The Last Two and Join By Underscore
    session_name = split_base_directory[-2] + "_" + split_base_directory[-1]

    return session_name

 # Split Into Two Seperate Files


def split_motion_corrected_data(base_directory, session_name):

    # Load Combined File
    combined_file = os.path.join(base_directory, "Motion_Corrected_Mask_Data.hdf5")
    combined_file_container = h5py.File(combined_file, 'r')
    combined_blue_data = combined_file_container["Blue_Data"]
    combined_violet_data = combined_file_container["Violet_Data"]
    print("Blue data shape", np.shape(combined_blue_data))

    # Get Chunk Structure
    number_of_pixels, number_of_frames = np.shape(combined_blue_data)
    preferred_chunk_size = 10000
    number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = Widefield_General_Functions.get_chunk_structure(preferred_chunk_size, number_of_pixels)
    print("Chunk sizes", chunk_sizes)

    # Get New File Names
    blue_file_name = session_name + "_Motion_Corrected_Masked_Blue_Data.hdf5"
    violet_file_name = session_name + "_Motion_Corrected_Masked_Violet_Data.hdf5"

    file_cache_size = 16561440000

    # Process Data
    with h5py.File(os.path.join(base_directory, blue_file_name), "w", rdcc_nbytes=file_cache_size) as seperate_blue_file:
        with h5py.File(os.path.join(base_directory, violet_file_name), "w", rdcc_nbytes=file_cache_size) as seperate_violet_file:

            #chunks=(preferred_chunk_size, number_of_frames),
            seperate_blue_data = seperate_blue_file.create_dataset("Data", (number_of_pixels, number_of_frames), dtype=np.uint16, compression="gzip", chunks=(preferred_chunk_size*2, number_of_frames))
            seperate_violet_data = seperate_violet_file.create_dataset("Data", (number_of_pixels, number_of_frames), dtype=np.uint16, compression="gzip", chunks=(preferred_chunk_size*2, number_of_frames))
    
            for chunk_index in range(number_of_chunks):
                print("Chunk Index", chunk_index, " of ", number_of_chunks, "Time: ", datetime.now())
                chunk_start = int(chunk_starts[chunk_index])
                chunk_stop = int(chunk_stops[chunk_index])

                data_chunk = combined_blue_data[chunk_start:chunk_stop]
                seperate_blue_data[chunk_start:chunk_stop] = data_chunk

                data_chunk = combined_violet_data[chunk_start:chunk_stop]
                seperate_violet_data[chunk_start:chunk_stop] = data_chunk



def plot_registration_shifts(base_directory):
    print("Plotting registration", base_directory)

    # Load Data
    x_shifts = np.load(os.path.join(base_directory, "X_Shifts.npy"))
    y_shifts = np.load(os.path.join(base_directory, "Y_Shifts.npy"))
    r_shifts = np.load(os.path.join(base_directory, "R_Shifts.npy"))

    # Create Figure
    figure_1 = plt.figure()
    rows = 1
    columns = 2
    translation_axis = figure_1.add_subplot(rows, columns, 1)
    rotation_axis = figure_1.add_subplot(rows, columns, 2)

    # Plot Data
    translation_axis.plot(x_shifts, c='b')
    translation_axis.plot(y_shifts, c='r')
    rotation_axis.plot(r_shifts, c='g')

    # Save Figure
    plt.savefig(os.path.join(base_directory, "Motion_Correction_Shifts.png"))
    plt.close()




session_list = [
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_13_Transition_Imaging"]

"""
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_15_Transition_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging",

    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_06_Transition_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging",

    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK20.1B/2021_11_22_Transition_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK20.1B/2021_11_24_Transition_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK20.1B/2021_11_26_Transition_Imaging",

    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK22.1A/2021_10_29_Transition_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK22.1A/2021_11_03_Transition_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK22.1A/2021_11_05_Transition_Imaging",

    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_10_Transition_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_12_Transition_Imaging",

    "/media/matthew/Seagate Expansion Drive1/Longitudinal_Analysis/NXAK4.1B/2021_02_04_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Switching_Analysis/Selected_sessions/NXAK16.1B/2021_06_23_Switching_Imaging",

    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Switching_Analysis/Selected_sessions/NRXN71.2A/2020_12_17_Switching_Imaging_Querty_not_yet_registered",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Switching_Analysis/Selected_sessions/NXAK16.1B/2021_06_23_Switching_Imaging",

    "/media/matthew/Seagate Expansion Drive/Switching_Analysis/Homs/NXAK10.1A/2021_05_20_Switching_Imaging",

    "/media/matthew/Seagate Expansion Drive/Switching_Analysis/Homs/NXAK24.1C/2021_10_14_Switching_Imaging",
    "/media/matthew/Seagate Expansion Drive/Switching_Analysis/Homs/NXAK24.1C/2021_11_10_Transition_Imaging",
 ]
"""



for session in session_list:

    # Get Output Directory
    output_directory = get_output_directory(session, output_stem)

    # Get Session Name
    session_name = get_session_name(output_directory)

    # Split Motion Corrected Data
    split_motion_corrected_data(output_directory, session_name)

    # Plot Registration Shifts
    plot_registration_shifts(output_directory)