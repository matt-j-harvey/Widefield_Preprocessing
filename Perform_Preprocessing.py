import os
from datetime import datetime

import Position_Mask
import Get_Max_Projection
import Motion_Correction
import Heamocorrection
import Get_Baseline_Frames

import matplotlib.pyplot as plt
import numpy as np
import h5py

"""
1.) Get Max Projection
2.) Assign Generous Mask
3.) Motion Correction
4.) Heamocorrection
5.) SVD
"""


def get_file_names(base_directory):

    file_list = os.listdir(base_directory)
    blue_file = None
    violet_file = None

    for file in file_list:
        if "Blue_Data" in file:
            blue_file = file
        elif "Violet_Data" in file:
            violet_file = file

    return blue_file, violet_file


def check_led_colours(base_directory):

    blue_file_name, violet_file_name = get_file_names(base_directory)

    # Load Delta F File
    blue_filepath = os.path.join(base_directory, blue_file_name)
    violet_filepath = os.path.join(base_directory, violet_file_name)

    blue_data_container = h5py.File(blue_filepath, 'r')
    violet_data_container = h5py.File(violet_filepath, 'r')

    blue_array = blue_data_container["Data"]
    violet_array = violet_data_container["Data"]

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

def get_motion_corrected_data_filename(base_directory):

    file_list = os.listdir(base_directory)
    for file in file_list:
        if "Motion_Corrected_Mask_Data" in file:
            return file




# Complete## REPLACE ON SERVER
# r"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Processed_Widefield_Data/NRXN71.2A/2020_11_14_Discrimination_Imaging",
# r"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Processed_Widefield_Data/NRXN71.2A/2020_11_17_Discrimination_Imaging",
# r"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Processed_Widefield_Data/NRXN71.2A/2020_11_19_Discrimination_Imaging",
# r"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Processed_Widefield_Data/NRXN71.2A/2020_11_21_Discrimination_Imaging",
# r"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Processed_Widefield_Data/NRXN71.2A/2020_12_03_Discrimination_Imaging"
# r"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Processed_Widefield_Data/NRXN71.2A/2020_12_01_Discrimination_Imaging",
# r"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Processed_Widefield_Data/NRXN71.2A/2020_12_05_Discrimination_Imaging",
# r"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Processed_Widefield_Data/NRXN71.2A/2020_12_07_Discrimination_Imaging",
# r"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Processed_Widefield_Data/NRXN71.2A/2020_12_09_Discrimination_Imaging",

"""
    r"/media/matthew/External_Harddrive_1/Processed_Widefield_Data/Beverly/2022_05_16_Mirror_Imaging",
    r"/media/matthew/External_Harddrive_1/Processed_Widefield_Data/Beverly/2022_05_18_Mirror_Imaging",
    r"/media/matthew/External_Harddrive_1/Processed_Widefield_Data/Beverly/2022_05_23_mirror_imaging",
    r"/media/matthew/External_Harddrive_1/Processed_Widefield_Data/Beverly/2022_05_27_mirror_imaging",
    
    r"/media/matthew/External_Harddrive_1/Processed_Widefield_Data/NRXN78.1D/2020_11_29_Switching_Imaging"
    
    r"/media/matthew/External_Harddrive_1/Processed_Widefield_Data/NRXN78.1A/2020_12_05_Switching_Imaging"
    r"/media/matthew/External_Harddrive_1/Processed_Widefield_Data/NRXN78.1A/2020_12_09_Switching_Imaging"
    
"""


#    r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_26_Switching_Imaging", # Keep
#r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_02_28_Switching_Imaging",  # Keep
#r"/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_03_02_Switching_Imaging",  # Keep
#r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_21_Switching_Imaging",  # Keep
#r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_23_Switching_Imaging",  # Keep
#r"/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_06_11_Switching_Imaging"  # Keep

#r"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_04_Discrimination_Imaging",
#r"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_06_Discrimination_Imaging",
#r"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_08_Discrimination_Imaging",
#r"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_10_Discrimination_Imaging",
#r"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_12_Discrimination_Imaging",
#r"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_14_Discrimination_Imaging",
#r"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_22_Discrimination_Imaging",

#+r"/media/matthew/External_Harddrive_1/Processed_Widefield_Data/NRXN78.1A/2020_11_14_Discrimination_Imaging",  #

#r"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_03_02_Switching_Imaging", # Swap
#r"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_03_04_Switching_Imaging", # Keep
#r"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_03_06_Switching_Imaging", # Swap

#r"/media/matthew/External_Harddrive_1/Processed_Widefield_Data/NXAK22.1A/2021_10_14_Switching_Imaging", # Keep
#r"/media/matthew/External_Harddrive_1/Processed_Widefield_Data/NXAK22.1A/2021_10_20_Switching_Imaging",  # Keep
#r"/media/matthew/External_Harddrive_1/Processed_Widefield_Data/NXAK22.1A/2021_10_22_Switching_Imaging",  # Keep
#r"/media/matthew/Expansion/Widefield_Analysis/NRXN71.2A/2020_12_13_Switching_Imaging", # swap
#r"/media/matthew/Expansion/Widefield_Analysis/NRXN71.2A/2020_12_15_Switching_Imaging" # Swap

# r"/media/matthew/External_Harddrive_1/Processed_Widefield_Data/NXAK4.1A/2021_03_31_Switching_Imaging", # Keep
# r"/media/matthew/External_Harddrive_1/Processed_Widefield_Data/NXAK4.1A/2021_04_02_Switching_Imaging", # Keep
# r"/media/matthew/External_Harddrive_1/Processed_Widefield_Data/NXAK4.1A/2021_04_04_Switching_Imaging", # Keep

#    r"/media/matthew/Expansion/Widefield_Analysis/NXAK10.1A/2021_05_22_Switching_Imaging", # Keep
# r"/media/matthew/Expansion/Widefield_Analysis/NXAK10.1A/2021_05_24_Switching_Imaging",

#r"/media/matthew/Expansion/Widefield_Analysis/NXAK10.1A/2021_06_14_Transition_Imaging",  # Keep
#r"/media/matthew/Expansion/Widefield_Analysis/NXAK10.1A/2021_06_16_Transition_Imaging",  # Keep
#r"/media/matthew/Expansion/Widefield_Analysis/NXAK10.1A/2021_06_18_Transition_Imaging",  # keep

#r"/media/matthew/Expansion/Widefield_Analysis/NXAK16.1B/2021_06_17_Switching_Imaging",  # Keep
#r"/media/matthew/Expansion/Widefield_Analysis/NXAK16.1B/2021_06_19_Switching_Imaging",  # Keep
#r"/media/matthew/Expansion/Widefield_Analysis/NXAK16.1B/2021_06_30_Transition_Imaging"  # Keep
#r"   #r"/media/matthew/Expansion/Widefield_Analysis/NXAK24.1C/2021_10_20_Switching_Imaging",  # Keep

#r"/media/matthew/Expansion/Widefield_Analysis/NXAK24.1C/2021_10_26_Switching_Imaging",  # Keep
#r"/media/matthew/Expansion/Widefield_Analysis/NXAK24.1C/2021_11_08_Transition_Imaging",  # Keep

#r"/media/matthew/Expansion/Widefield_Analysis/NXAK20.1B/2021_11_15_Switching_Imaging",  # Keep
#r"/media/matthew/Expansion/Widefield_Analysis/NXAK20.1B/2021_11_17_Switching_Imaging",  # Keep
#r"/media/matthew/Expansion/Widefield_Analysis/NXAK20.1B/2021_11_19_Switching_Imaging",  # Keep
#r"/media/matthew/Expansion/Widefield_Analysis/NXAK20.1B/2021_11_24_Transition_Imaging",  # Keep
#r"/media/matthew/Expansion/Widefield_Analysis/NXAK20.1B/2021_11_26_Transition_Imaging",  # Keep


session_list = ["/media/matthew/External_Harddrive_1/Opto_Test/KVIP25.5H/2022_07_26_Opto_Test_No_Filter"]
session_list = ["/media/matthew/External_Harddrive_1/Opto_Test/Projector_Calib/2022_07_26_Calibration"]



session_list = [
r"/media/matthew/External_Harddrive_2/Opto_Test/Local_Injections/KPVB17.1E/2022_09_19_Opto_Test_No_Filter"
]

session_list = [r"/media/matthew/External_Harddrive_2/Opto_Test/Local_Injections/KPVB17.1E/2022_09_20_Opto_Test_Filter"]

session_list = [r"/media/matthew/External_Harddrive_2/Opto_Test/KPVB17.1f/2022_09_20_Opto_Test_Filter"]

number_of_sessions = len(session_list)

"""
# Check LED Colors
for base_directory in session_list:
    check_led_colours(base_directory)

# Get Max Projections
for session_index in range(number_of_sessions):
    base_directory = session_list[session_index]
    Get_Max_Projection.check_max_projection(base_directory, base_directory)

# Assign Masks
Position_Mask.position_mask(session_list, session_list)
"""

# Process Data
for session_index in range(number_of_sessions):

    base_directory = session_list[session_index]
    print("Session ", session_index, " of ", number_of_sessions, base_directory)

    # Perform Motion Correction
    print("Performing Motion Correction", datetime.now())
    Motion_Correction.perform_motion_correction(base_directory, base_directory)

    # Perform Heamocorrection
    print("Performing Heamocorrection", datetime.now())
    #Get_Baseline_Frames.get_baseline_frames(base_directory, base_directory)
    #Heamocorrection.perform_heamocorrection(base_directory, use_baseline_frames=False)

    # Perform SVD Compression
    print("Performing SVD Compression", datetime.now())
    #Blockwise_Approximate_SVD.perform_svd_compression(output_directory)
