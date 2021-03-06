import os
from datetime import datetime

import Position_Mask
import Get_Max_Projection
import Motion_Correction_Adapted
import Heamocorrection_V2
import Perform_SVD_Compression_Incremental
import Get_Baseline_Frames

import matplotlib.pyplot as plt
import numpy as np

"""
1.) Get Max Projection
2.) Assign Generous Mask
3.) Motion Correction
4.) Heamocorrection
5.) SVD
"""




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



# Have Returned Motion Corrected Data
"""
    "/media/matthew/Seagate Expansion Drive1/Learning_Analysis/NXAK20.1B/2021_09_28_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Learning_Analysis/NXAK20.1B/2021_09_30_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Learning_Analysis/NXAK20.1B/2021_10_02_Discrimination_Imaging",
    
"""
# Not Yet Deleted Delta F



# To Add
"""
"/media/matthew/Seagate Expansion Drive1/Learning_Analysis/NRXN78.1D/2020_11_16_Discrimination_Imaging",
"/media/matthew/Seagate Expansion Drive1/Learning_Analysis/NRXN78.1D/2020_11_17_Discrimination_Imaging",
"/media/matthew/Seagate Expansion Drive1/Learning_Analysis/NRXN78.1D/2020_11_19_Discrimination_Imaging",
"/media/matthew/Seagate Expansion Drive1/Learning_Analysis/NRXN78.1D/2020_11_21_Discrimination_Imaging",
"/media/matthew/Seagate Expansion Drive1/Learning_Analysis/NRXN78.1D/2020_11_23_Discrimination_Imaging",
"/media/matthew/Seagate Expansion Drive1/Learning_Analysis/NRXN78.1D/2020_11_25_Discrimination_Imaging",
"""


# Duplicates
#/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Switching_Analysis/Selected_sessions/NRXN71.2A/2020_12_15_Switching_Imaging

# Errors
#"/media/matthew/Seagate Expansion Drive/Switching_Analysis/Homs/NRXN71.2A/2020_12_15_Switching_Imaging",

# SVD Error - ?Value too large

# Have Had Delta F Calculated
"""    
    "/media/matthew/Seagate Expansion Drive1/Learning_Analysis/NXAK20.1B/2021_10_04_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Learning_Analysis/NXAK20.1B/2021_10_06_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Learning_Analysis/NXAK20.1B/2021_10_09_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Learning_Analysis/NXAK20.1B/2021_10_11_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Learning_Analysis/NXAK20.1B/2021_10_13_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Learning_Analysis/NXAK20.1B/2021_10_15_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Learning_Analysis/NXAK20.1B/2021_10_17_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive1/Learning_Analysis/NXAK20.1B/2021_10_19_Discrimination_Imaging",


    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_06_Transition_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging",

    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK20.1B/2021_11_22_Transition_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK20.1B/2021_11_26_Transition_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK20.1B/2021_11_24_Transition_Imaging",
    
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_08_Transition_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_10_Transition_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_12_Transition_Imaging",

    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK24.1C/2021_11_05_Transition_Imaging",
    "/media/matthew/Seagate Expansion Drive/Switching_Analysis/Homs/NXAK24.1C/2021_11_10_Transition_Imaging",

    "/media/matthew/Seagate Expansion Drive1/Longitudinal_Analysis/NXAK4.1B/2021_02_04_Discrimination_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Switching_Analysis/Selected_sessions/NXAK16.1B/2021_06_23_Switching_Imaging",

    r"/media/matthew/Expansion/Widefield_Analysis/KGCA7.5A/2022_05_03_Switching_Imaging_CNO"

    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Switching_Analysis/Selected_sessions/NRXN71.2A/2020_12_17_Switching_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Switching_Analysis/Selected_sessions/NXAK16.1B/2021_06_23_Switching_Imaging",
    "/media/matthew/Seagate Expansion Drive/Switching_Analysis/Homs/NXAK10.1A/2021_05_20_Switching_Imaging",
    "/media/matthew/Seagate Expansion Drive/Switching_Analysis/Homs/NXAK24.1C/2021_10_14_Switching_Imaging",
        
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_04_02_Transition_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_04_08_Transition_Imaging",
    r"/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging",
    
    "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_03_23_Transition_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_03_31_Transition_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK7.1B/2021_04_02_Transition_Imaging",
    
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_13_Transition_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_15_Transition_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging",
    
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK22.1A/2021_10_29_Transition_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK22.1A/2021_11_03_Transition_Imaging",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK22.1A/2021_11_05_Transition_Imaging",
    
    "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Switching_Analysis/Selected_sessions/NRXN78.1D/2020_11_29_Switching_Imaging",
    "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Switching_Analysis/Selected_sessions/NRXN78.1D/2020_12_07_Switching_Imaging",
    r"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Switching_Analysis/Selected_sessions/NRXN78.1D/2020_12_05_Switching_Imaging",

"""


# nvert Blue
# "/media/matthew/Seagate Expansion Drive1/Learning_Analysis/NRXN78.1D/2020_11_15_Discrimination_Imaging",
# "/media/matthew/Seagate Expansion Drive1/Learning_Analysis/NRXN78.1D/2020_11_14_Discrimination_Imaging",

"""

"""


session_list = [
    "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Opto_Test/KCVP_1.1H/2022_05_09_Opto_Test_Filter"
]

#, - is motion corrected, needs heamocorrection

number_of_sessions = len(session_list)


# Check Output Directories
output_directory_list = []
for base_directory in session_list:
    output_directory = get_output_directory(base_directory, output_stem)
    output_directory_list.append(output_directory)

"""
# Get Max Projections
for session_index in range(number_of_sessions):
    base_directory = session_list[session_index]
    output_directory = output_directory_list[session_index]
    Get_Max_Projection.check_max_projection(base_directory, output_directory)
"""

# Assign Masks
#Position_Mask.position_mask(session_list, output_directory_list)

# Process Data
number_of_sessions = len(session_list)
for session_index in range(number_of_sessions):
    print("Session ", session_index, " of ", number_of_sessions)

    base_directory = session_list[session_index]
    output_directory = output_directory_list[session_index]

    # Perform Motion Correction
    print("Performing Motion Correction", datetime.now())
    Motion_Correction_Adapted.perform_motion_correction(base_directory, output_directory)
    
    # Perform Heamocorrection
    print("Performing Heamocorrection", datetime.now())
    #Get_Baseline_Frames.get_baseline_frames(base_directory, output_directory)
    Heamocorrection_V2.perform_heamocorrection(output_directory, use_baseline_frames=False)
    
    # Perform SVD Compression
    #print("Performing SVD Compression", datetime.now())
    #Perform_SVD_Compression_Incremental.perform_svd_compression(output_directory)

