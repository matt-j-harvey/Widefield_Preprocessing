import numpy as np
import matplotlib.pyplot as plt
import h5py
import tables
from scipy import signal, ndimage, stats
import os
import cv2
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA, FactorAnalysis, TruncatedSVD
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import shutil

import Preprocessing_Utils
import Heamocorrection


def get_motion_corrected_data_filename(base_directory):
    file_list = os.listdir(base_directory)
    for file_name in file_list:
        if "Motion_Corrected" in file_name:
            return file_name

def get_example_images(base_directory, output_directory, default_position=10000):

    # Load Motion Corrected Data
    motion_corrected_filename = get_motion_corrected_data_filename(base_directory)
    motion_corrected_file = os.path.join(base_directory, motion_corrected_filename)
    motion_corrected_data_container = h5py.File(motion_corrected_file, 'r')
    blue_matrix = motion_corrected_data_container["Blue_Data"]
    violet_matrix = motion_corrected_data_container["Violet_Data"]
    print("Blue Matrix", np.shape(blue_matrix))

    # Get Blue and Violet Example Images
    blue_image = blue_matrix[:, default_position]
    violet_image = violet_matrix[:, default_position]

    # Load Mask
    indicies, image_height, image_width = load_mask(base_directory)

    # Reconstruct Images
    blue_image = create_image_from_data(blue_image, indicies, image_height, image_width)
    violet_image = create_image_from_data(violet_image, indicies, image_height, image_width)

    # Save Images
    np.save(os.path.join(output_directory, "Blue_Example_Image.npy"), blue_image)
    np.save(os.path.join(output_directory, "Violet_Example_Image.npy"), violet_image)

    # Close File
    motion_corrected_data_container.close()


def run_heamocorrection_on_server(mouse_name, save_root):

    # Get Save Location
    save_location = os.path.join(save_root, mouse_name)
    if not os.path.exists(save_location):
        os.mkdir(save_location)

    # Get Remote Directory
    network_directory = "/run/user/1000/gvfs"
    mapped_disks = os.listdir(network_directory)
    print("Mapped Disks", mapped_disks)
    z_drive = os.path.join(network_directory, mapped_disks[0])
    server_folder = os.path.join(z_drive, "Data/Matt/Processed", mouse_name)

    # List Session Directories
    subfolder_list = os.listdir(server_folder)

    print("Subfolders", subfolder_list)
    for session in tqdm(subfolder_list):
        print("Processing Session: ", session, "at", datetime.now())

        # Check Save Location Exists
        session_save_folder = os.path.join(save_location, session)
        if not os.path.exists(session_save_folder):
            os.mkdir(session_save_folder)

        # Set Base Directory
        base_directory = os.path.join(server_folder, session)

        # Perform Heamocorrection
        Heamocorrection.perform_heamocorrection(base_directory, session_save_folder, exclusion_point=3000, lowcut_filter=True, low_cut_freq=0.0033, gaussian_filter=True, gaussian_filter_width=1, use_baseline_frames=True)

        # Get Example Blue and Violet Images
        get_example_images(base_directory, session_save_folder)

        # Copy Mask
        source_mask = os.path.join(base_directory, "Generous_Mask.npy")
        dest_mask = os.pah.join(session_save_folder, "Generous_Mask.npy")
        shutil.copyfile(source_mask, dest_mask)




# Changr Subfolder List
control_mice = ["NRXN78.1A"] #, "NRXN78.1D", "NXAK7.1B", "NXAK4.1B", "NXAK14.1A", "NXAK22.1A"]
save_root = r"//media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Control_Data"
for mouse in control_mice:
    run_heamocorrection_on_server(mouse, save_root)

"""
mutant_mice = ["NRXN71.2A", "NXAK4.1A", "NXAK10.1A", "NXAK16.1B", "NXAK20.1B", "NXAK24.1C"]
save_root = r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Neurexin_Data"
for mouse in mutant_mice:
    run_heamocorrection_on_server(mouse, save_root)
"""

