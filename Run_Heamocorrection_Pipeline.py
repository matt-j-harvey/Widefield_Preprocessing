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
import Downsample_Existing_Data
import Downsampled_Delta_F_With_Regression
import Create_Downsampled_Mask_Dict
import Get_Example_Images
import Convert_DF_To_Tables


def get_motion_corrected_data_filename(base_directory):
    file_list = os.listdir(base_directory)
    for file_name in file_list:
        if "Motion_Corrected" in file_name:
            return file_name


def run_heamocorrection_locally(mouse_name, motion_corrected_data_root, save_root, subfolder_list=None, operating_system='linux'):

    # Get Save Location
    save_location = os.path.join(save_root, mouse_name)
    if not os.path.exists(save_location):
        os.mkdir(save_location)

    # List Session Directories
    if subfolder_list == None:
        subfolder_list = os.listdir(motion_corrected_data_root)

    print("Subfolders", subfolder_list)
    for session in tqdm(subfolder_list):
        print("Processing Session: ", session, "at", datetime.now())

        # Check Save Location Exists
        session_save_folder = os.path.join(save_location, session)
        if not os.path.exists(session_save_folder):
            os.mkdir(session_save_folder)

        # Set Base Directory
        base_directory = os.path.join(server_folder, session)

        # Copy Mask
        source_mask = os.path.join(base_directory, "Generous_Mask.npy")
        dest_mask = os.path.join(session_save_folder, "Generous_Mask.npy")
        shutil.copyfile(source_mask, dest_mask)

        # Create Downsampled Mask Dict
        Create_Downsampled_Mask_Dict.create_downsampled_mask_dict(session_save_folder)

        # Downsample Raw Data
        Downsample_Existing_Data.downsample_session(base_directory, session_save_folder)

        # Create Example Images
        Get_Example_Images.get_example_images(session_save_folder, session_save_folder)

        # Perform Heamocorrection
        Downsampled_Delta_F_With_Regression.create_delta_f_file(session_save_folder, session_save_folder)


def run_heamocorrection_on_server(mouse_name, save_root, subfolder_list=None, operating_system='linux'):

    # Get Save Location
    save_location = os.path.join(save_root, mouse_name)
    if not os.path.exists(save_location):
        os.mkdir(save_location)

    # Get Remote Directory
    if operating_system == 'linux':
        # Get Remote Directory
        network_directory = "/run/user/1000/gvfs"
        mapped_disks = os.listdir(network_directory)
        print("Mapped Disks", mapped_disks)
        z_drive = os.path.join(network_directory, mapped_disks[1])
        server_folder = os.path.join(z_drive, "Data/Matt/Processed", mouse_name)

    elif operating_system == 'windows':
        z_drive = r"Z:\\"
        server_folder = os.path.join(z_drive, "Data/Matt/Processed", mouse_name)

    # List Session Directories
    if subfolder_list == None:
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

        # Copy Mask
        source_mask = os.path.join(base_directory, "Generous_Mask.npy")
        dest_mask = os.path.join(session_save_folder, "Generous_Mask.npy")
        shutil.copyfile(source_mask, dest_mask)

        # Create Downsampled Mask Dict
        Create_Downsampled_Mask_Dict.create_downsampled_mask_dict(session_save_folder)

        # Downsample Raw Data
        Downsample_Existing_Data.downsample_session(base_directory, session_save_folder)

        # Create Example Images
        Get_Example_Images.get_example_images(session_save_folder, session_save_folder)

        # Perform Heamocorrection
        Downsampled_Delta_F_With_Regression.create_delta_f_file(session_save_folder, session_save_folder)


# Change Subfolder List
control_mice = ["NRXN78.1A", "NRXN78.1D", "NXAK7.1B", "NXAK4.1B", "NXAK14.1A", "NXAK22.1A"]
mutant_mice = ["NXAK24.1C"]# [, "NRXN71.2A"] #"NXAK10.1A",  #"NXAK4.1A", "NXAK16.1B"
save_root = r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Mutant_Data"
subfolder_list = None


for mouse in mutant_mice:
    run_heamocorrection_on_server(mouse, save_root, subfolder_list=subfolder_list, operating_system='linux')
