import numpy as np
import matplotlib.pyplot as plt
import tables
from tqdm import tqdm
import os
from skimage.transform import resize

import Preprocessing_Utils


def downsample_delta_f_matrix(base_directory):

    # Load Activity Matrix
    delta_f_file = os.path.join(base_directory, "Delta_F.h5")
    delta_f_container = tables.open_file(delta_f_file, "r")
    activity_matrix = delta_f_container.root.Data

    # Load Masks
    indicies, image_height, image_width = Preprocessing_Utils.load_generous_mask(base_directory)
    downsample_indicies, downsample_height, downsample_width = Preprocessing_Utils.load_tight_mask_downsized()

    # Load Alignment Dictionary
    alignment_dictionary = np.load(os.path.join(base_directory, "Cluster_Alignment_Dictionary.npy"), allow_pickle=True)[()]

    # Downsample Data
    downsampled_data = []
    for frame in tqdm(activity_matrix):

        # Recreate Image
        frame = Preprocessing_Utils.create_image_from_data(frame, indicies, image_height, image_width)

        # Align To Common Framework
        frame = Preprocessing_Utils.transform_image(frame, alignment_dictionary)

        # Downsample
        frame = resize(frame, output_shape=(100, 100),preserve_range=True)

        frame = np.reshape(frame, 100 * 100)

        frame = frame[downsample_indicies]

        downsampled_data.append(frame)

    downsampled_data = np.array(downsampled_data)

    np.save(os.path.join(base_directory, "Downsampled_Aligned_Data.npy"), downsampled_data)

    return downsampled_data


session_list = [
    #"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_29_Switching_Imaging",
    #"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_12_05_Switching_Imaging",
    #"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_12_07_Switching_Imaging",

    "/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_28_Switching_Imaging",
    "/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_05_Switching_Imaging",
    "/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_09_Switching_Imaging",

    "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_02_Switching_Imaging",
    "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_04_Switching_Imaging",
    "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_06_Switching_Imaging",

    "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_26_Switching_Imaging",
    "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_28_Switching_Imaging",
    "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_02_Switching_Imaging",

    "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_21_Switching_Imaging",
    "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_23_Switching_Imaging",
    "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_11_Switching_Imaging",

    "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_14_Switching_Imaging",
    "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_20_Switching_Imaging",
    "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_22_Switching_Imaging",

]

session_list = [
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_13_Switching_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_15_Switching_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_17_Switching_Imaging",

    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_03_31_Switching_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_02_Switching_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_04_04_Switching_Imaging",

    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_20_Switching_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_22_Switching_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_24_Switching_Imaging",

    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_17_Switching_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_19_Switching_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_23_Switching_Imaging",

    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_15_Switching_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_17_Switching_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_11_19_Switching_Imaging",

    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_14_Switching_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_20_Switching_Imaging",
    "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_26_Switching_Imaging"

]

session_list = [#r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Control_Data/NRXN78.1D/2020_11_29_Switching_Imaging",
                r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Control_Data/NRXN78.1D/2020_12_05_Switching_Imaging",
                r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Control_Data/NRXN78.1D/2020_12_07_Switching_Imaging"]

for base_directory in session_list:
    downsample_delta_f_matrix(base_directory)