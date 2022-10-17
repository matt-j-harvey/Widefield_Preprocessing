import h5py
import numpy as np
import tables
import matplotlib.pyplot as plt
import os
from skimage.transform import downscale_local_mean, resize
from tqdm import tqdm

import Preprocessing_Utils




def load_downsampled_mask(base_directory):
    downsampled_mask_dict = np.load(os.path.join(base_directory, "Downsampled_mask_dict.npy"), allow_pickle=True)[()]
    indicies = downsampled_mask_dict['indicies']
    image_height = downsampled_mask_dict['image_height']
    image_width = downsampled_mask_dict['image_width']
    return indicies, image_height, image_width





def load_smallest_mask(base_directory):

    indicies, image_height, image_width = load_downsampled_mask(base_directory)
    template = np.zeros(image_height * image_width)
    template[indicies] = 1
    template = np.reshape(template, (image_height, image_width))
    template = template[0:300, 0:300]
    template = resize(template, (100,100),preserve_range=True, order=0, anti_aliasing=True)
    template = np.reshape(template, 100 * 100)
    downsampled_indicies = np.nonzero(template)
    return downsampled_indicies, 100, 100

def downsample_to_100(base_directory, output_directory):

    # Load Processed Data
    delta_f_file = os.path.join(output_directory, "Downsampled_Delta_F.h5")
    delta_f_file_container = tables.open_file(delta_f_file, mode="r")
    delta_f_matrix = delta_f_file_container.root["Data"]
    number_of_frames, number_of_pixels = np.shape(delta_f_matrix)
    print("Number of frames", number_of_frames)
    print("Number of pixels", number_of_pixels)

    # Load Mask
    indicies, image_height, image_width = load_downsampled_mask(base_directory)
    print("Indicies", len(indicies))
    downsampled_indicies, downsampled_height, downsampled_width = load_smallest_mask(base_directory)

    # Define Chunking Settings
    preferred_chunk_size = 10000
    number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = Preprocessing_Utils.get_chunk_structure(preferred_chunk_size, number_of_frames)

    downsampled_data = []
    for chunk_index in tqdm(range(number_of_chunks)):

        # Get Selected Indicies
        chunk_start = int(chunk_starts[chunk_index])
        chunk_stop = int(chunk_stops[chunk_index])

        chunk_data = delta_f_matrix[chunk_start:chunk_stop]

        for frame in chunk_data:
            template = np.zeros(image_height * image_width)
            template[indicies] = frame
            template = np.reshape(template, (image_height, image_width))
            template = template[0:300, 0:300]
            template = downscale_local_mean(template, (3,3))
            template = np.reshape(template, (downsampled_height * downsampled_width))
            frame_data = template[downsampled_indicies]
            downsampled_data.append(frame_data)

    delta_f_file_container.close()

    downsampled_data = np.array(downsampled_data)
    np.save(os.path.join(base_directory, "100_By_100_Data.npy"), downsampled_data)

session_list = ["/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Control_Data/NRXN78.1A/2020_11_28_Switching_Imaging",
                "/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Control_Data/NRXN78.1A/2020_12_05_Switching_Imaging",
                "/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Control_Data/NRXN78.1A/2020_12_09_Switching_Imaging"]

for base_directory in session_list:
    downsample_to_100(base_directory, base_directory)