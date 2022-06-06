import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import TruncatedSVD
import h5py
import sys
from sklearn.decomposition import TruncatedSVD


import datetime
sys.path.append(r"C:\Users\Creature\Documents\Matt Scripts\Widefield_Preprocessing")

import Widefield_General_Functions



def get_grid(height, width, divisor):

    block_height = int(height / divisor)
    block_width = int(width / divisor)

    blocks = []

    for y in range(divisor):
        for x in range(divisor):

            y_start = y * block_height
            y_stop = y_start + block_height

            x_start = x * block_width
            x_stop = x_start + block_width

            blocks.append([y_start, y_stop, x_start, x_stop])

    return blocks, block_height, block_width




def perform_blockwise_svd(video_matrix, block_pixels_list, number_of_pixels, number_of_components=100, block_components=20):

    block_spatial_components_list = []
    block_temporal_components_list = []
    blocks_empty_list = []

    number_of_blocks = len(block_pixels_list)
    for block_index in range(number_of_blocks):
        print("Block", block_index, " of ", number_of_blocks, "at ", datetime.datetime.now())

        # Get Block Indicies
        block_pixels = block_pixels_list[block_index]

        # If Block Empty
        if len(block_pixels) == 0:
            blocks_empty_list.append(True)
            pass

        else:
            blocks_empty_list.append(False)

            # Get Block Data
            block_data = video_matrix[:, block_pixels]

            block_data = np.nan_to_num(block_data)

            # Get Block SVD
            block_model = TruncatedSVD(n_components=block_components)
            block_model.fit(block_data)
            block_spatial_components = block_model.components_
            block_temporal_components = block_model.transform(block_data)

            # Append To List
            block_temporal_components_list.append(block_temporal_components)
            block_spatial_components_list.append(block_spatial_components)

    # Stack Temporal Components
    block_temporal_components_list = np.hstack(block_temporal_components_list)

    # Perform SVD On Temporal Components
    model = TruncatedSVD(n_components=number_of_components)
    model.fit(block_temporal_components_list)

    # Map Spatial Components Back To Blocks
    joint_spatial_components = model.components_

    # Create Empty Array To Hold Them
    reassembled_spatial_components = np.zeros((number_of_components, number_of_pixels))


    for component_index in range(number_of_components):
        component_data = joint_spatial_components[component_index]

        reconstructed_component = np.zeros((number_of_pixels))

        included_block_count = 0
        for block_index in range(number_of_blocks):
            block_pixels = block_pixels_list[block_index]

            if blocks_empty_list[block_index] == False:

                # Get Spatial Components For This Block
                block_start = included_block_count * block_components
                block_stop = block_start + block_components

                # Get Loadings from the joint SVD for this block
                block_component_loadings = component_data[block_start:block_stop]

                # Get The Origional Spatial Components For This Block
                block_spatial_components = block_spatial_components_list[included_block_count]

                # Scale The Original Spatial Components By The Loadings from the Joint Temporal SVD
                block_contribution = np.dot(block_component_loadings, block_spatial_components)

                # Put Back Into Full Component
                reconstructed_component[block_pixels] = block_contribution

                included_block_count += 1

        # Reshape Block Contribution Back Into Square
        reassembled_spatial_components[component_index] = reconstructed_component

    combined_temporal_components = model.transform(block_temporal_components_list)

    return reassembled_spatial_components, combined_temporal_components


def load_mask(home_directory):

    # Loads the mask for a video, returns a list of which pixels are included, as well as the original image height and width
    mask = np.load(home_directory + "/Generous_Mask.npy")

    image_height = np.shape(mask)[0]
    image_width = np.shape(mask)[1]

    mask = np.where(mask>0.1, 1, 0)
    mask = mask.astype(int)
    flat_mask = np.ndarray.flatten(mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    return indicies, image_height, image_width, mask


def create_index_mask(mask):

    # Get Indicies
    mask = np.where(mask > 0.1, 1, 0)
    mask = mask.astype(int)
    flat_mask = np.ndarray.flatten(mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    # Create Index Mask
    index_mask = np.ones(np.shape(mask))
    index_mask = np.multiply(index_mask, -1)
    index_mask = np.ndarray.flatten(index_mask)
    index_mask[indicies] = indicies
    index_mask = np.reshape(index_mask, (np.shape(mask)))
    index_mask = np.ndarray.astype(index_mask, np.int)
    return index_mask


def get_block_indicies(indicies, image_height, image_width, mask):

    # Get Block Details
    block_list, block_height, block_width = get_grid(image_height, image_width, 3)
    pixel_list = list(range(len(indicies)))

    # Create Index Mask
    index_mask = np.ones(np.shape(mask))
    index_mask = np.multiply(index_mask, -1)
    index_mask = np.ndarray.flatten(index_mask)
    index_mask[indicies] = pixel_list
    index_mask = np.reshape(index_mask, (np.shape(mask)))
    index_mask = np.ndarray.astype(index_mask, np.int)

    # Get Indicies In Each Block
    block_indicies_list = []
    for block in block_list:

        # Get Block Indicies
        block_indicies = index_mask[block[0]:block[1], block[2]:block[3]]
        block_indicies = np.unique(block_indicies)
        block_indicies = list(block_indicies)
        block_indicies.sort()
        if -1 in block_indicies:
            block_indicies.remove(-1)

        block_indicies_list.append(block_indicies)

    return block_indicies_list


def plot_components(base_directory):

    # Load Sparial Components
    spatial_components = np.load(os.path.join(base_directory, "Blockwise_SVD_Spatial_Components.npy"))
    print(np.shape(spatial_components))

    # Load Mask
    indicies, image_height, image_width, mask = load_mask(base_directory)

    # Create Figure
    figure_1 = plt.figure()
    rows = 10
    columns = 10

    for x in range(100):

        component_data = spatial_components[x]
        component_image = np.zeros(image_height * image_width)
        component_image[indicies] = component_data
        component_image = np.reshape(component_image, (image_height, image_width))

        axis = figure_1.add_subplot(rows, columns, x+1)
        axis.imshow(component_image)
        axis.axis('off')

    plt.show()

    for x in range(100):

        component_data = spatial_components[x]
        component_image = np.zeros(image_height * image_width)
        component_image[indicies] = component_data
        component_image = np.reshape(component_image, (image_height, image_width))
        plt.title(str(x))
        plt.imshow(component_image)
        plt.show()



def perform_svd_compression(base_directory):

    # Set File Details
    data_file = os.path.join(base_directory, "Delta_F.hdf5")

    # Load Mask
    indicies, image_height, image_width, mask = load_mask(base_directory)

    # Get Block Indicies
    block_pixels_list = get_block_indicies(indicies, image_height, image_width, mask)

    # Load Delta F Data
    downsampled_file_object = h5py.File(data_file, 'r')
    data_matrix = downsampled_file_object["Data"]

    # Perform SVD
    reassembled_spatial_components, combined_temporal_components = perform_blockwise_svd(data_matrix, block_pixels_list, len(indicies))

    np.save(os.path.join(base_directory, "Blockwise_SVD_Spatial_Components.npy"), reassembled_spatial_components)
    np.save(os.path.join(base_directory, "Blockwise_SVD_Temporal_Components.npy"), combined_temporal_components)


#base_directory = r"//media/matthew/Expansion/Widefield_Analysis/NRXN78.1D/2020_12_07_Switching_Imaging"
#perform_svd_compression(base_directory)
#plot_components(base_directory)