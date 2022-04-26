import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import sys

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions





def view_coefficient_vector(pixel_assignments, cluster_label_list, cluster_vector, indicies, image_height, image_width):

    # Create Blank Template
    template = np.zeros(np.shape(indicies))
    print("Template Shape", np.shape(template))

    # Fill Each Cluster
    number_of_clusters = len(cluster_vector)
    for cluster_index in range(number_of_clusters):

        cluster_value = cluster_vector[cluster_index]
        cluster_label = cluster_label_list[cluster_index]

        print("Cluster Index", cluster_index)
        print("Cluster Label", cluster_label)
        print("Cluster Value", cluster_value)

        # Fill Template
        pixel_indexes = np.where(pixel_assignments == cluster_label, 1, 0)
        pixel_indexes = list(np.nonzero(pixel_indexes))
        pixel_indexes.sort()
        template[pixel_indexes] = cluster_value


    # Create Image
    cluster_image = Widefield_General_Functions.create_image_from_data(template, indicies, image_height, image_width)

    return cluster_image




def assign_deta_f_to_allen_regions(base_directory, template_directory):

    #Load Pixel Assignments
    pixel_assignments = np.load(os.path.join(template_directory, "Pixel_Assignmnets.npy"))
    cluster_list = list(np.unique(pixel_assignments))
    number_of_clusters = len(cluster_list)
    print(np.shape(pixel_assignments))
    print(cluster_list)

    # Get Cluster Indicies
    cluster_indicies_list = []
    for cluster in cluster_list:

        mask_array = np.where(pixel_assignments == cluster, 1, 0)
        cluster_pixels = np.nonzero(mask_array)[0]
        cluster_pixels = list(cluster_pixels)
        cluster_pixels.sort()

        cluster_indicies_list.append(cluster_pixels)


    # Load Delta F Matrix
    delta_f_matrix_filepath = os.path.join(base_directory, "Delta_F_Registered.hdf5")
    delta_f_matrix_container = h5py.File(delta_f_matrix_filepath, 'r')
    delta_f_matrix = delta_f_matrix_container['Data']

    # Check Chunk Structure
    number_of_timepoints = np.shape(delta_f_matrix)[0]
    print("Number of timepoints", number_of_timepoints)
    chunk_size = 20000
    number_of_chunks, chunk_sizes, chunk_start_list, chunk_stop_list = Widefield_General_Functions.get_chunk_structure(chunk_size, number_of_timepoints)

    atlas_delta_f = []
    for chunk_index in range(number_of_chunks):
        print("Chunk ", chunk_index, " of ", number_of_chunks)

        chunk_start = chunk_start_list[chunk_index]
        chunk_stop = chunk_stop_list[chunk_index]
        delta_f_chunk = delta_f_matrix[chunk_start:chunk_stop]

        atlas_delta_f_chunk = []
        for cluster_index in range(number_of_clusters):

            # Get Cluster Pixels
            cluster_pixels = cluster_indicies_list[cluster_index]

            # Get Cluster Delta F
            cluster_delta_f = delta_f_chunk[:, cluster_pixels]
            cluster_delta_f = np.nan_to_num(cluster_delta_f)

            # Get Cluster Mean
            cluster_delta_f = np.mean(cluster_delta_f, axis=1)

            # Add To Matrix
            atlas_delta_f_chunk.append(cluster_delta_f)

        atlas_delta_f_chunk = np.array(atlas_delta_f_chunk)
        atlas_delta_f.append(atlas_delta_f_chunk)


        # Visualsie For Sanity Checl
        """
        indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)

        plt.ion()
        for datapoint_index in range(chunk_sizes[chunk_index]):
            datapoint = atlas_delta_f_chunk[:, datapoint_index]

            print("Datapoint Shape", np.shape(datapoint))
            image = view_coefficient_vector(pixel_assignments, cluster_list, datapoint, indicies, image_height, image_width)

            plt.imshow(image, vmin=0, vmax=1, cmap='inferno')
            plt.draw()
            plt.pause(0.1)
            plt.clf()
        """

    atlas_delta_f = np.hstack(atlas_delta_f)
    print("Atlas Delta F Shape", np.shape(atlas_delta_f))
    atlas_delta_f = np.transpose(atlas_delta_f)
    print("Atlas Delta F Shape", np.shape(atlas_delta_f))

    np.save(os.path.join(base_directory, "Allen_Region_Delta_F.npy"), atlas_delta_f)






template_directory = "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging"

session_list = [
            #"/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_02_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_08_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_03_23_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_03_31_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_15_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK22.1A/2021_10_29_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK22.1A/2021_11_05_Transition_Imaging"
            ]

mutants = [
"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK10.1A/2021_06_18_Transition_Imaging",
"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging",

"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_10_Transition_Imaging",
"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_12_Transition_Imaging",

"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK24.1C/2021_11_10_Transition_Imaging",
"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK24.1C/2021_11_10_Transition_Imaging",

"/media/matthew/Seagate Expansion Drive1/Transition_2/NXAK20.1B/2021_11_22_Transition_Imaging",
"/media/matthew/Seagate Expansion Drive1/Transition_2/NXAK20.1B/2021_11_24_Transition_Imaging",
"/media/matthew/Seagate Expansion Drive1/Transition_2/NXAK20.1B/2021_11_26_Transition_Imaging",
]


for base_directory in mutants:
    assign_deta_f_to_allen_regions(base_directory, template_directory)