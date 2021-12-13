import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy import ndimage
import h5py
from datetime import datetime

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions




def load_transformation_details(base_directory):

    # Load Transformation Details
    transformation_details = np.load(os.path.join(base_directory, "Transformation_Dictionary.npy"), allow_pickle=True)
    transformation_details = transformation_details[()]
    return transformation_details


def register_chunk(data, session_indicies, template_indicies, image_height, image_width, transformation_dictionary):

    # Load Variables From Dictionary
    rotation = transformation_dictionary['rotation']
    x_shift = transformation_dictionary['x_shift']
    y_shift = transformation_dictionary['y_shift']

    # Create Empty array To Hold Data
    registered_data = []

    # Inset Data Into Empty Array
    count = 0
    for frame in data:
#
        # Create Empty Image To Hold Pixel Data
        image = np.zeros(image_height * image_width)

        # Insert Pixel Data
        image[session_indicies] = frame

        # Reshape Into 2D Image
        image = np.ndarray.reshape(image, (image_height, image_width))

        # Rotate
        image = ndimage.rotate(image, rotation, reshape=False, order=1)

        # Translate
        image = np.roll(a=image, axis=0, shift=y_shift)
        image = np.roll(a=image, axis=1, shift=x_shift)

        # Cheeky Gaussian
        #image = ndimage.gaussian_filter(image, sigma=1)

        # Flatten
        image = np.ndarray.reshape(image, (image_height * image_width))

        # Take Active Pixels
        registered_data.append(image[template_indicies])

        count += 1

    registered_data = np.array(registered_data)

    """
    template = np.zeros(image_height * image_width)
    template[template_indicies] = registered_data[0]
    template = np.ndarray.reshape(template, (image_height,image_width))
    plt.imshow(template)
    plt.show()
    """

    return registered_data




def register_delta_f(base_directory, output_file):

    # Get File Locations
    delta_f_unregistered_file = os.path.join(base_directory, "Delta_F_Unregistered.hdf5")
    session_indicies_file = os.path.join(base_directory, "session_indicies.npy")

    # Load Delta F File
    delta_f_container = h5py.File(delta_f_unregistered_file, 'r')
    delta_f = delta_f_container["Data"]

    # Load Mask
    template_indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)
    session_indicies = np.load(session_indicies_file)

    # Get Data Structure
    number_of_frames = np.shape(delta_f)[0]
    number_of_active_pixels = np.shape(template_indicies)[0]

    # Load Transformation Dictionary
    transformation_dictionary = load_transformation_details(base_directory)

    # Define Chunking Settings
    preferred_chunk_size = 20000
    number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = Widefield_General_Functions.get_chunk_structure(preferred_chunk_size, number_of_frames)

    # Register Data
    with h5py.File(output_file, "w") as f:
        dataset = f.create_dataset("Data", (number_of_frames, number_of_active_pixels), dtype=np.float32, chunks=True, compression="gzip")

        for chunk_index in range(number_of_chunks):
            print("Chunk:", str(chunk_index).zfill(2), "of", number_of_chunks, " ", datetime.now())
            chunk_start = chunk_starts[chunk_index]
            chunk_stop = chunk_stops[chunk_index]
            registered_chunk = register_chunk(delta_f[chunk_start:chunk_stop], session_indicies, template_indicies, image_height, image_width, transformation_dictionary)
            dataset[chunk_start:chunk_stop] = registered_chunk


