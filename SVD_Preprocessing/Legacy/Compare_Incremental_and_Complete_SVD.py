import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions



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

    return indicies, image_height, image_width





base_directory = r"/media/matthew/29D46574463D2856/NXAK14.1A_2021_06_15_Transition_Imaging"
output_file = "Blue_Data_SVD"

# Load FUll SVD
full_transformed_data = np.load(os.path.join(base_directory, output_file) + "_transformed_data.npy")
full_components       = np.load(os.path.join(base_directory, output_file) + "_components.npy")
full_singular_values  = np.load(os.path.join(base_directory, output_file) + "_singular_values.npy")

print(np.shape(full_components))

# Load Incremental SVD
incremental_transformed_data = np.load(os.path.join(base_directory, output_file) + "_incremental_transformed_data.npy")
incremental_components       = np.load(os.path.join(base_directory, output_file) + "_incremental_components.npy")
incremental_singular_values  = np.load(os.path.join(base_directory, output_file) + "_incremental_singular_values.npy")


# Load Mask
indicies, image_height, image_width = load_mask(base_directory)

# View Components
figure_1 = plt.figure()
rows =  2
columns = 5

count = 1
for x in range(5):
    full_component = full_components[x]
    incremental_component = incremental_components[x]

    full_component = Widefield_General_Functions.create_image_from_data(full_component, indicies, image_height, image_width)
    incremental_component = Widefield_General_Functions.create_image_from_data(incremental_component, indicies, image_height, image_width)

    full_axis        = figure_1.add_subplot(rows, columns, count)
    incremental_axis = figure_1.add_subplot(rows, columns, count + 1)

    full_axis.imshow(np.abs(full_component))
    incremental_axis.imshow(np.abs(incremental_component))

    full_axis.set_title("Full Component " + str(x + 1))
    incremental_axis.set_title("Incremental Component " + str(x + 1))

    count += 2

plt.show()