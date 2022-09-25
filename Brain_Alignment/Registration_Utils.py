import numpy as np
import os
from scipy import ndimage
from skimage.transform import rescale
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


def get_musall_cmap():
    cmap = LinearSegmentedColormap.from_list('mycmap', [

        (0, 0.87, 0.9, 1),
        (0, 0, 1, 1),
        (0, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 1, 0, 1),

    ])

    return cmap

# Load Generous Mask
def load_generous_mask(base_directory):

    # Loads the mask for a video, returns a list of which pixels are included, as well as the original image height and width
    mask = np.load(os.path.join(base_directory, "Generous_Mask.npy"))

    image_height = np.shape(mask)[0]
    image_width = np.shape(mask)[1]

    mask = np.where(mask>0.1, 1, 0)
    mask = mask.astype(int)
    flat_mask = np.ndarray.flatten(mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    return indicies, image_height, image_width

def load_generous_mask_downsampled(base_directory, downsample_factor=8):

    # Loads the mask for a video, returns a list of which pixels are included, as well as the original image height and width
    mask = np.load(os.path.join(base_directory, "Generous_Mask.npy"))

    # Downsample Mask
    mask = rescale(mask, scale=float(1 / downsample_factor), preserve_range=True)

    image_height = np.shape(mask)[0]
    image_width = np.shape(mask)[1]

    mask = np.where(mask>0.1, 1, 0)
    mask = mask.astype(int)
    flat_mask = np.ndarray.flatten(mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    return indicies, image_height, image_width



# Load Generous Mask
def load_consensus_mask():

    # Loads the mask for a video, returns a list of which pixels are included, as well as the original image height and width
    mask = np.load("/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/mask.npy")

    image_height = np.shape(mask)[0]
    image_width = np.shape(mask)[1]

    mask = np.where(mask>0.1, 1, 0)
    mask = mask.astype(int)
    flat_mask = np.ndarray.flatten(mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    return indicies, image_height, image_width



# Transform Image
def transform_image(image, variable_dictionary, invert=False):

    # Unpack Dict
    angle = variable_dictionary['rotation']
    x_shift = variable_dictionary['x_shift']
    y_shift = variable_dictionary['y_shift']

    # Inverse
    if invert == True:
        angle = -1 * angle
        x_shift = -1 * x_shift
        y_shift = -1 * y_shift

    transformed_image = np.copy(image)
    transformed_image = np.nan_to_num(transformed_image)
    transformed_image = ndimage.rotate(transformed_image, angle, reshape=False, prefilter=True, order=1)

    """
    plt.title("Just rotated")
    plt.imshow(transformed_image, vmin=0, vmax=2)
    plt.show()
    """

    transformed_image = np.roll(a=transformed_image, axis=0, shift=y_shift)
    transformed_image = np.roll(a=transformed_image, axis=1, shift=x_shift)

    return transformed_image


# Create Image From Data
def create_image_from_data(data, indicies, image_height, image_width):
    template = np.zeros((image_height, image_width))
    data = np.nan_to_num(data)
    np.put(template, indicies, data)
    image = np.ndarray.reshape(template, (image_height, image_width))

    return image

def flatten(l):
    return [item for sublist in l for item in sublist]


def load_mouse_sessions(mouse_name, session_type):

    # This Is The Location of My Experimental Logbook
    logbook_file_location = r"/home/matthew/Documents/Experiment_Logbook.ods"

    #  Read Logbook As A Dataframe
    logbook_dataframe = pd.read_excel(logbook_file_location, engine="odf")

    # Return A List Of the File Directories Of Sessions Matching The Mouse Name and Session Type
    selected_sessions = logbook_dataframe.loc[(logbook_dataframe["Mouse"] == mouse_name) & (logbook_dataframe["Session Type"] == session_type), ["Filepath"]].values.tolist()

    # Flatten The Subsequent Nested List
    selected_sessions = flatten(selected_sessions)

    return selected_sessions

def load_all_sessions():

    # This Is The Location of My Experimental Logbook
    logbook_file_location = r"/home/matthew/Documents/Experiment_Logbook.ods"

    #  Read Logbook As A Dataframe
    logbook_dataframe = pd.read_excel(logbook_file_location, engine="odf")

    session_list = logbook_dataframe[["Filepath"]].values.tolist()
    session_list = flatten(session_list)

    return session_list


def load_analysis_container(analysis_name):

    # This Is The Location of My Analysis Logbook
    logbook_file_location = r"/home/matthew/Documents/Github_Code/Workflows/Analysis_Containers.ods"

    #  Read Logbook As A Dataframe
    logbook_dataframe = pd.read_excel(logbook_file_location, engine="odf")

    # Return First Analysis Container With This Name
    selected_analysis = logbook_dataframe.loc[(logbook_dataframe["Analysis_Name"] == analysis_name)].values[0]

    # Unpack Container
    start_window = selected_analysis[1]
    stop_window = selected_analysis[2]
    onset_files = selected_analysis[3].replace(' ', '')
    tensor_names = selected_analysis[4].replace(' ', '')
    behaviour_traces = selected_analysis[5] #.replace(' ', '')
    difference_conditions = selected_analysis[6].replace(' ', '')

    onset_files = onset_files.split(',')
    tensor_names = tensor_names.split(',')
    behaviour_traces = behaviour_traces.split(',')
    difference_conditions = difference_conditions.split(',')

    if difference_conditions[0] == 'None':
        difference_conditions = None

    else:
        for x in range(2):
            difference_conditions[x] = int(difference_conditions[x])

    # Return Container
    return [start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions]



