import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
import os
import math
import scipy
import tables
from bisect import bisect_left
import cv2
from sklearn.decomposition import TruncatedSVD
from pathlib import Path
import joblib
from matplotlib.colors import LinearSegmentedColormap
from skimage.transform import resize
from scipy import ndimage

# Mathmatical Functions

def factor_number(number_to_factor):
    factor_list = []
    for potential_factor in range(1, number_to_factor):
        if number_to_factor % potential_factor == 0:
            factor_pair = [potential_factor, int(number_to_factor/potential_factor)]
            factor_list.append(factor_pair)

    return factor_list

def load_downsampled_mask(base_directory):
    mask_dict = np.load(os.path.join(base_directory, "Downsampled_mask_dict.npy"), allow_pickle=True)[()]
    indicies = mask_dict["indicies"]
    image_height = mask_dict["image_height"]
    image_width = mask_dict["image_width"]
    return indicies, image_height, image_width

def get_chunk_structure(chunk_size, array_size):
    number_of_chunks = int(np.ceil(array_size / chunk_size))
    remainder = array_size % chunk_size

    # Get Chunk Sizes
    chunk_sizes = []
    if remainder == 0:
        for x in range(number_of_chunks):
            chunk_sizes.append(chunk_size)

    else:
        for x in range(number_of_chunks - 1):
            chunk_sizes.append(chunk_size)
        chunk_sizes.append(remainder)

    # Get Chunk Starts
    chunk_starts = []
    chunk_start = 0
    for chunk_index in range(number_of_chunks):
        chunk_starts.append(chunk_size * chunk_index)

    # Get Chunk Stops
    chunk_stops = []
    chunk_stop = 0
    for chunk_index in range(number_of_chunks):
        chunk_stop += chunk_sizes[chunk_index]
        chunk_stops.append(chunk_stop)

    return number_of_chunks, chunk_sizes, chunk_starts, chunk_stops


def get_best_grid(number_of_items):
    factors = factor_number(number_of_items)
    factor_difference_list = []

    #Get Difference Between All Factors
    for factor_pair in factors:
        factor_difference = abs(factor_pair[0] - factor_pair[1])
        factor_difference_list.append(factor_difference)

    #Select Smallest Factor difference
    smallest_difference = np.min(factor_difference_list)
    best_pair = factor_difference_list.index(smallest_difference)

    return factors[best_pair]


def invert_dictionary(dictionary):
    inv_map = {v: k for k, v in dictionary.items()}
    return inv_map


def take_closest(myList, myNumber):

    """
    Assumes myList is sorted. Returns closest value to myNumber.
    If two numbers are equally close, return the smallest number.
    """

    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before

def ResampleLinear1D(original, targetLen):
    original = np.array(original, dtype=np.float)
    index_arr = np.linspace(0, len(original)-1, num=targetLen, dtype=np.float)
    index_floor = np.array(index_arr, dtype=np.int) #Round down
    index_ceil = index_floor + 1
    index_rem = index_arr - index_floor #Remain

    val1 = original[index_floor]
    val2 = original[index_ceil % len(original)]
    interp = val1 * (1.0-index_rem) + val2 * index_rem
    assert(len(interp) == targetLen)
    return interp


# File Manipulation Functions
def get_ai_filename(base_directory):

    #Get List of all files
    file_list = os.listdir(base_directory)
    ai_filename = None

    #Get .h5 files
    h5_file_list = []
    for file in file_list:
        if file[-3:] == ".h5":
            h5_file_list.append(file)

    #File the H5 file which is two dates seperated by a dash
    for h5_file in h5_file_list:
        original_filename = h5_file

        #Remove Ending
        h5_file = h5_file[0:-3]

        #Split By Dashes
        h5_file = h5_file.split("-")

        if len(h5_file) == 2 and h5_file[0].isnumeric() and h5_file[1].isnumeric():
            ai_filename = "/" + original_filename
            return ai_filename

def get_bodycam_filename(base_directory):

    file_list = os.listdir(base_directory)

    for file in file_list:
        file_split = file.split('_')
        if file_split[-1] == '1.mp4' and file_split[-2] == 'cam':
            return file

def get_eyecam_filename(base_directory):

    file_list = os.listdir(base_directory)

    for file in file_list:
        file_split = file.split('_')
        if file_split[-1] == '2.mp4' and file_split[-2] == 'cam':
            return file


def check_directory(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


def load_ai_recorder_file(ai_recorder_file_location):
    table = tables.open_file(ai_recorder_file_location, mode='r')
    data = table.root.Data

    number_of_seconds = np.shape(data)[0]
    number_of_channels = np.shape(data)[1]
    sampling_rate = np.shape(data)[2]

    data_matrix = np.zeros((number_of_channels, number_of_seconds * sampling_rate))

    for second in range(number_of_seconds):
        data_window = data[second]
        start_point = second * sampling_rate

        for channel in range(number_of_channels):
            data_matrix[channel, start_point:start_point + sampling_rate] = data_window[channel]

    data_matrix = np.clip(data_matrix, a_min=0, a_max=None)
    return data_matrix


def load_generous_mask(home_directory):

    # Loads the mask for a video, returns a list of which pixels are included, as well as the original image height and width
    mask = np.load(os.path.join(home_directory, "Generous_Mask.npy"))

    image_height = np.shape(mask)[0]
    image_width = np.shape(mask)[1]

    mask = np.where(mask>0.1, 1, 0)
    mask = mask.astype(int)
    flat_mask = np.ndarray.flatten(mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    return indicies, image_height, image_width


def get_background_pixels(indicies, image_height, image_width):
    template = np.ones(image_height * image_width)
    template[indicies] = 0
    template = np.reshape(template, (image_height, image_width))
    background_pixels = np.nonzero(template)
    return background_pixels



def transform_mask_or_atlas(image, variable_dictionary):

    image_height = 600
    image_width = 608

    # Unpack Dictionary
    angle = variable_dictionary['rotation']
    x_shift = variable_dictionary['x_shift']
    y_shift = variable_dictionary['y_shift']
    x_scale = variable_dictionary['x_scale']
    y_scale = variable_dictionary['y_scale']

    # Copy
    transformed_image = np.copy(image)

    # Scale
    original_height, original_width = np.shape(transformed_image)
    new_height = int(original_height * y_scale)
    new_width = int(original_width * x_scale)
    transformed_image = resize(transformed_image, (new_height, new_width), preserve_range=True)

    # Rotate
    transformed_image = ndimage.rotate(transformed_image, angle, reshape=False, prefilter=True)

    # Insert Into Background
    mask_height, mask_width = np.shape(transformed_image)
    centre_x = 200
    centre_y = 200
    background_array = np.zeros((1000, 1000))
    x_start = centre_x + x_shift
    x_stop = x_start + mask_width

    y_start = centre_y + y_shift
    y_stop = y_start + mask_height

    background_array[y_start:y_stop, x_start:x_stop] = transformed_image

    # Take Chunk
    transformed_image = background_array[centre_y:centre_y + image_height, centre_x:centre_x + image_width]

    # Rebinarize
    transformed_image = np.where(transformed_image > 0.5, 1, 0)

    return transformed_image



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





def load_tight_mask_downsized():

    mask = np.load("/home/matthew/Documents/Allen_Atlas_Templates/Mask_Array.npy")
    mask_alignment_dictionary = np.load(r"/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Tight_Mask_Alignment_Dictionary.npy", allow_pickle=True)[()]

    # Transform Mask
    mask = transform_mask_or_atlas(mask, mask_alignment_dictionary)
    mask = resize(mask, (100, 100), preserve_range=True, order=0, anti_aliasing=False)

    image_height = np.shape(mask)[0]
    image_width = np.shape(mask)[1]

    mask = np.where(mask>0.1, 1, 0)
    mask = mask.astype(int)
    flat_mask = np.ndarray.flatten(mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    return indicies, image_height, image_width

def determine_trial_type(home_directory):
    trial_type = None
    home_directory_list = home_directory.split("/")
    print(home_directory_list)
    if "Discrimination" in home_directory_list[-1]:
        trial_type = "Discrimination"
    elif "Switching" in home_directory_list[-1]:
        trial_type = "Switching"

    return trial_type



def get_musall_cmap():
    cmap = LinearSegmentedColormap.from_list('mycmap', [
        (0, 0.87, 0.9, 1),
        (0, 0, 1, 1),
        (0, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 1, 0, 1),
    ])

    return cmap



# AI Recorder Processing Functions

def create_stimuli_dictionary():

    channel_index_dictionary = {
        "Photodiode"        :0,
        "Reward"            :1,
        "Lick"              :2,
        "Visual 1"          :3,
        "Visual 2"          :4,
        "Odour 1"           :5,
        "Odour 2"           :6,
        "Irrelevance"       :7,
        "Running"           :8,
        "Trial End"         :9,
        "Camera Trigger"    :10,
        "Camera Frames"     :11,
        "LED 1"             :12,
        "LED 2"             :13,
        "Mousecam"          :14,
        "Optogenetics"      :15,
        }

    return channel_index_dictionary

def get_offset(onset, stream, threshold=0.5):

    count = 50
    on = True
    while on:
        if stream[onset + count] < threshold:
            on = False
            return onset + count
        else:
            count += 1

def get_step_onsets(trace, threshold=1, window=3):
    state = 0
    number_of_timepoints = len(trace)
    onset_times = []
    time_below_threshold = 0

    onset_line = []

    for timepoint in range(number_of_timepoints):
        if state == 0:
            if trace[timepoint] > threshold:
                state = 1
                onset_times.append(timepoint)
                time_below_threshold = 0
            else:
                pass
        elif state == 1:
            if trace[timepoint] > threshold:
                time_below_threshold = 0
            else:
                time_below_threshold += 1
                if time_below_threshold > window:
                    state = 0
                    time_below_threshold = 0
        onset_line.append(state)

    return onset_times

def get_nearest_frame(stimuli_onsets, frame_onsets):


    frame_times = frame_onsets.keys()
    nearest_frames = []
    window_size = 50

    for onset in stimuli_onsets:
        smallest_distance = 1000
        closest_frame = None

        window_start = onset - window_size
        window_stop  = onset + window_size

        for timepoint in range(window_start, window_stop):

            #There is a frame at this time
            if timepoint in frame_times:
                distance = abs(onset - timepoint)

                if distance < smallest_distance:
                    smallest_distance = distance
                    closest_frame = frame_onsets[timepoint]

        if closest_frame != None:
            if closest_frame > 11:
                nearest_frames.append(closest_frame)

    nearest_frames = np.array(nearest_frames)
    return nearest_frames


# Data Extraction Functions

def get_selected_widefield_frames(onsets, start_window, stop_window):

    selected_fames = []

    for onset in onsets:
        trial_start = onset + start_window
        trial_stop = onset + stop_window
        trial_frames = list(range(trial_start, trial_stop))
        selected_fames.append(trial_frames)

    return selected_fames


def get_selected_widefield_data(selected_widefield_onsets, widefield_data):

    selected_widefield_data = []

    for trial in selected_widefield_onsets:
        trial_data = []
        for frame in trial:
            frame_data = widefield_data[frame]
            trial_data.append(frame_data)

        selected_widefield_data.append(trial_data)

    selected_widefield_data = np.array(selected_widefield_data)
    return selected_widefield_data


def create_image_from_data(data, indicies, image_height, image_width):
    template = np.zeros((image_height, image_width))
    data = np.nan_to_num(data)
    np.put(template, indicies, data)
    image = np.ndarray.reshape(template, (image_height, image_width))

    return image

def get_video_details(video_file):
    cap = cv2.VideoCapture(video_file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return frameCount, frameHeight, frameWidth

def get_mousecam_files(base_directory):

    # Get List Of All Files In Directory
    file_list = os.listdir(base_directory)

    cam_1_file = None
    cam_2_file = None

    # Iterate Through Files and See If They Contain The Phrase "Cam"
    for file in file_list:
        if "cam_1" in file:
            cam_1_file = file
        elif "cam_2" in file:
            cam_2_file = file


    # If We Are Unable To Find Either One - Return Error
    if cam_1_file == None or cam_2_file == None:
        print("Mousecam Files Not Found")
        return None, None

    else:

        # Check Camera Details To Distinquish Between Eye and Body Cam
        cam_1_details = get_video_details(os.path.join(base_directory, cam_1_file))
        cam_2_details = get_video_details(os.path.join(base_directory, cam_2_file))

        print("Cam 1 File: ", cam_1_file, "Frames:", cam_1_details[0], " Height:", cam_1_details[1], " Width:", cam_1_details[2])
        print("Cam 2 File: ", cam_2_file, "Frames:", cam_2_details[0], " Height:", cam_2_details[1], " Width:", cam_2_details[2])

        if cam_1_details[1] == 480:
            bodycam = cam_1_file
            eyecam = cam_2_file
        else:
            bodycam = cam_2_file
            eyecam = cam_1_file

        return bodycam, eyecam
