import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py
import os
import tables
from scipy import signal, ndimage, stats
from sklearn.neighbors import KernelDensity
import cv2
from matplotlib import gridspec


def create_stimuli_dictionary(legacy=False):

    if legacy == False:

        channel_index_dictionary = {
            "Photodiode": 0,
            "Reward": 1,
            "Lick": 2,
            "Visual 1": 3,
            "Visual 2": 4,
            "Odour 1": 5,
            "Odour 2": 6,
            "Irrelevance": 7,
            "Running": 8,
            "Trial End": 9,
            "Camera Trigger": 10,
            "Camera Frames": 11,
            "LED 1": 12,
            "LED 2": 13,
            "Mousecam": 14,
            "Optogenetics": 15,
        }

    else:
        channel_index_dictionary = {
            "Reward": 0,
            "Lick": 1,
            "Visual 1": 2,
            "Visual 2": 3,
            "Odour 1": 4,
            "Odour 2": 5,
            "Irrelevance": 6,
            "Running": 7,
            "Trial End": 8,
            "Camera Trigger": 9,
            "Camera Frames": 10,
            "LED 1": 11,
            "LED 2": 12
        }


    return channel_index_dictionary


def load_ai_recorder_file(ai_recorder_file_location):
    table = tables.open_file(ai_recorder_file_location, mode='r')
    data = table.root.Data

    number_of_seconds = np.shape(data)[0]
    number_of_channels = np.shape(data)[1]
    sampling_rate = np.shape(data)[2]

    print("Number of seconds", number_of_seconds)

    data_matrix = np.zeros((number_of_channels, number_of_seconds * sampling_rate))

    for second in range(number_of_seconds):
        data_window = data[second]
        start_point = second * sampling_rate

        for channel in range(number_of_channels):
            data_matrix[channel, start_point:start_point + sampling_rate] = data_window[channel]

    data_matrix = np.clip(data_matrix, a_min=0, a_max=None)
    return data_matrix


def load_trial_details(home_directory, name, time_frame_dict):
    file_directory = home_directory + "/Stimuli_Evoked_Responses/" + name + "/" + name + "_Trial_Details.npy"
    trial_details = np.load(file_directory)

    trial_starts    = []
    max_duration    = 0

    number_of_trials = np.shape(trial_details)[0]

    for trial in range(number_of_trials):

        frame_start = trial_details[trial][0]
        frame_stop  = trial_details[trial][1]

        start_time = time_frame_dict[frame_start]
        stop_time = time_frame_dict[frame_stop+1]

        trial_starts.append(start_time)
        duration = stop_time - start_time
        if duration > max_duration:
            max_duration = duration

    return trial_starts, max_duration


def get_trial_average_trace(ai_recorder_data, trial_starts, max_duration, channel_of_interest):

    channel_index_dictionary = create_stimuli_dictionary(legacy=False)
    channel_trace = ai_recorder_data[channel_index_dictionary[channel_of_interest]]

    number_of_trials = len(trial_starts)
    trial_matrix = np.zeros((number_of_trials, max_duration))

    for trial in range(number_of_trials):
        start = trial_starts[trial]
        stop  = start + max_duration
        trial_trace = channel_trace[start:stop]
        trial_matrix[trial] = trial_trace

    mean_trace = np.mean(trial_matrix, axis=0)
    standard_deviation = np.std(trial_matrix, axis=0)

    upper_bound = np.add(mean_trace, standard_deviation)
    lower_bound = np.subtract(mean_trace, standard_deviation)

    return mean_trace, upper_bound, lower_bound

def get_average_behaviour_trace(home_directory, ai_recorder_data, responses_save_location, time_frame_dict, name):

    # Create Save Directory
    save_directory = responses_save_location + "/" + name
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    trial_starts, max_duration = load_trial_details(home_directory, name, time_frame_dict)


    print("Trial starts: ", trial_starts)
    print("Max duration: ", max_duration)

    behaviour_matrix = []
    behaviour_matrix.append(get_trial_average_trace(ai_recorder_data, trial_starts, max_duration, "Running"))
    behaviour_matrix.append(get_trial_average_trace(ai_recorder_data, trial_starts, max_duration, "Lick"))
    behaviour_matrix.append(get_trial_average_trace(ai_recorder_data, trial_starts, max_duration, "Reward"))
    behaviour_matrix.append(get_trial_average_trace(ai_recorder_data, trial_starts, max_duration, "Odour 1"))
    behaviour_matrix.append(get_trial_average_trace(ai_recorder_data, trial_starts, max_duration, "Odour 2"))
    behaviour_matrix.append(get_trial_average_trace(ai_recorder_data, trial_starts, max_duration, "Visual 1"))
    behaviour_matrix.append(get_trial_average_trace(ai_recorder_data, trial_starts, max_duration, "Visual 2"))
    behaviour_matrix = np.array(behaviour_matrix)

    # Get File Name
    filename = save_directory + "/" + name + "_Behaviour_Matrix.npy"

    print(filename)
    np.save(filename, behaviour_matrix)




def extract_trial_aligned_behaviour(home_directory, airecorder_file_name, condition_names):

    #Set File Structure
    ai_recorder_file_location = home_directory + airecorder_file_name
    responses_save_location = home_directory + "/Stimuli_Evoked_Responses"

    #Load Data
    ai_recorder_data = load_ai_recorder_file(ai_recorder_file_location)
    frame_time_dict = np.load(home_directory + "/Stimuli_Onsets/Frame_Times.npy", allow_pickle=True)
    frame_time_dict = frame_time_dict[()]
    time_frame_dict = {v: k for k, v in  frame_time_dict.items()} #Invert The Dictionary

    get_average_behaviour_trace(home_directory, ai_recorder_data, responses_save_location, time_frame_dict, condition_names[0])
    get_average_behaviour_trace(home_directory, ai_recorder_data, responses_save_location, time_frame_dict, condition_names[1])


"""
home_directory = r"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Analysis/71.2A_2020_11_17_Discrimination"
ai_recorder_name = "/20201117-121825.h5"
condition_names = ["All Vis 1", "All Vis 2"]
extract_trial_aligned_behaviour(home_directory, ai_recorder_name, condition_names)
"""