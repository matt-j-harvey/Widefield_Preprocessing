import os
import h5py
import numpy as np
import sys
import matplotlib.pyplot as plt



sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions



def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_trial_tensor(delta_f_matrix, onset_list, start_window, stop_window):

    print("Delta F matrix", np.shape(delta_f_matrix))

    trial_tensor = []
    for onset in onset_list:
        trial_start = onset + start_window
        trial_stop = onset + stop_window
        print("Trial start", trial_start)
        print("Trial stop", trial_stop)
        trial_data = delta_f_matrix[trial_start:trial_stop]
        trial_tensor.append(trial_data)

    trial_tensor = np.array(trial_tensor)
    print("Trial Tensor Shape", np.shape(trial_tensor))
    return trial_tensor


def match_frames(stimuli_onsets, frame_onsets):

    nearest_frames = []
    frame_onsets_array = np.array(frame_onsets)

    for onset in stimuli_onsets:
        nearest_frame_onset = find_nearest(frame_onsets_array, onset)
        nearest_frame_index = frame_onsets.index(nearest_frame_onset)
        nearest_frames.append(nearest_frame_index)

    nearest_frames = np.array(nearest_frames)
    return nearest_frames



def load_mask(home_directory):

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



def view_trial_tensor(trial_tensor, indicies, image_height, image_width, base_directory):

    # Get Average
    print("Trial Tensor Shape", np.shape(trial_tensor))
    average_trace = np.mean(trial_tensor, axis=0)

    number_of_timepoints = np.shape(trial_tensor)[1]

    interval = 36
    timelist = list(range(-100 * interval, 100 * interval,  interval))

    # Create Save Directory
    save_directory = os.path.join(base_directory, "Stimuli_Response")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    for timepoint in range(number_of_timepoints):

        timepoint_data = average_trace[timepoint]

        template = np.zeros(image_height * image_width)
        template[indicies] = timepoint_data


        template = np.reshape(template, (image_height, image_width))

        time_value = timelist[timepoint]
        plt.title(str(time_value))
        plt.imshow(template, vmin=0, vmax=1, cmap='jet')
        plt.axis('off')
        plt.savefig(os.path.join(save_directory, str(timepoint).zfill(3) + ".png"))
        plt.close()


def get_opto_led_average(base_directory):

    # Load Delta F File
    delta_f_file = os.path.join(base_directory, "Delta_F.hdf5")
    data_container = h5py.File(delta_f_file, 'r')
    data_matrix = data_container["Data"]
    print(np.shape(data_matrix))


    # Load AI Recorder File
    ai_filename = Widefield_General_Functions.get_ai_filename(base_directory)
    ai_data = Widefield_General_Functions.load_ai_recorder_file(base_directory + ai_filename)
    stimuli_dictionary = Widefield_General_Functions.create_stimuli_dictionary()
    optogenetic_trace = ai_data[stimuli_dictionary['Optogenetics']]

    # Get Opto Step Onsets
    optogenetic_step_onsets = Widefield_General_Functions.get_step_onsets(optogenetic_trace)
    #plt.plot(optogenetic_trace)
    #plt.scatter(x=optogenetic_step_onsets, y=np.ones(len(optogenetic_step_onsets)))
    #plt.show()

    # Get Camera Onsets
    led_trace = ai_data[stimuli_dictionary["LED 1"]]
    camera_onsets = Widefield_General_Functions.get_step_onsets(led_trace)

    # Match Opto Onsets To Camera Onsets
    led_stimulation_onsets = match_frames(optogenetic_step_onsets, camera_onsets)

    # Get Trial Tensor
    trial_start = -100
    trial_stop = 100
    trial_tensor = get_trial_tensor(data_matrix, led_stimulation_onsets, trial_start, trial_stop)

    # Load Mask
    indicies, image_height, image_width = load_mask(base_directory)

    # View Trial Tensor
    view_trial_tensor(trial_tensor, indicies, image_height, image_width, base_directory)

    #plt.plot(led_trace)
    #plt.scatter(x=camera_onsets, y=np.ones(len(camera_onsets)))
    #plt.show()

base_directory = r"/media/matthew/Expansion/Widefield_Analysis/KCPV_1.1E/2022_05_15_LED_Test"
get_opto_led_average(base_directory)