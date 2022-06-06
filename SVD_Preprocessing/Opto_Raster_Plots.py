import os
import h5py
import numpy as np
import sys
from scipy.io import loadmat
import matplotlib.pyplot as plt

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions



def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def get_stimuli_indexes(stimuli_pool, selected_stimulus):

    stimuli_indexes = []
    for x in range(len(stimuli_pool)):
        stimuli = stimuli_pool[x]
        if stimuli == selected_stimulus:
            stimuli_indexes.append(x)

    return stimuli_indexes


def get_stim_log_file(base_directory):

    file_list = os.listdir(base_directory)

    for file in file_list:
        if file[0:10] == 'opto_stim_':
            return file


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def match_frames(stimuli_onsets, frame_onsets):

    nearest_frames = []
    frame_onsets_array = np.array(frame_onsets)

    for onset in stimuli_onsets:
        nearest_frame_onset = find_nearest(frame_onsets_array, onset)
        nearest_frame_index = frame_onsets.index(nearest_frame_onset)
        nearest_frames.append(nearest_frame_index)

    nearest_frames = np.array(nearest_frames)
    return nearest_frames


def get_opto_raster(base_directory):

    # Load Delfa F Data
    delta_f_file = os.path.join(base_directory, "Delta_F.hdf5")
    delta_f_file_container = h5py.File(delta_f_file, 'r')
    delta_f_matrix = delta_f_file_container['Data']
    delta_f_matrix = np.array(delta_f_matrix)
    print("Delta F MAtrix Shape", np.shape(delta_f_matrix))

    # Load Ai File
    ai_recorder_filename = Widefield_General_Functions.get_ai_filename(base_directory)
    ai_filepath = base_directory + ai_recorder_filename
    ai_data = Widefield_General_Functions.load_ai_recorder_file(ai_filepath)

    # Extract Required Traces
    stimuli_dictionary = Widefield_General_Functions.create_stimuli_dictionary()
    opto_trace = ai_data[stimuli_dictionary['Optogenetics']]
    frame_trace = ai_data[stimuli_dictionary["LED 1"]]

    # Get Opto Onsets
    threshold = 2
    opto_onsets = Widefield_General_Functions.get_step_onsets(opto_trace, threshold=threshold, window=1000)

    # Get Frame Onsets
    frame_onsets = Widefield_General_Functions.get_step_onsets(frame_trace)

    # Get Matching Frames
    opto_onset_frames = match_frames(opto_onsets, frame_onsets)

    # Split Into Trial Type
    stim_log_file = get_stim_log_file(base_directory)
    stim_log = loadmat(os.path.join(base_directory, stim_log_file))
    stim_pool = stim_log['opto_session_data'][0][0][0]
    number_of_stimuli = np.max(stim_pool)
    print("number of stimuli", number_of_stimuli)


    trial_start = -100
    trial_stop = 100
    time_list = list(range(-100, 100, 25))

    for stimuli_index in range(1, number_of_stimuli + 1):

        print("Stimuli: ", stimuli_index)
        output_directory = os.path.join(base_directory, "Stimuli_" + str(stimuli_index))
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)

        stimuli_indexes = get_stimuli_indexes(stim_pool, stimuli_index)
        stimuli_onsets = opto_onset_frames[stimuli_indexes]

        # Create Tensor
        trial_frames = Widefield_General_Functions.get_selected_widefield_frames(stimuli_onsets, trial_start, trial_stop)
        trial_tensor = Widefield_General_Functions.get_selected_widefield_data(trial_frames, delta_f_matrix)
        print("Trial Tensor Shape", np.shape(trial_tensor))

        # Get Mean Response
        mean_response = np.mean(trial_tensor, axis=0)

        # Transpose
        mean_response = np.transpose(mean_response)

        # Plot Raster
        figure_1 = plt.figure()
        axis_1 = figure_1.add_subplot(1,1,1)
        axis_1.imshow(mean_response)
        #axis_1.set_xticks(time_list)
        axis_1.axvline(x=100)
        forceAspect(axis_1)

        plt.savefig(os.path.join(base_directory, "Stim_" + str(stimuli_index) + "_raster.png"))
        plt.close()
        #np.save(os.path.join(output_directory, "mean_response.npy"), mean_response)
        print("Mean Response Shape", np.shape(mean_response))
        """
        # Get Mask Details
        indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)

        plt.ion()

        for frame_index in range(0, trial_stop - trial_start):
            plt.title(str(frame_index + trial_start))
            frame_data = mean_response[frame_index]
            frame_image = Widefield_General_Functions.create_image_from_data(frame_data, indicies, image_height, image_width)

            plt.imshow(frame_image, cmap='jet', vmin=0, vmax=1)
            plt.draw()
            plt.pause(0.1)
            plt.savefig(os.path.join(output_directory, str(frame_index).zfill(3) + ".png"))
            plt.clf()
        """
    """
    plt.plot(frame_trace, alpha=0.2)
    plt.plot(opto_trace, alpha=0.3)
    plt.scatter(opto_onsets, np.multiply(np.ones(len(opto_onsets)), np.max(opto_trace)), c='r')

    plt.show()
    """


session_list = [r"/media/matthew/Expansion/Widefield_Analysis/KCVP_1.1H/2022_05_17_Opto_Test_No_Filter",
                r"/media/matthew/Expansion/Widefield_Analysis/KCPV_1.1D/2022_05_17_Opto_Test_No_Filter",
                r"/media/matthew/Expansion/Widefield_Analysis/KCPV_1.1E/2022_05_12_Opto_Test_No_Filter",
                r"/media/matthew/Expansion/Widefield_Analysis/KCVP_1.1G/2022_05_09_Opto_Test_No_Filter"]

session_list = [r"/media/matthew/Expansion/Widefield_Analysis/KCVP_1.1H/2022_05_09_Opto_Test_Filter",
                r"/media/matthew/Expansion/Widefield_Analysis/KCPV_1.1D/2022_05_12_Opto_Test_No_Filter",
                r"/media/matthew/Expansion/Widefield_Analysis/KCPV_1.1E/2022_05_11_Opto_Test_Filter",
                r"/media/matthew/Expansion/Widefield_Analysis/KCVP_1.1G/2022_07_05_Opto_Test"]


for base_directory in session_list:
    get_opto_raster(base_directory)

