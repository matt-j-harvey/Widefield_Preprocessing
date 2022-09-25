import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions



def get_output_directory(base_directory, output_stem):

    split_base_directory = base_directory.split("/")

    # Check Mouse Directory
    mouse_directory = os.path.join(output_stem, split_base_directory[-2])
    if not os.path.exists(mouse_directory):
        os.mkdir(mouse_directory)

    # Check Session Directory
    session_directory = os.path.join(mouse_directory, split_base_directory[-1])
    if not os.path.exists(session_directory):
        os.mkdir(session_directory)

    return session_directory



def get_closest_preceeding_frame(selected_time, onset_list):

    preceeding_time_distances = []
    time_distances = []

    # Get all Time Distances
    for onset in onset_list:
        time_difference = selected_time - onset
        time_distances.append(time_difference)

        if time_difference >= 0:
            preceeding_time_distances.append(time_difference)

    if len(preceeding_time_distances) > 0:

        # Get Smallest Time Difference
        smallest_time_difference = np.min(preceeding_time_distances)

        # Get Closest Frame
        closest_preceeding_frame = time_distances.index(smallest_time_difference)

        return closest_preceeding_frame

    # Is There Are No Preceeding Frames Do Not Include This Stimuli
    else:
        return False



def get_preceeding_frames(frame_onset_list, stimuli_onset_list, preceeding_window=25):

    # Sort Lists
    frame_onset_list = list(frame_onset_list)
    stimuli_onset_list = list(stimuli_onset_list)

    frame_onset_list.sort()
    stimuli_onset_list.sort()

    preceeding_frame_indexes = []

    for stimuli_onset in stimuli_onset_list:
        closest_frame = get_closest_preceeding_frame(stimuli_onset, frame_onset_list)

        if closest_frame != False:
            window_start = closest_frame - preceeding_window

            if window_start < 0:
                window_start = 0

            for x in range(window_start, closest_frame):
                preceeding_frame_indexes.append(x)

    return preceeding_frame_indexes


def get_baseline_frames(base_directory, output_directory):

    # Function to Get Blue and Violet Frames 1 Second Before any visual stimuli Onsets

    # Load Ai Recorder File
    ai_file = Widefield_General_Functions.get_ai_filename(base_directory)
    ai_file_location = base_directory + ai_file
    ai_data = Widefield_General_Functions.load_ai_recorder_file(ai_file_location)

    # Create Stimuli Dictionary
    stimuli_dictionary = Widefield_General_Functions.create_stimuli_dictionary()

    # Extract Blue and Violet Traces as Well As Vis Stim Traces
    blue_led_trace = ai_data[stimuli_dictionary["LED 1"]]
    violet_led_trace = ai_data[stimuli_dictionary["LED 2"]]
    vis_1_trace = ai_data[stimuli_dictionary["Visual 1"]]
    vis_2_trace = ai_data[stimuli_dictionary["Visual 2"]]

    # Get Blue and Violet Frame Indexes
    blue_led_onsets = Widefield_General_Functions.get_step_onsets(blue_led_trace)
    violet_led_onsets = Widefield_General_Functions.get_step_onsets(violet_led_trace)
    """
    plt.plot(blue_led_trace, c='b')
    plt.plot(violet_led_trace, c='m')
    plt.scatter(blue_led_onsets, np.ones(len(blue_led_onsets)) * 3, c='b')
    plt.scatter(violet_led_onsets, np.ones(len(violet_led_onsets)) * 3, c='m')
    plt.show()
    """

    # Get All Visual Stimuli Frame Indexes
    vis_1_onsets = Widefield_General_Functions.get_step_onsets(vis_1_trace)
    vis_2_onsets = Widefield_General_Functions.get_step_onsets(vis_2_trace)

    """
    plt.plot(vis_1_trace, c='b')
    plt.plot(vis_2_trace, c='r')
    plt.scatter(vis_1_onsets, np.ones(len(vis_1_onsets)) * 3, c='b')
    plt.scatter(vis_2_onsets, np.ones(len(vis_2_onsets)) * 3, c='r')
    plt.show()
    """

    # Get The 25 Frame Indexes Prior To Each Visual Stimulus
    all_visual_onsets = vis_1_onsets + vis_2_onsets
    blue_preceeding_frame_indexes = get_preceeding_frames(blue_led_onsets, all_visual_onsets, preceeding_window=25)
    violet_preceeding_frame_indexes = get_preceeding_frames(violet_led_onsets, all_visual_onsets, preceeding_window=25)

    """
    plt.plot(vis_1_trace, c='r')
    plt.plot(vis_2_trace, c='r')
    plt.plot(blue_led_trace, c='b')
    for frame in blue_preceeding_frame_indexes:
        frame_time = blue_led_onsets[frame]
        plt.scatter([frame_time], [3], c='k')

    plt.show()
    """

    # Save These Frame Indexes
    np.save(os.path.join(output_directory, "Blue_Baseline_Frames.npy"), blue_preceeding_frame_indexes)
    np.save(os.path.join(output_directory, "Violet_Baseline_Frames.npy"), violet_preceeding_frame_indexes)


