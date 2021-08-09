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


def create_stimuli_dictionary(legacy=False):

    if legacy == False:
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

    #Round 1 Channel Index Dictionary

    return channel_index_dictionary


def get_step_onsets(trace, threshold=1, window=10):
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

    return onset_times, onset_line



def get_frame_indexes(frame_stream):

    frame_indexes = {}
    state = 1
    threshold = 2
    count = 0

    for timepoint in range(0, len(frame_stream)):

        if frame_stream[timepoint] > threshold:
            if state == 0:
                state = 1
                frame_indexes[timepoint] = count
                count += 1

        else:
            if state == 1:
                state = 0
            else:
                pass

    return frame_indexes


def split_stream_by_context(stimuli_onsets, context_onsets, context_window):
    context_negative_onsets = []
    context_positive_onsets = []

    # Iterate Through Visual 1 Onsets
    for stimuli_onset in stimuli_onsets:
        context = False
        window_start = stimuli_onset
        window_end = stimuli_onset + context_window

        for context_onset in context_onsets:
            if context_onset >= window_start and context_onset <= window_end:
                context = True

        if context == True:
            context_positive_onsets.append(stimuli_onset)
        else:
            context_negative_onsets.append(stimuli_onset)

    return context_negative_onsets, context_positive_onsets


def split_visual_onsets_by_context(visual_1_onsets, visual_2_onsets, odour_1_onsets, odour_2_onsets, following_window_size=7000):

    combined_odour_onsets = odour_1_onsets + odour_2_onsets
    visual_block_stimuli_1, odour_block_stimuli_1 = split_stream_by_context(visual_1_onsets, combined_odour_onsets, following_window_size)
    visual_block_stimuli_2, odour_block_stimuli_2 = split_stream_by_context(visual_2_onsets, combined_odour_onsets, following_window_size)

    onsets_list = [visual_block_stimuli_1, visual_block_stimuli_2, odour_block_stimuli_1, odour_block_stimuli_2]

    return onsets_list


def split_trials_by_lick(onsets, lick_trace, lick_threshold=0.1, lick_window=3000): #lick threshold = 0.1

    onsets_with_lick = []
    onsets_without_lick = []

    for onset in onsets:
        following_lick_window =lick_trace[onset:onset+lick_window]
        if np.max(following_lick_window) > lick_threshold:
            onsets_with_lick.append(onset)
        else:
            onsets_without_lick.append(onset)

    return onsets_with_lick, onsets_without_lick


def get_stable_visual_trials(correct_vis_1_onsets, correct_vis_2_onsets, combined_visual_onsets, combined_odour_onsets):

    preceeding_window = 25000
    following_window = 25000

    stable_vis_1_onsets = []
    stable_vis_2_onsets = []

    #Get Stable Visual Onsets
    for onset in correct_vis_1_onsets:
        is_preceeded    = False
        is_followed     = False
        no_olfactory    = True

        start_window = onset - preceeding_window
        stop_window  = onset + following_window

        # Chek Is Preceeded and Followed by another visual stimuli
        for other_onset in combined_visual_onsets:
            if other_onset > start_window and other_onset < onset:
                is_preceeded = True

            if other_onset > onset and other_onset < stop_window:
                is_followed = True

        # Check no olfactory Stimuli
        for other_onset in combined_odour_onsets:
            if other_onset > start_window and other_onset < following_window:
                no_olfactory = False

        # If All Criteria Are Met, Trial is Stable
        if is_preceeded and is_followed and no_olfactory:
            stable_vis_1_onsets.append(onset)

    #Get Stable Visual Onsets
    for onset in correct_vis_2_onsets:
        is_preceeded    = False
        is_followed     = False
        no_olfactory    = True

        start_window = onset - preceeding_window
        stop_window  = onset + following_window

        # Chek Is Preceeded and Followed by another visual stimuli
        for other_onset in combined_visual_onsets:
            if other_onset > start_window and other_onset < onset:
                is_preceeded = True

            if other_onset > onset and other_onset < stop_window:
                is_followed = True

        # Check no olfactory Stimuli
        for other_onset in combined_odour_onsets:
            if other_onset > start_window and other_onset < following_window:
                no_olfactory = False

        # If All Criteria Are Met, Trial is Stable
        if is_preceeded and is_followed and no_olfactory:
            stable_vis_2_onsets.append(onset)

    return stable_vis_1_onsets, stable_vis_2_onsets





def get_stable_odour_trials(correct_odour_1_onsets, correct_odour_2_onsets, combined_odour_onsets, lick_trace):

    preceeding_window      = 25000
    following_window       = 25000
    preceeding_lick_window = 3500
    following_lick_window  = 2000
    lick_threshold = 1

    stable_odour_1_onsets = []
    stable_odour_2_onsets = []

    #Get Stable Odour 1 onsets
    for onset in correct_odour_1_onsets:
        is_preceeded    = False
        is_followed     = False
        ignored_visual  = False

        # Check Is Preceeded and Followed by an Odour Stimuli
        start_window = onset - preceeding_window
        stop_window  = onset + following_window
        for other_onset in combined_odour_onsets:
            if other_onset > start_window and other_onset < onset:
                is_preceeded = True

            if other_onset > onset and other_onset < stop_window:
                is_followed = True

        #Check no lick to visual stimuli
        preceeding_lick_trace = lick_trace[onset-preceeding_lick_window:onset]
        if np.max(preceeding_lick_trace) < lick_threshold:
            ignored_visual = True

        if is_preceeded and is_followed and ignored_visual:
            stable_odour_1_onsets.append(onset)


    #Get Stable Odour 2 onsets
    for onset in correct_odour_2_onsets:
        is_preceeded    = False
        is_followed     = False
        ignored_visual  = False

        # Check Is Preceeded and Followed by an Odour Stimuli
        start_window = onset - preceeding_window
        stop_window  = onset + following_window
        for other_onset in combined_odour_onsets:
            if other_onset > start_window and other_onset < onset:
                is_preceeded = True

            if other_onset > onset and other_onset < stop_window:
                is_followed = True

        #Check no lick to visual stimuli
        preceeding_lick_trace = lick_trace[onset-preceeding_lick_window:onset]
        if np.max(preceeding_lick_trace) < lick_threshold:
            ignored_visual = True

        if is_preceeded and is_followed and ignored_visual:
            stable_odour_2_onsets.append(onset)


    return stable_odour_1_onsets, stable_odour_2_onsets


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



def get_visual_onsets_in_stable_odour_trials(visual_1_onsets, visual_2_onsets, stable_odour_1_onsets, stable_odour_2_onsets):

    following_window_size = 5000

    combined_stable_odour_onsets = stable_odour_1_onsets + stable_odour_2_onsets
    vis_1_onsets_in_stable_odour_trials = []
    vis_2_onsets_in_stable_odour_trials = []

    #Get Vis 1 onsets in stable odour trials
    for visual_onset in visual_1_onsets:
        following_window = visual_onset + following_window_size

        for odour_onset in combined_stable_odour_onsets:
            if odour_onset > visual_onset and odour_onset <= following_window:
                vis_1_onsets_in_stable_odour_trials.append(visual_onset)

    # Get Vis 2 onsets in stable odour trials
    for visual_onset in visual_2_onsets:
        following_window = visual_onset + following_window_size

        for odour_onset in combined_stable_odour_onsets:
            if odour_onset > visual_onset and odour_onset <= following_window:
                vis_2_onsets_in_stable_odour_trials.append(visual_onset)

    return vis_1_onsets_in_stable_odour_trials, vis_2_onsets_in_stable_odour_trials

def normalise_trace(trace):
    trace = np.divide(trace, np.max(trace))
    return trace



def visualise_onsets(onsets_list, traces_list, colour_list=['y', 'b', 'r', 'g', 'm']):

    for onset_type in onsets_list:
        onsets     = onset_type[0]
        onset_name = onset_type[1]

        plt.title(onset_name)

        for trace_index in range(len(traces_list)):
            trace = traces_list[trace_index]
            colour = colour_list[trace_index]
            plt.plot(trace, c=colour)

        plt.scatter(onsets, np.ones(len(onsets))*np.max(traces_list))
        plt.show()



def determine_trial_type(home_directory):
    trial_type = None
    home_directory_list = home_directory.split("/")
    print(home_directory_list)
    if "Discrimination" in home_directory_list[-1]:
        trial_type = "Discrimination"
    elif "Switching" in home_directory_list[-1]:
        trial_type = "Switching"
    elif "Transition" in home_directory_list[-1]:
        trial_type = "Transition"

    return trial_type

def visualise_raw_traces(ai_recorder_data):
    number_of_traces = np.shape(ai_recorder_data)[0]

    for trace in range(number_of_traces):
        plt.title(trace)
        plt.plot(ai_recorder_data[trace])
        plt.show()



def exclude_trial_outside_imaging_window(onsets_list, first_frame_time, last_frame_time, buffer_window=5000):
    included_onsets = []

    for onset in onsets_list:
        if onset > (first_frame_time + buffer_window) and onset < (last_frame_time - buffer_window):
            included_onsets.append(onset)

    return included_onsets


def get_closest(list, value):
    return min(list, key=lambda x: abs(x - value))


def classify_trials(home_directory, ai_recorder_file):

    # Create Output Folder
    save_directory = home_directory + "/" + "Stimuli_Onsets"
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Determine Trial Type
    trial_type = determine_trial_type(home_directory)
    print("Trial Type", trial_type)

    # Load AI Recorder Data
    ai_recorder_data = load_ai_recorder_file(home_directory + ai_recorder_file)

    # Get Visual and Odour Onsets
    channel_index_dictionary = create_stimuli_dictionary(legacy=False)

    lick_channel     = channel_index_dictionary["Lick"]
    visual_1_channel = channel_index_dictionary["Visual 1"]
    visual_2_channel = channel_index_dictionary["Visual 2"]
    odour_1_channel  = channel_index_dictionary["Odour 1"]
    odour_2_channel  = channel_index_dictionary["Odour 2"]
    frame_channel    = channel_index_dictionary["LED 1"]
    mousecam_channel = channel_index_dictionary["Mousecam"]

    # Get Frame Indexes
    frame_onsets = get_frame_indexes(ai_recorder_data[frame_channel])
    np.save(save_directory + "/Frame_Times.npy", frame_onsets)

    # Get Times Of the First And Last Camera Frames
    keys_list = list(frame_onsets.keys())
    first_frame_time = keys_list[0]
    last_frame_time  = keys_list[-1]
    #print("First Frame Time", first_frame_time)
    #print("Last rame Time", last_frame_time)

    #Extract Traces
    lick_trace = ai_recorder_data[lick_channel]
    vis_1_trace = ai_recorder_data[visual_1_channel]
    vis_2_trace = ai_recorder_data[visual_2_channel]
    odour_1_trace = ai_recorder_data[odour_1_channel]
    odour_2_trace = ai_recorder_data[odour_2_channel]

    # Get All Stimuli Onsets
    visual_1_onsets, visual_1_line = get_step_onsets(ai_recorder_data[visual_1_channel])
    visual_2_onsets, visual_2_line = get_step_onsets(ai_recorder_data[visual_2_channel])
    odour_1_onsets, odour_1_line   = get_step_onsets(ai_recorder_data[odour_1_channel])
    odour_2_onsets, odour_2_line   = get_step_onsets(ai_recorder_data[odour_2_channel])

    print("Number of vis 1 trials: ", len(visual_1_onsets))
    print("Number of vis 2 trials: ", len(visual_2_onsets))

    # Debug Code
    """
    plt.title(home_directory + "Visual Onsets")
    plt.plot(vis_1_trace, c='g')
    plt.plot(vis_2_trace, c='b')

    vis_1_y_values = np.multiply(np.ones(len(visual_1_onsets)), np.max(vis_1_trace))
    vis_2_y_values = np.multiply(np.ones(len(visual_2_onsets)), np.max(vis_2_trace))

    plt.scatter(visual_1_onsets, vis_1_y_values, c='g')
    plt.scatter(visual_2_onsets, vis_2_y_values, c='b')
    #plt.plot(ai_recorder_data[mousecam_channel], c='m')
    plt.show()
    """

    # Exclude Any Trials Outside The Imaging Window
    visual_1_onsets = exclude_trial_outside_imaging_window(visual_1_onsets, first_frame_time, last_frame_time)
    visual_2_onsets = exclude_trial_outside_imaging_window(visual_2_onsets, first_frame_time, last_frame_time)
    odour_1_onsets  = exclude_trial_outside_imaging_window(odour_1_onsets,  first_frame_time, last_frame_time)
    odour_2_onsets  = exclude_trial_outside_imaging_window(odour_2_onsets,  first_frame_time, last_frame_time)

    combined_odour_onsets = odour_1_onsets + odour_2_onsets
    combined_visual_onsets = visual_1_onsets + visual_2_onsets

    if trial_type == "Discrimination":
        print("Discrimination Session")
        correct_visual_1, incorrect_visual_1 = split_trials_by_lick(visual_1_onsets, ai_recorder_data[lick_channel])
        incorrect_visual_2, correct_visual_2 = split_trials_by_lick(visual_2_onsets, ai_recorder_data[lick_channel])

        # Get Nearest Frames
        all_vis_1_frames       = get_nearest_frame(visual_1_onsets,    frame_onsets)
        all_vis_2_frames       = get_nearest_frame(visual_2_onsets,    frame_onsets)
        correct_vis_1_frames   = get_nearest_frame(correct_visual_1,   frame_onsets)
        correct_vis_2_frames   = get_nearest_frame(correct_visual_2,   frame_onsets)
        incorrect_vis_1_frames = get_nearest_frame(incorrect_visual_1, frame_onsets)
        incorrect_vis_2_frames = get_nearest_frame(incorrect_visual_2, frame_onsets)

        # Save Frame Indexes
        np.save(save_directory + "/All_vis_1_frame_indexes.npy",        all_vis_1_frames)
        np.save(save_directory + "/All_vis_2_frame_indexes.npy",        all_vis_2_frames)
        np.save(save_directory + "/Correct_vis_1_frame_indexes.npy",    correct_vis_1_frames)
        np.save(save_directory + "/Correct_vis_2_frame_indexes.npy",    correct_vis_2_frames)
        np.save(save_directory + "/Incorrect_vis_1_frame_indexes.npy",  incorrect_vis_1_frames)
        np.save(save_directory + "/Incorrect_vis_2_frame_indexes.npy",  incorrect_vis_2_frames)

        #print("all bis 1 frames", all_vis_1_frames)
        #print("all vis 2 frames", all_vis_2_frames)


    elif trial_type == "Switching":

        # Split Visual Onsets By Context
        visual_onsets_by_context = split_visual_onsets_by_context(visual_1_onsets, visual_2_onsets, odour_1_onsets, odour_2_onsets)
        visual_context_all_vis_1 = visual_onsets_by_context[0]
        visual_context_all_vis_2 = visual_onsets_by_context[1]
        odour_context_all_vis_1  = visual_onsets_by_context[2]
        odour_context_all_vis_2  = visual_onsets_by_context[3]


        # Split Onsets by Lick
        visual_context_correct_vis_1,   visual_context_incorrect_vis_1 = split_trials_by_lick(visual_context_all_vis_1, lick_trace)
        visual_context_incorrect_vis_2, visual_context_correct_vis_2   = split_trials_by_lick(visual_context_all_vis_2,  lick_trace)

        odour_context_incorrect_vis_1, odour_context_correct_vis_1     = split_trials_by_lick(odour_context_all_vis_1, lick_trace, lick_window=2000)
        odour_context_incorrect_vis_2, odour_context_correct_vis_2     = split_trials_by_lick(odour_context_all_vis_2, lick_trace, lick_window=2000)

        correct_odour_1, incorrect_odour_1                             = split_trials_by_lick(odour_1_onsets,  lick_trace)
        incorrect_odour_2, correct_odour_2                             = split_trials_by_lick(odour_2_onsets,  lick_trace)


        # Get Nearest Frames
        visual_context_correct_vis_1_frames     = get_nearest_frame(visual_context_correct_vis_1,    frame_onsets)
        visual_context_correct_vis_2_frames     = get_nearest_frame(visual_context_correct_vis_2,    frame_onsets)
        visual_context_incorrect_vis_1_frames   = get_nearest_frame(visual_context_incorrect_vis_1,  frame_onsets)
        visual_context_incorrect_vis_2_frames   = get_nearest_frame(visual_context_incorrect_vis_2,  frame_onsets)
        odour_context_correct_vis_1_frames      = get_nearest_frame(odour_context_correct_vis_1,     frame_onsets)
        odour_context_correct_vis_2_frames      = get_nearest_frame(odour_context_correct_vis_2,     frame_onsets)
        odour_context_incorrect_vis_1_frames    = get_nearest_frame(odour_context_incorrect_vis_1,   frame_onsets)
        odour_context_incorrect_vis_2_frames    = get_nearest_frame(odour_context_incorrect_vis_2,   frame_onsets)
        correct_odour_1_frames                  = get_nearest_frame(correct_odour_1,                 frame_onsets)
        correct_odour_2_frames                  = get_nearest_frame(correct_odour_2,                 frame_onsets)
        incorrect_odour_1_frames                = get_nearest_frame(incorrect_odour_1,               frame_onsets)
        incorrect_odour_2_frames                = get_nearest_frame(incorrect_odour_2,               frame_onsets)

        #Visualise Onsets By Outcome
        onsets_list = [
            [visual_context_correct_vis_1,      "visual_context_correct_vis_1"],
            [visual_context_correct_vis_2,      "visual_context_correct_vis_2"],
            [visual_context_incorrect_vis_1,    "visual_context_incorrect_vis_1"],
            [visual_context_incorrect_vis_2,    "visual_context_incorrect_vis_2"],
            [odour_context_correct_vis_1,       "odour_context_correct_vis_1"],
            [odour_context_correct_vis_2,       "odour_context_correct_vis_2"],
            [odour_context_incorrect_vis_1,     "odour_context_incorrect_vis_1"],
            [odour_context_incorrect_vis_2,     "odour_context_incorrect_vis_2"],
            [correct_odour_1,                   "correct_odour_1"],
            [correct_odour_2,                   "correct_odour_2"],
            [incorrect_odour_1,                 "incorrect_odour_1"],
            [incorrect_odour_2,                 "incorrect_odour_2"],
            ]

        traces_list = [ lick_trace,
                        vis_1_trace,
                        vis_2_trace,
                        odour_1_trace,
                        odour_2_trace]

        traces_list = np.array(traces_list)

        #visualise_onsets(onsets_list, traces_list)

        # Save Frame Indexes
        np.save(save_directory + "/visual_context_correct_vis_1_frames.npy",    visual_context_correct_vis_1_frames)
        np.save(save_directory + "/visual_context_correct_vis_2_frames.npy",    visual_context_correct_vis_2_frames)
        np.save(save_directory + "/visual_context_incorrect_vis_1_frames.npy",  visual_context_incorrect_vis_1_frames)
        np.save(save_directory + "/visual_context_incorrect_vis_2_frames.npy",  visual_context_incorrect_vis_2_frames)
        np.save(save_directory + "/odour_context_correct_vis_1_frames.npy",     odour_context_correct_vis_1_frames)
        np.save(save_directory + "/odour_context_correct_vis_2_frames.npy",     odour_context_correct_vis_2_frames)
        np.save(save_directory + "/odour_context_incorrect_vis_1_frames.npy",   odour_context_incorrect_vis_1_frames)
        np.save(save_directory + "/odour_context_incorrect_vis_2_frames.npy",   odour_context_incorrect_vis_2_frames)
        np.save(save_directory + "/correct_odour_1_frames.npy",                 correct_odour_1_frames)
        np.save(save_directory + "/correct_odour_2_frames.npy",                 correct_odour_2_frames)
        np.save(save_directory + "/incorrect_odour_1_frames.npy",               incorrect_odour_1_frames)
        np.save(save_directory + "/incorrect_odour_2_frames.npy",               incorrect_odour_2_frames)

    if trial_type == "Transition":
        print("Transition Session")

        # Split Visual Onsets By Context
        visual_onsets_by_context = split_visual_onsets_by_context(visual_1_onsets, visual_2_onsets, odour_1_onsets, odour_2_onsets)
        visual_context_all_vis_1 = visual_onsets_by_context[0]
        visual_context_all_vis_2 = visual_onsets_by_context[1]
        odour_context_all_vis_1 = visual_onsets_by_context[2]
        odour_context_all_vis_2 = visual_onsets_by_context[3]

        # Split Onsets by Lick
        visual_context_correct_vis_1, visual_context_incorrect_vis_1 = split_trials_by_lick(visual_context_all_vis_1, lick_trace)
        visual_context_incorrect_vis_2, visual_context_correct_vis_2 = split_trials_by_lick(visual_context_all_vis_2, lick_trace)

        odour_context_incorrect_vis_1, odour_context_correct_vis_1 = split_trials_by_lick(odour_context_all_vis_1, lick_trace, lick_window=2000)
        odour_context_incorrect_vis_2, odour_context_correct_vis_2 = split_trials_by_lick(odour_context_all_vis_2, lick_trace, lick_window=2000)

        correct_odour_1, incorrect_odour_1 = split_trials_by_lick(odour_1_onsets, lick_trace)
        incorrect_odour_2, correct_odour_2 = split_trials_by_lick(odour_2_onsets, lick_trace)

        # Get Nearest Frames
        visual_context_correct_vis_1_frames = get_nearest_frame(visual_context_correct_vis_1, frame_onsets)
        visual_context_correct_vis_2_frames = get_nearest_frame(visual_context_correct_vis_2, frame_onsets)
        visual_context_incorrect_vis_1_frames = get_nearest_frame(visual_context_incorrect_vis_1, frame_onsets)
        visual_context_incorrect_vis_2_frames = get_nearest_frame(visual_context_incorrect_vis_2, frame_onsets)
        odour_context_correct_vis_1_frames = get_nearest_frame(odour_context_correct_vis_1, frame_onsets)
        odour_context_correct_vis_2_frames = get_nearest_frame(odour_context_correct_vis_2, frame_onsets)
        odour_context_incorrect_vis_1_frames = get_nearest_frame(odour_context_incorrect_vis_1, frame_onsets)
        odour_context_incorrect_vis_2_frames = get_nearest_frame(odour_context_incorrect_vis_2, frame_onsets)
        correct_odour_1_frames = get_nearest_frame(correct_odour_1, frame_onsets)
        correct_odour_2_frames = get_nearest_frame(correct_odour_2, frame_onsets)
        incorrect_odour_1_frames = get_nearest_frame(incorrect_odour_1, frame_onsets)
        incorrect_odour_2_frames = get_nearest_frame(incorrect_odour_2, frame_onsets)

        # Visualise Onsets By Outcome
        onsets_list = [
            [visual_context_correct_vis_1, "visual_context_correct_vis_1"],
            [visual_context_correct_vis_2, "visual_context_correct_vis_2"],
            [visual_context_incorrect_vis_1, "visual_context_incorrect_vis_1"],
            [visual_context_incorrect_vis_2, "visual_context_incorrect_vis_2"],
            [odour_context_correct_vis_1, "odour_context_correct_vis_1"],
            [odour_context_correct_vis_2, "odour_context_correct_vis_2"],
            [odour_context_incorrect_vis_1, "odour_context_incorrect_vis_1"],
            [odour_context_incorrect_vis_2, "odour_context_incorrect_vis_2"],
            [correct_odour_1, "correct_odour_1"],
            [correct_odour_2, "correct_odour_2"],
            [incorrect_odour_1, "incorrect_odour_1"],
            [incorrect_odour_2, "incorrect_odour_2"],
        ]

        traces_list = [lick_trace,
                       vis_1_trace,
                       vis_2_trace,
                       odour_1_trace,
                       odour_2_trace]

        traces_list = np.array(traces_list)

        combined_odour_trace = np.add(odour_1_trace, odour_2_trace)
        odour_context_all_vis_1_offset = turn_onsets_to_offsets(odour_context_all_vis_1, vis_1_trace)
        odour_context_all_vis_2_offset = turn_onsets_to_offsets(odour_context_all_vis_2, vis_2_trace)


        # Get Typical Timing
        delay_distribution = []
        for onset in odour_context_all_vis_1_offset:
            closest = get_closest(combined_odour_onsets, onset)
            delay = closest - onset
            delay_distribution.append(delay)

        for onset in odour_context_all_vis_2_offset:
            closest = get_closest(combined_odour_onsets, onset)
            delay = closest - onset
            delay_distribution.append(delay)

        plt.hist(delay_distribution)
        plt.show()


        # Get Vis 1 Transition Onsets
        preceeding_odour_window = 10000
        transition_onsets = []
        for onset in visual_context_all_vis_1:
            odour_window = combined_odour_trace[onset-preceeding_odour_window:onset]
            if np.max(odour_window) > 1:
                transition_onsets.append(onset)

        #turn onsets to offsets
        transition_offsets = turn_onsets_to_offsets(transition_onsets, vis_1_trace)



        transition_frames = get_nearest_frame(transition_offsets, frame_onsets)
        np.save(save_directory + "/transition_frames.npy", transition_frames)

        x_values = list(range(len(vis_1_trace)))
        plt.plot(x_values, combined_odour_trace, c='g')
        plt.plot(x_values, vis_1_trace, c='b')
        plt.scatter(transition_offsets, np.ones(len(transition_offsets)))
        plt.show()


def turn_onsets_to_offsets(onsets, trace):
    step_size = 1

    offsets = []
    for onset in onsets:
        searching = True
        timepoint = onset
        initial_value = trace[onset]
        while searching:
            value = trace[timepoint]
            difference = initial_value - value
            if difference > step_size:
                offsets.append(timepoint)
                searching = False
            else:
                timepoint += 1

    return offsets




