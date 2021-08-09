import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py
import os
import tables
from scipy import signal, ndimage, stats
from sklearn.neighbors import KernelDensity
import cv2
from matplotlib import gridspec, cm

def plot_trace(x_cords, trace):
    plt.plot(x_cords, trace[0], c='b')
    plt.fill_between(x=x_cords, y1= trace[0], y2= trace[1], color='b', alpha = 0.2)
    plt.fill_between(x=x_cords, y1= trace[0], y2= trace[2], color='b', alpha = 0.2)
    # plt.show()


def normalise_behavioural_matrix(behaviour_matrix, step_size):

    # Get Behaviour Matrix Details
    number_of_traces = np.shape(behaviour_matrix)[0]
    x_cords = list(range(np.shape(behaviour_matrix[0])[1]))

    # Scale Traces:
    for trace_index in range(number_of_traces):
        trace = behaviour_matrix[trace_index]
        max_value = np.max(trace)
        normalised_trace = np.divide(trace, max_value)
        behaviour_matrix[trace_index] = normalised_trace

    # Offset traces
    for trace_index in range(number_of_traces):
        trace = behaviour_matrix[trace_index]
        offset_trace = np.add(trace, step_size * trace_index)
        behaviour_matrix[trace_index] = offset_trace

    return behaviour_matrix


def create_image_from_data(data, image_height, image_width, indicies):
    template = np.zeros([image_height, image_width])
    data = np.nan_to_num(data)
    np.put(template, indicies, data)
    image = np.ndarray.reshape(template, (image_height, image_width))
    image = ndimage.gaussian_filter(image, 2)

    return image


def draw_temporal_window(number_of_timepoints, frame, frame_step, window_size, number_of_traces, trace_offset, axis):

    # Draw Temporal Window
    x_values = list(range(number_of_timepoints))
    window = np.zeros(number_of_timepoints)
    baseline = np.zeros(number_of_timepoints)
    time_window_start = int((frame * frame_step) - (window_size * frame_step))
    time_window_end = int((frame * frame_step) + (window_size * frame_step))

    if time_window_start < 0:
        time_window_start = 0
    if time_window_end > number_of_timepoints:
        time_window_end = number_of_timepoints

    window[time_window_start:time_window_end] = number_of_traces * trace_offset
    axis.fill_between(x=x_values, y2=baseline, y1=window, interpolate=True, alpha=0.5)


def load_mask(home_directory):

    mask = np.load(home_directory + "/mask.npy")
    mask = np.where(mask>0.1, 1, 0)
    mask = mask.astype(int)
    image_height = np.shape(mask)[0]
    image_width = np.shape(mask)[1]
    flat_mask = np.ndarray.flatten(mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    print("maks height", image_height)
    print("mask width", image_width)

    return indicies, image_height, image_width

def get_colour(input_value, colour_map, scale_factor):

    input_value = input_value * scale_factor
    cmap = cm.get_cmap(colour_map)
    colour = cmap(input_value)

    return colour


def combine_images_into_video(image_folder, video_name):

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), frameSize=(width, height), fps=15)  # 0, 12

    count = 0
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
        count += 1

    cv2.destroyAllWindows()
    video.release()
    print("Finished - Video Is: " ,video_name)



def joinly_scale_matricies(matrix_1, matrix_2, trace_offset):

    number_of_traces = np.shape(matrix_1)[0]

    for trace_index in range(number_of_traces):

        # Load Traces
        trace_1 = matrix_1[trace_index]
        trace_2 = matrix_2[trace_index]

        # Scale Traces
        trace_1, trace_2 = jointly_scale_traces(trace_1, trace_2)

        # Offset Traces
        trace_1 = np.add(trace_1, trace_index * trace_offset)
        trace_2 = np.add(trace_2, trace_index * trace_offset)

        # Save Back as Original Traces
        matrix_1[trace_index] = trace_1
        matrix_2[trace_index] = trace_2

    return matrix_1, matrix_2


def jointly_scale_traces(trace_1, trace_2):

    trace_1_min = np.min(trace_1)
    trace_2_min = np.min(trace_2)
    min_value = min([trace_1_min, trace_2_min])

    if min_value < 0:
        trace_1 = np.add(trace_1, abs(min_value))
        trace_2 = np.add(trace_2, abs(min_value))

    if min_value > 0:
        trace_1 = np.subtract(trace_1, min_value)
        trace_2 = np.subtract(trace_2, min_value)

    trace_1_max = np.max(trace_1)
    trace_2_max = np.max(trace_2)
    max_value = max([trace_1_max, trace_2_max])

    scaled_trace_1 = np.divide(trace_1, max_value)
    scaled_trace_2 = np.divide(trace_2, max_value)

    return scaled_trace_1, scaled_trace_2



def load_trial_details(stimulus_name):

    file_name = stimuli_evoked_responses_directory + "/" + stimulus_name + "/" + stimulus_name + "_Trial_Details.npy"
    trial_details = np.load(file_name)

    print("trial details shape", np.shape(trial_details))
    print(trial_details[0])



def scale_matrix(matrix, vmin, vmax):

    #print("matrix", np.max(matrix), np.min(matrix))
    #print("Vmax", vmax)
    #print("Vmin", vmin)

    #Clip
    matrix = np.clip(matrix, a_min=vmin, a_max=vmax)

    #Set Minimum to zero
    matrix = np.subtract(matrix, vmin)
    #print("Min should now be zero", np.min(matrix))
    #print("new max", np.max(matrix))


    #Set Max to 1
    new_vmax = np.max(matrix)
    #print("new vmax", new_vmax)
    matrix = np.divide(matrix, new_vmax)
    matrix = np.nan_to_num(matrix)
    #print("max should now be 1 - - ", np.max(matrix))

    return matrix


def reconstruct_comparison_video(home_directory, stimulus_names, traces_to_include, trial_data):

    #Get Time List
    trial_start = trial_data[0]
    trial_end = trial_data[1]
    time_list = list(range(trial_start, trial_end))



    stimuli_evoked_responses_directory = home_directory + "/Stimuli_Evoked_Responses"
    data_folder_1 = stimuli_evoked_responses_directory + "/" + stimulus_names[0]
    data_folder_2 = stimuli_evoked_responses_directory + "/" + stimulus_names[1]

    rotation = np.load(home_directory + "/rotation.npy")

    #Load Allen Atlas Boundary Data
    #boundary_data = np.load(home_directory + "/boundary_mapping.npy")


    # Check Output Folder Exists
    output_folder = stimuli_evoked_responses_directory + "/" + stimulus_names[0] + "_V_" + stimulus_names[1]
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)


    # Load Mask
    indicies, image_height, image_width = load_mask(home_directory)

    # Load Activity Matrix
    activity_matrix_1 = data_folder_1 + "/" + stimulus_names[0] + "_Activity_Matrix_Average.npy"
    activity_matrix_2 = data_folder_2 + "/" + stimulus_names[1] + "_Activity_Matrix_Average.npy"
    activity_matrix_1 = np.load(activity_matrix_1, allow_pickle=True)
    activity_matrix_2 = np.load(activity_matrix_2, allow_pickle=True)

    # Load Behaviour Matrix
    behaviour_matrix_1 = data_folder_1 + "/" + stimulus_names[0] + "_Behaviour_Matrix.npy"
    behaviour_matrix_2 = data_folder_2 + "/" + stimulus_names[1] + "_Behaviour_Matrix.npy"
    behaviour_matrix_1 = np.load(behaviour_matrix_1, allow_pickle=True)
    behaviour_matrix_2 = np.load(behaviour_matrix_2, allow_pickle=True)

    # Remove Any Traces We Dont Want
    trace_name_dict = create_trace_name_dict()

    trace_number_to_include = []
    for trace in traces_to_include:
        trace_number_to_include.append(trace_name_dict[trace])

    traces_to_remove = []
    for x in range(np.shape(behaviour_matrix_1)[0]):
        if x not in trace_number_to_include:
            traces_to_remove.append(x)

    behaviour_matrix_1 = np.delete(behaviour_matrix_1, traces_to_remove, axis=0)
    behaviour_matrix_2 = np.delete(behaviour_matrix_2, traces_to_remove, axis=0)

    # Normalise Traces
    trace_offset = 1.5
    behaviour_matrix_1, behaviour_matrix_2 = joinly_scale_matricies(behaviour_matrix_1, behaviour_matrix_2, trace_offset)

    # Get Window Dimensions
    number_of_timepoints = np.shape(behaviour_matrix_1[0])[1]
    number_of_traces     = np.shape(behaviour_matrix_1)[0]
    number_of_frames     = np.shape(activity_matrix_1)[0]
    frame_step = number_of_timepoints / number_of_frames
    window_size = 2

    print("activity matrix", np.shape(activity_matrix_1))
    print("behaviour matrix", np.shape(behaviour_matrix_1))

    # Get X Values
    x_values = list(range(number_of_timepoints))
    colourmap = cm.get_cmap("jet")


    activity_matrix_1 = np.nan_to_num(activity_matrix_1)
    activity_matrix_2 = np.nan_to_num(activity_matrix_2)

    image_1_max = np.percentile(activity_matrix_1, 99)
    image_2_max = np.percentile(activity_matrix_2, 99)
    image_max = np.max([image_1_max, image_2_max])

    image_1_min = np.percentile(activity_matrix_1, 1)
    image_2_min = np.percentile(activity_matrix_2, 1)
    image_min = np.min([image_1_min, image_2_min])

    print(np.shape(activity_matrix_1))
    print("time list", time_list, len(time_list))
    print("number of frames", number_of_frames)

    print("max", image_max)
    print("min", image_min)

    activity_matrix_1 = scale_matrix(activity_matrix_1, image_min, image_max)
    activity_matrix_2 = scale_matrix(activity_matrix_2, image_min, image_max)



    # Draw Activity
    for frame in range(number_of_frames):

        # Create Figure
        figure_1 = plt.figure(dpi=200, constrained_layout=True)
        grid_spec = gridspec.GridSpec(nrows=3, ncols=6, figure=figure_1)

        traces_1_axis = figure_1.add_subplot(grid_spec[0, 0:2])
        traces_2_axis = figure_1.add_subplot(grid_spec[0, 2:4])

        #image_1_axis = figure_1.add_subplot(1,1,1)
        image_1_axis = figure_1.add_subplot(grid_spec[1:3, 0:2])
        image_2_axis = figure_1.add_subplot(grid_spec[1:3, 2:4])
        image_3_axis = figure_1.add_subplot(grid_spec[1:3, 4:6])

        traces_1_axis.set_axis_off()
        traces_2_axis.set_axis_off()
        image_1_axis.set_axis_off()
        image_2_axis.set_axis_off()
        image_3_axis.set_axis_off()

        # Plot Behavioural Traces
        draw_temporal_window(number_of_timepoints, frame, frame_step, window_size, number_of_traces, trace_offset, traces_1_axis)
        draw_temporal_window(number_of_timepoints, frame, frame_step, window_size, number_of_traces, trace_offset, traces_2_axis)


        for trace_index in range(number_of_traces):
            trace = behaviour_matrix_1[trace_index]
            colour = get_colour(float(trace_index / number_of_traces), "tab10", 1)
            traces_1_axis.plot(x_values, trace[0], c=colour)
            traces_1_axis.fill_between(x=x_values, y1=trace[0], y2=trace[1], color=colour, alpha=0.2)
            traces_1_axis.fill_between(x=x_values, y1=trace[0], y2=trace[2], color=colour, alpha=0.2)

        for trace_index in range(number_of_traces):
            trace = behaviour_matrix_2[trace_index]
            colour = get_colour(float(trace_index / number_of_traces), "tab10", 1)
            traces_2_axis.plot(x_values, trace[0], c=colour)
            traces_2_axis.fill_between(x=x_values, y1=trace[0], y2=trace[1], color=colour, alpha=0.2)
            traces_2_axis.fill_between(x=x_values, y1=trace[0], y2=trace[2], color=colour, alpha=0.2)

        count = 0
        for trace in traces_to_include:
            traces_2_axis.text(number_of_timepoints + 200, count * trace_offset, trace, fontsize="small")
            count += 1

        # Plot Brain Activity
        image_1 = create_image_from_data(activity_matrix_1[frame], image_height, image_width, indicies)
        image_2 = create_image_from_data(activity_matrix_2[frame], image_height, image_width, indicies)
        image_3 = np.subtract(image_1, image_2)
        image_3 = ndimage.gaussian_filter(image_3, sigma=2)

        image_1 = ndimage.rotate(image_1, -1 * rotation, reshape=False)
        image_2 = ndimage.rotate(image_2, -1 * rotation, reshape=False)


        #Get RGBA Colours from Colourmap
        image_1 = colourmap(image_1)
        image_2 = colourmap(image_2)

        image_1_axis.imshow(image_1, vmin=image_min, vmax=image_max)
        image_2_axis.imshow(image_2, vmin=image_min, vmax=image_max)
        image_3_axis.imshow(image_3, vmin=- 2 * image_max, cmap='bwr', vmax= 2 * image_max)

        # Add Time Labels
        frame_time = int(time_list[frame] * frame_step)
        image_1_axis.text(420, 50, str(frame_time) + "ms", c='w')
        image_2_axis.text(420, 50, str(frame_time) + "ms", c='w')
        #print(frame_time)

        #Add Condition Labls
        image_1_axis.text(20, -50, stimulus_names[0], c='k')
        image_2_axis.text(20, -50, stimulus_names[1], c='k')
        image_3_axis.text(20, -50, "Difference",      c='k')

        plt.savefig(output_folder + "/" + str(frame).zfill(6) + ".png", box_inches='tight', pad_inches=0)
        plt.close()


    combine_images_into_video(output_folder, stimuli_evoked_responses_directory + "/" + stimulus_names[0] + "_v_" + stimulus_names[1] + ".avi")




def reconstruct_single_video(base_directory, stimulus_name, traces_to_include):

    stimuli_evoked_responses_directory = base_directory + "/Stimuli_Evoked_Responses"

    # Check Output Folder Exists
    data_folder = stimuli_evoked_responses_directory + "/" + stimulus_name
    output_folder = data_folder + "/Images"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Load Mask
    indicies, image_height, image_width = load_mask()

    # Load Activity Matrix
    activity_matrix = data_folder + "/" + stimulus_name + "_Activity_Matrix.npy"
    activity_matrix = np.load(activity_matrix, allow_pickle=True)

    # Load Behaviour Matrix
    behaviour_matrix = data_folder + "/" + stimulus_name + "_Behaviour_Matrix.npy"
    behaviour_matrix = np.load(behaviour_matrix, allow_pickle=True)

    # Remove Any Traces We Dont Want
    trace_number_to_include = []
    for trace in traces_to_include:
        trace_number_to_include.append(trace_name_dict[trace])

    traces_to_remove = []
    for x in range(np.shape(behaviour_matrix)[0]):
        if x not in trace_number_to_include:
            traces_to_remove.append(x)

    behaviour_matrix = np.delete(behaviour_matrix, traces_to_remove, axis=0)


    # Normalise Traces
    trace_offset = 1.5
    behaviour_matrix = normalise_behavioural_matrix(behaviour_matrix, trace_offset)

    # Get Window Dimensions
    number_of_timepoints = np.shape(behaviour_matrix[0])[1]
    number_of_frames = np.shape(activity_matrix)[0]
    number_of_traces = np.shape(behaviour_matrix)[0]
    frame_step = number_of_timepoints / number_of_frames
    window_size = 3

    #Get X Values
    x_values = list(range(number_of_timepoints))

    # Draw Activity
    for frame in range(number_of_frames):

        #Create Figure
        fig1 = plt.figure(dpi=200)
        ax1 = fig1.add_subplot(1, 2, 2)
        ax2 = fig1.add_subplot(1, 2, 1)
        ax1.set_axis_off()
        ax2.set_axis_off()

        #Plot Behavioural Traces
        draw_temporal_window(number_of_timepoints, frame, frame_step, window_size, number_of_traces, trace_offset, ax2)
        for trace_index in range(number_of_traces):
            trace = behaviour_matrix[trace_index]
            colour = get_colour(float(trace_index/number_of_traces), "tab10", 1)
            ax2.plot(x_values, trace[0], c=colour)
            ax2.fill_between(x=x_values, y1=trace[0], y2=trace[1], color=colour, alpha=0.2)
            ax2.fill_between(x=x_values, y1=trace[0], y2=trace[2], color=colour, alpha=0.2)

        #Plot Brain Activity
        image = create_image_from_data(activity_matrix[frame], image_height, image_width, indicies)
        ax1.imshow(image, vmin=0, cmap='jet', vmax=1.5)

        plt.savefig(output_folder + "/" + str(frame).zfill(6) + ".png", box_inches='tight', pad_inches=0)
        plt.close()

    combine_images_into_video(data_folder + "/Images", data_folder + "/" + stimulus_name + ".avi")


def create_trace_name_dict():

    trace_name_dict = {
    "running":0,
    "lick":1,
    "reward":2,
    "odour_1":3,
    "odour_2":4,
    "visual_1":5,
    "visual_2":6
    }



    return trace_name_dict



def create_comparison_video(home_directory, condition_names, trial_data, streams_to_include):

    reconstruct_comparison_video(home_directory, condition_names, streams_to_include, trial_data)

    #boundary_data = np.divide(boundary_data, np.max(boundary_data))
    #boundary_data = np.multiply(boundary_data, 2)

    #reconstruct_comparison_video(["first_odour_1_average", "stable_odour_1_average"], ["running", "lick", "reward", "odour_1", "odour_2", "visual_1", "visual_2"])
    #reconstruct_comparison_video(["first_odour_2_average", "stable_odour_2_average"], ["running", "lick", "reward", "odour_1", "odour_2", "visual_1", "visual_2"])


    #reconstruct_single_video("Visual_Block_Visual_1", ["running", "lick", "reward", "visual_1"])
    #reconstruct_single_video("Visual_Block_Visual_2", ["running", "lick",  "visual_2"])
    #reconstruct_single_video("Odour_Block_Visual_1",  ["running", "lick", "reward",  "odour_1", "odour_2", "visual_1"])
    #reconstruct_single_video("Odour_Block_Visual_2",  ["running", "lick", "odour_1", "odour_2", "visual_2"])


    #reconstruct_comparison_video(["vis_1_transition_average", "vis_1_stable_average"],["running", "lick", "reward", "odour_1", "odour_2", "visual_1", "visual_2"])

    #reconstruct_comparison_video(["vis_2_offset_visual_average", "vis_2_offset_odour_average"], ["running", "lick", "reward", "odour_1", "odour_2", "visual_1", "visual_2"])

    #reconstruct_comparison_video(["Odour_1_Cued_Average", "Odour_1_Uncued_Average"], ["running", "lick", "reward", "odour_1", "odour_2", "visual_1", "visual_2"])
    #reconstruct_comparison_video(["Odour_2_Cued_Average", "Odour_2_Uncued_Average"], ["running", "lick", "reward", "odour_1", "odour_2", "visual_1", "visual_2"])



    #Basic Stimuli
    #reconstruct_comparison_video(["Visual_Block_Visual_1", "Visual_Block_Visual_2"], ["running", "lick", "reward", "visual_1", "visual_2"])
    #reconstruct_comparison_video(["Odour_1",   "Odour_2"], ["running", "lick", "reward", "odour_1", "odour_2", "visual_1", "visual_2"])

    #Stimuli Affected By Context
    #reconstruct_comparison_video(["Visual_Block_Visual_1", "Odour_Block_Visual_1"], ["running", "lick", "reward", "odour_1", "odour_2", "visual_1"])
    #reconstruct_comparison_video(["Visual_Block_Visual_2", "Odour_Block_Visual_2"], ["running", "lick", "reward", "odour_1", "odour_2", "visual_2"])


"""
home_directory = r"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Analysis/71.2A_2020_11_17_Discrimination"
condition_names = ["All Vis 1", "All Vis 2"]

trial_start     = -10 #Should be negative
trial_end       = 100
window_size     = 2
use_baseline    = False

trial_details = [trial_start, trial_end, window_size, use_baseline]
streams_to_include = ["running", "lick",  "odour_1", "odour_2", "visual_1", "visual_2"]

create_comparison_video(home_directory, condition_names, trial_details, streams_to_include)
"""
