import sys
import numpy as np
import os

sys.path.append("/home/matthew/Documents/Github_Code/Trial_Aligned_Analysis/Widefield_Analysis")

import Widefield_General_Functions
import Template_Masking
import Heamocorrection
import Classify_Trials
import Extract_Trial_Aligned_Activity
import Extract_Trial_Aligned_Behaviour
import Create_Activity_Movie_Module


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
            print("Ai filename is: ", ai_filename)
            return ai_filename


def determine_trial_type(home_directory):
    trial_type = None
    home_directory_list = home_directory.split("/")
    print(home_directory_list)
    if "Discrimination" in home_directory_list[-1]:
        trial_type = "Discrimination"
    elif "Switching" in home_directory_list[-1]:
        trial_type = "Switching"

    return trial_type



def perform_comparison(trial_onset_filenames, condition_names, streams_to_include):
    print("Extracting Activity", condition_names[0], condition_names[1])
    Extract_Trial_Aligned_Activity_4.extract_trial_aligned_activity(base_directory, trial_onset_filenames, condition_names, trial_details)
    print("Extracting Behaviour", condition_names[0], condition_names[1])
    Extract_Trial_Aligned_Behaviour_2.extract_trial_aligned_behaviour(base_directory, ai_recorder_filename, condition_names)
    print("Creating Movie", condition_names[0], condition_names[1])
    Create_Activity_Movie_Module_2.create_comparison_video(base_directory, condition_names, trial_details, streams_to_include)


def extract_switching_stimuli_evoked_responses():
    print("Extracting Switching Trial Responses")

    streams_to_include = ["running", "lick", "odour_1", "odour_2", "visual_1", "visual_2"]

    # Correct Vis 1 v Correct Vis 2
    trial_onset_filenames = ["visual_context_correct_vis_1_frames.npy", "visual_context_correct_vis_2_frames.npy"]
    condition_names = ["Visual Context Correct Vis 1", "Visual Context Correct Vis 2"]
    perform_comparison(trial_onset_filenames, condition_names, streams_to_include)
    #Region_Quantification_V3.quantify_region_responses(base_directory, condition_names)

    """
    # Correct Visual Context Vis 1 v Incorrect Visual Context Vis 1
    trial_onset_filenames = ["visual_context_correct_vis_1_frames.npy", "visual_context_incorrect_vis_1_frames.npy"]
    condition_names = ["Visual Context Correct Vis 1", "Visual Context Incorrect Vis 1"]
    perform_comparison(trial_onset_filenames, condition_names, streams_to_include)
    #Region_Quantification_V3.quantify_region_responses(base_directory, condition_names)

    # Correct Visual Context Vis 2 v Incorrect Visual Context Vis 2
    trial_onset_filenames = ["visual_context_correct_vis_2_frames.npy", "visual_context_incorrect_vis_2_frames.npy"]
    condition_names = ["Visual Context Correct Vis 2", "Visual Context Incorrect Vis 2"]
    perform_comparison(trial_onset_filenames, condition_names, streams_to_include)
    #Region_Quantification_V3.quantify_region_responses(base_directory, condition_names)
    """

    # Correct Odour 1 v Correct Odour 2
    trial_onset_filenames = ["correct_odour_1_frames.npy", "correct_odour_2_frames.npy"]
    condition_names = ["Correct Odour 1", "Correct Odour 2"]
    perform_comparison(trial_onset_filenames, condition_names, streams_to_include)
    #Region_Quantification_V3.quantify_region_responses(base_directory, condition_names)

    # Correct Visual Context V1 v Correct Odour Context V1
    trial_onset_filenames = ["visual_context_correct_vis_1_frames.npy", "odour_context_correct_vis_1_frames.npy"]
    condition_names = ["Visual Context Correct Vis 1", "Odour Context Correct Vis 1"]
    perform_comparison(trial_onset_filenames, condition_names, streams_to_include)
    #Region_Quantification_V3.quantify_region_responses(base_directory, condition_names)

    #Correct visual Context V1 v Correct Odour Context V1
    trial_onset_filenames = ["visual_context_correct_vis_2_frames.npy", "odour_context_correct_vis_2_frames.npy"]
    condition_names = ["Visual Context Correct Vis 2", "Odour Context Correct Vis 2"]
    perform_comparison(trial_onset_filenames, condition_names, streams_to_include)
    #Region_Quantification_V3.quantify_region_responses(base_directory, condition_names)





def extract_discrimination_stimuli_evoked_responses(base_directory, ai_recorder_filename):
    print("Extracting Discrimination Trial Responses")

    streams_to_include = ["running", "lick", "odour_1", "odour_2", "visual_1", "visual_2"]

    # Create All Onsets Movie
    trial_onset_filenames = ["All_vis_1_frame_indexes.npy", "All_vis_2_frame_indexes.npy"]
    condition_names = ["All Vis 1", "All Vis 2"]
    Extract_Trial_Aligned_Activity_4.extract_trial_aligned_activity(base_directory, trial_onset_filenames, condition_names, trial_details)
    Extract_Trial_Aligned_Behaviour_2.extract_trial_aligned_behaviour(base_directory, ai_recorder_filename, condition_names)
    Create_Activity_Movie_Module_2.create_comparison_video(base_directory, condition_names, trial_details, streams_to_include)
    #Region_Quantification_V3.quantify_region_responses(base_directory, condition_names)

    # Create V1 Correct v Incorrect
    correct_trial_onsets   = np.load(base_directory + "/Stimuli_Onsets" + "/Correct_vis_1_frame_indexes.npy")
    incorrect_trial_onsets = np.load(base_directory + "/Stimuli_Onsets" + "/Incorrect_vis_1_frame_indexes.npy")

    if len(correct_trial_onsets) > 0 and len(incorrect_trial_onsets) > 0:
        if correct_trial_onsets.all() != None and incorrect_trial_onsets.all()!= None:
            print("Making V1 Comparison")
            trial_onset_filenames = ["Correct_vis_1_frame_indexes.npy", "Incorrect_vis_1_frame_indexes.npy"]
            condition_names = ["Correct Vis 1", "Incorrect Vis 1"]
            Extract_Trial_Aligned_Activity_4.extract_trial_aligned_activity(base_directory, trial_onset_filenames, condition_names, trial_details)
            Extract_Trial_Aligned_Behaviour_2.extract_trial_aligned_behaviour(base_directory, ai_recorder_filename, condition_names)
            Create_Activity_Movie_Module_2.create_comparison_video(base_directory, condition_names, trial_details, streams_to_include)
            #Region_Quantification_V3.quantify_region_responses(base_directory, condition_names)

    # Create V2 Correct v Incorrect
    correct_trial_onsets = np.load(base_directory + "/Stimuli_Onsets" + "/Correct_vis_2_frame_indexes.npy")
    incorrect_trial_onsets = np.load(base_directory + "/Stimuli_Onsets" + "/Incorrect_vis_2_frame_indexes.npy")

    if len(correct_trial_onsets) > 0 and len(incorrect_trial_onsets) > 0:
        if correct_trial_onsets.all() != None and incorrect_trial_onsets.all() != None:
            print("Making V2 Comparison")
            trial_onset_filenames = ["Correct_vis_2_frame_indexes.npy", "Incorrect_vis_2_frame_indexes.npy"]
            condition_names = ["Correct Vis 2", "Incorrect Vis 2"]
            Extract_Trial_Aligned_Activity_4.extract_trial_aligned_activity(base_directory, trial_onset_filenames, condition_names, trial_details)
            Extract_Trial_Aligned_Behaviour_2.extract_trial_aligned_behaviour(base_directory, ai_recorder_filename, condition_names)
            Create_Activity_Movie_Module_2.create_comparison_video(base_directory, condition_names, trial_details, streams_to_include)
            #Region_Quantification_V3.quantify_region_responses(base_directory, condition_names)




sessions = [#"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Switching_Analysis/Selected_sessions/NRXN78.1A/2020_12_09_Switching",
            #"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Switching_Analysis/Selected_sessions/NRXN78.1D/2020_11_29_Switching",
            #"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Switching_Analysis/Selected_sessions/NRXN71.2A/2020_12_17_Switching_Imaging",
            #"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Switching_Analysis/Selected_sessions/NXAK4.1A/2021_04_06_Switching_Imaging",
            #"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Switching_Analysis/Selected_sessions/NXAK4.1B/2021_03_04_Switching_Imaging",
            #"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Switching_Analysis/Selected_sessions/NXAK7.1B/2021_03_02_Switching_Imaging",
            #"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Switching_Analysis/Selected_sessions/NXAK7.1E/2021_03_03_Switching_Imaging",
            #"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Switching_Analysis/Selected_sessions/NXAK10.1A/2021_05_20_Switching_Imaging",
            #"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Switching_Analysis/Selected_sessions/NXAK14.1A/2021_06_09_Switching_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Switching_Analysis/Selected_sessions/NXAK16.1B/2021_06_23_Switching_Imaging"
    ]



for base_directory in sessions:
    ai_recorder_filename = get_ai_filename(base_directory)
    print(ai_recorder_filename)

    trial_start     = -10 #Should be negative
    trial_end       = 100
    window_size     = 2
    use_baseline    = True
    trial_details = [trial_start, trial_end, window_size, use_baseline]

    # Step 0 Perform Template Masking
    print("Masking Data")
    #Template_MaskingV2.perform_template_masking(base_directory)

    # Step 1 Perform Heamocorrection
    print("Performing Heamocorrection")
    Heamocorrection_3.perform_heamocorrection(base_directory)

    # Step 2 Register To Allen Atlas
    print("Registering To Allen Atlas")
    #Extract_NMF_Components_V3.extract_nmf_components(base_directory)
    #Allen_Common_Coordiate_Registration_V3.register_to_allen_atlas(base_directory)

    # Step 3 Classify Trials
    print("Classify Trials")
    #Classify_Trials_4.classify_trials(base_directory, ai_recorder_filename)

    """
    trial_type = determine_trial_type(base_directory)
    
    if trial_type == "Discrimination":
        extract_discrimination_stimuli_evoked_responses(base_directory, ai_recorder_filename)
    
    elif trial_type == "Switching":
        extract_switching_stimuli_evoked_responses()
    """