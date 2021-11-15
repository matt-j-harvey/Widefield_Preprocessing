import sys
import numpy as np
import os
import Widefield_General_Functions
import Heamocorrection
import Get_Max_Projection

session_list = ["/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NRXN78.1A/2020_12_09_Switching_Imaging"]


for base_directory in session_list:

    print("Getting Max Projection")
    Get_Max_Projection.check_max_projection(base_directory)

    print("Performing Heamocorrection")
    Heamocorrection.perform_heamocorrection(base_directory)

