import numpy as np
import matplotlib.pyplot as plt
import Get_Max_Projection

base_directory = r"/media/matthew/Expansion/Widefield_Analysis/NXAK22.1A/2021_10_29_Transition_Imaging"

Get_Max_Projection.check_max_projection(base_directory, base_directory)
max_projection = np.load(base_directory + "/max_projection.npy")
plt.imshow(max_projection)
plt.show()