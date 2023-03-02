import numpy as np
import os
import matplotlib.pyplot as plt


base_directory = r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_09_15_Spontaneous"

# Load SVT
svt = np.load(os.path.join(base_directory, "Churchland_Preprocessing_Experimental", "Corrected_SVT.npy"))
u = np.load(os.path.join(base_directory, "Churchland_Preprocessing_Experimental", "U.npy"))
print(np.shape(svt))
print(np.shape(u))

sample = np.dot(u, svt[:, 0:100])

print(np.shape(sample))

plt.ion()

for x in range(100):

    plt.imshow(sample[:, :, x])
    plt.draw()
    plt.pause(0.1)
    plt.clf()