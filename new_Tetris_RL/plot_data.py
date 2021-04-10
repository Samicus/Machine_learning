import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


data = loadtxt('data.csv', delimiter=',')

plt.figure()
plt.plot(data)
plt.plot(range(len(data)), smooth(data, 40))
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.show()


