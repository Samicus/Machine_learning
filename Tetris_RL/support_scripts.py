import numpy as np
import matplotlib.pyplot as plt

def plot_rewards(reward_tots):
    plt.figure()
    plt.plot(reward_tots)
    plt.plot(range(len(reward_tots)), smooth(reward_tots, 20))
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.show()

def binatodeci(binary):
    deci = sum(val*(2**idx) for idx, val in enumerate(reversed(binary)))
    return int(deci)

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth