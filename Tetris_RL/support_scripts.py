import numpy as np
import matplotlib.pyplot as plt

def plot_rewards(reward_tots):
    plt.figure()
    plt.plot(reward_tots)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.show()

def binatodeci(binary):
    deci = sum(val*(2**idx) for idx, val in enumerate(reversed(binary)))
    return int(deci)