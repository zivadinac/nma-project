import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

dat = np.load('data/stringer_spontaneous.npy', allow_pickle=True).item()

filtered_run = gaussian_filter(dat["run"], 5)

def detect_movement_onset(run_speed, stop_len=5, run_len=5, filter_sigma=5, speed_thr=2):
    speed = gaussian_filter(run_speed, filter_sigma) if filter_sigma > 0 else run_speed
    speed = speed[:,0]
    movement = speed > speed_thr
    movement_onset = np.zeros_like(movement)

    for i, e in enumerate(movement):
        if e and np.all(0 == movement[i-stop_len:i]) and np.all(movement[i:i+run_len]):
            movement_onset[i] = 1

    return movement_onset, speed


"""
movement_onset, speed = detect_movement_onset(dat["run"], run_len=5, stop_len=5)
plt.plot(speed)
plt.plot(movement_onset)
plt.show()
print("Number of movement onsets: ", movement_onset.sum())
"""
