import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

dat = np.load('data/stringer_spontaneous.npy', allow_pickle=True).item()


# run 
print(np.median(dat["run"]))
print(np.percentile(dat["run"], 75))
print(np.mean(dat["run"]))

def get_movement_lengths(movement):
    lengths = []
    i = 0

    while i < len(movement):
        l = 0
        while movement[i]:
            l += 1
            i += 1
        if l > 0:
            lengths.append(l)
        i += 1

    return lengths
        
# gaussian filtering, 5 is ok
filtered_run = gaussian_filter(dat["run"], 5)
#filtered_run = dat["run"]
plt.plot(dat["run"])
plt.plot(filtered_run)
plt.show()

plt.plot(np.gradient(dat["run"][:,0]))
plt.plot(np.gradient(filtered_run[:,0]))
plt.show()


movement_thr = dat["run"].mean()
movement_thr = np.median(dat["run"])
#movement_thr = np.percentile(dat["run"], 90)
detected_movement = (filtered_run > movement_thr).astype(int)
movement_lengths = get_movement_lengths(detected_movement)
#movement_start = np.diff(detected_movement, axis=0) > 0
movement_start = np.diff(np.gradient(filtered_run[:,0]) > 0.1, axis=0)
print(movement_start.sum())

#plt.plot(dat["run"])
plt.plot(movement_start)
plt.plot(filtered_run)
plt.plot(dat["run"])
plt.show()

plt.hist(movement_lengths, 30)
plt.show()


#pupil_area_thr = 
filtered_pupil_area = gaussian_filter(dat["pupilArea"], 15)[:,0]
plt.plot(filtered_pupil_area)
#plt.show()

pupil_area_grad = np.gradient(filtered_pupil_area)
pupil_area_change = pupil_area_grad > 5
print(pupil_area_change.sum())
plt.plot(filtered_pupil_area)
plt.plot(pupil_area_change)
#plt.plot(pupil_area_grad)
plt.show()
