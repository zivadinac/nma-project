import numpy as np
import matplotlib.pyplot as plt

dat = np.load('data/stringer_spontaneous.npy', allow_pickle=True).item()


movement_thr = dat["run"].mean()
movement_start = np.diff((dat["run"] > movement_thr).astype(int), axis=0) > 0
print(movement_start.sum())
plt.eventplot(movement_start)
plt.plot(dat["run"])
plt.show()
