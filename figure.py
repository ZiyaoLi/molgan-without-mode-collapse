import json
import numpy as np
import matplotlib.pyplot as plt

DIRECTORIES = ['molgan+qed']
recs = []
interests = ["loss D", "loss G", "QED score", "valid score", "unique score", "novel score"]

fig, ax = plt.subplots(2, 3)
indices = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]


for _d in DIRECTORIES:
    for _ in range(5):
        with open(_d + '/%d.json' % (_ + 1)) as fh:
            _rec = json.load(fh)
        recs.append(_rec)

    xs = np.arange(len(recs[0]))

    for i, k in enumerate(interests):
        k_hist = [
            [step[k] for step in rec] for rec in recs
        ]
        k_array = np.stack(k_hist)
        k_array[np.isnan(k_array)] = 0
        k_mean = np.mean(k_array, 0)
        k_intv = np.std(k_array, 0) * 1.96

        k_ax = ax[indices[i][0]][indices[i][1]]
        k_ax.plot(xs, k_mean, color='r')
        k_ax.fill_between(xs, k_mean - k_intv, k_mean + k_intv, color='r', alpha=0.5)
        k_ax.set_title(k)

fig.savefig('test.png', format='png')





