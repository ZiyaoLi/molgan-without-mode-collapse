import json
import numpy as np
import matplotlib.pyplot as plt

NAME = 'buggy095'
META_DIR = 'results/'
DIRECTORIES = ['molgan_095_buggy',
               'molganb_095_buggy',
               'pacstats_clip_elastic_095_buggy']
DIRECTORIES = [META_DIR + d for d in DIRECTORIES]
COLORS = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

INTERESTS = ["loss D", "loss G", "loss V", "loss RL", "QED score",
             "diversity score", "drugcandidate score", "valid score", "unique score", "novel score"]

PLT_LAYOUT = (2, 5)
INDEX = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]

fig, ax = plt.subplots(*PLT_LAYOUT, figsize=(18, 12))


for _d, _c in zip(DIRECTORIES, COLORS[:len(DIRECTORIES)]):
    recs = []
    for _ in range(5):
        import os
        if not os.path.exists(_d + '/%d.json' % (_ + 1)):
            continue
        with open(_d + '/%d.json' % (_ + 1)) as fh:
            _rec = json.load(fh)
        __rec = []
        for r in _rec:
            if r['loss D'] is np.nan or r['loss D'] == 0.0:
                for k in r:
                    r[k] = float(np.nan)
            __rec.append(r)
        recs.append(_rec)

    xs = np.arange(len(recs[0]))

    for i, k in enumerate(INTERESTS):
        k_hist = [
            [step[k] for step in rec] for rec in recs
        ]
        k_array = np.stack(k_hist)
        #k_array[np.isnan(k_array)] = 0
        k_mean = np.nanmean(k_array, 0)
        k_intv = np.nanstd(k_array, 0)

        k_ax = ax[INDEX[i][0]][INDEX[i][1]]
        k_ax.plot(xs, k_mean, color=_c)
        k_ax.fill_between(xs, k_mean - k_intv, k_mean + k_intv, color=_c, alpha=0.2)
        k_ax.set_title(k)

fig.savefig(META_DIR + NAME + '.png', format='png')





