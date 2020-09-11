import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib

F = [200]
N = [6]

K = np.size(N)

for k in range(K):
    fs = F[k]
    Fs = int(np.floor(fs))

    n = N[k]

    print('Heatmapeanding n={} a {}Hz'.format(n, Fs))

    sims = 1000
    n_e = 20
    rotores = 6
    labelnumber = 6
    time_all = 20
    datapoints = int(time_all * fs + 1)
    timestamps = np.linspace(0, time_all, datapoints)
    path = 'Results/Testing/RF_n{}_f{}'.format(n, Fs)
    outputs = np.empty((rotores * sims * datapoints, labelnumber))

    for r in range(rotores):
        ddir = pathlib.Path(path + "/fail{}".format(r + 1))
        for s in range(sims):
            fn = ddir / "run{}".format(s + 1)
            assert (fn.exists())
            outputs[(s + r * sims) * datapoints:(s + 1 + r * sims) * datapoints, :] = pd.read_csv(fn).values[:,
                                                                                      -labelnumber:]

    for l in range(rotores):
        salidas = pd.DataFrame(outputs[:, l].reshape(datapoints, rotores * sims, order='F'), index=timestamps)
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(salidas.T, xticklabels=250, yticklabels=int(rotores * sims / 6), ax=ax, cmap="viridis", vmin=0,
                    vmax=1)
        plt.title('Fail {}'.format(l + 1), fontsize=25, color=np.array([0.2, 0.2, 0.7]))
        plt.xlabel('Time [S]', fontsize=20)
        plt.ylabel('Number of simulation', fontsize=20)
        plt.savefig('Results/Figures/RF_n{}_f{}/Fail{}_heatmap_n{}_fs{}'.format(n, Fs, l + 1, n, Fs))  # ,format='tiff')