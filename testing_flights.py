VUELOS = ['fail_indoors1', 'fail_indoors2', 'fail_indoors3', 'fail_indoors4', 'fail_outdoors', 'mov_indoors',
          'mov_outdoors', 'vuelo de simulacion']  # , 'movement1','movement2','fail1','fail2']
fmot = ["f{}_kg".format(i + 1) for i in range(6)]
rpy = ['roll_deg', 'pitch_deg', 'yaw_deg']
rpy_ref = ['rollref_deg', 'pitchref_deg', 'yawref_deg']
rpy_speed = ['rollspeed_degperrsec', 'pitchspeed_degperrsec', 'yawspeed_degperrsec']
zlmn = ['Z', 'L', 'M', 'N']
CLASSIFIERS = ['rpy+fmot']
FEATURES = [rpy + fmot]
import pandas as pd
import numpy as np
import pathlib
import joblib
import matplotlib.pyplot as plt

version1 = 7
sub_version = 1

version = "{}.{}".format(version1, sub_version)
version2 = "{}_{}".format(version1, sub_version)  # solo para el archivo

fs = 100
Fs = int(np.floor(fs))
n = 3
orden = 1
sims = 500
labelnumber = 6
factor = round(200 / fs)
factor2 = 1

for k, features in enumerate(FEATURES):

    clf = joblib.load('Clasificadores/SVM({})_n{}_f{}_tr{}_{}'.format(CLASSIFIERS[k], n, Fs, sims, version))

    for p, name in enumerate(VUELOS):

        ddir = pathlib.Path("../Data/Vuelos/test{}.csv".format(p + 1))
        assert (ddir.exists())
        dfs = []
        dfs.append(pd.read_csv(ddir).set_index("time_sec"))

        matrix = dfs[0][features].values[::factor, :]
        matrix0 = dfs[0][features].values[::factor, :]

        for _ in range(orden - 1):  # expando potencias de los features
            matrix0 = matrix0 * matrix[:, :len(features)]
            matrix = np.append(matrix, matrix0, axis=1)

        matrix2 = dfs[0][rpy + rpy_ref].values[::factor, :]  # la que uso para graficar los angulos
        matrix3 = dfs[0][fmot].values[::factor, :]  # la que uso para graficar las fmot
        timestamps = dfs[0].index[::factor]

        # Armo la ventana
        datapoints = matrix.shape[0]
        len_features = matrix.shape[1]

        matrix = np.append(matrix, np.zeros((n - 1, len_features)), axis=0)
        matrix = np.append(matrix, np.zeros((matrix.shape[0], (n - 1) * len(features))), axis=1)

        for j in range(n - 1):
            matrix[j + 1:datapoints + j + 1, len_features * (j + 1):len_features * (j + 2)] = matrix[:datapoints,
                                                                                              :len_features]

        matrix = matrix[:datapoints, :]  # recorto las primeras y ultimas n filas incompletas

        # Clasificacion
        predict = clf.predict_proba(matrix)
        outputs_fail = pd.DataFrame(index=timestamps)

        for j in range(labelnumber):
            outputs_fail[j] = pd.DataFrame(data=predict[:, j + 1], index=timestamps)

        # filtrado de picos de yaw de matrix2
        for j in range(matrix2.shape[0]):
            if np.abs(matrix2[j, 2] - matrix2[j - 1, 2]) > 10:
                matrix2[j, 2] = matrix2[j - 1, 2]

        # filtrado de picos de yaw ref
        for j in range(matrix2.shape[0]):
            if np.abs(matrix2[j, 5] - matrix2[j - 1, 5]) > 10:
                matrix2[j, 5] = matrix2[j - 1, 5]

        # Busco rearmar para que grafique piola los colores
        roll = np.array(([matrix[:, 0], matrix2[:, 3]]))
        pitch = np.array(([matrix[:, 1], matrix2[:, 4]]))
        yaw = np.array(([matrix[:, 2], matrix2[:, 5]]))

        # GRAFICOS:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        fig = plt.figure(figsize=(20, 15))

        ax1 = plt.subplot(311)
        plt.title('{} ({}) V{}'.format(name, CLASSIFIERS[k], version), fontsize=30)

        # Angulos y referencias:
        ax1.plot(timestamps.values[::factor2], roll[0, ::factor2], color=colors[0])
        ax1.plot(timestamps.values[::factor2], roll[1, ::factor2], color=colors[0], linestyle='dashed')
        ax1.plot(timestamps.values[::factor2], pitch[0, ::factor2], color=colors[1])
        ax1.plot(timestamps.values[::factor2], pitch[1, ::factor2], color=colors[1], linestyle='dashed')
        ax1.plot(timestamps.values[::factor2], yaw[0, ::factor2], color=colors[2])
        ax1.plot(timestamps.values[::factor2], yaw[1, ::factor2], color=colors[2], linestyle='dashed')
        ax1.legend(['R', 'R ref', 'P', 'P ref', 'Y', 'Y ref'], loc="lower left", fontsize='20')
        ax1.set_ylim([-24.99, 24.99])
        ax1.set_ylabel('Angles [°]', fontsize=20)
        ax1.grid()
        ax1.tick_params(labelsize=20)

        # Motores:
        ax2 = plt.subplot(312)
        ax2.plot(timestamps.values, matrix3)
        ax2.legend(['f1', 'f2', 'f3', 'f4', 'f5', 'f6'], loc="lower left", fontsize='20')
        # ax2.set_xlim([27,33])
        ax2.set_ylim([0, 0.8])
        ax2.set_ylabel('Forces [kg]', fontsize=20)
        ax2.tick_params(labelsize=20)
        ax2.grid()

        # Flags:
        ax3 = plt.subplot(313)
        ax3.plot(timestamps.values[::factor2], outputs_fail.values[::factor2, ])
        # ax3.set_xlim([0,6.7])
        ax3.set_ylim([-0.01, 1.01])
        ax3.set_ylabel('Flags', fontsize=20)
        ax3.legend(['F1', 'F2', 'F3', 'F4', 'F5', 'F6'], loc="upper left", fontsize='20')
        ax3.set_xlabel('Time [S]', fontsize=20)
        ax3.tick_params(labelsize=20)
        ax3.grid()

        # Save Fig:
        version3 = int(version1)
        plt.savefig('Results/Vuelos{}/{}(v{})'.format(version3, name, version2, CLASSIFIERS[k]),
                    bbox_inches='tight')  # ,format='eps')

print('\n\n---------------------------------------------------------------------------------------------------\n')
# print('V{}'.format(version))
# print(clf[1])