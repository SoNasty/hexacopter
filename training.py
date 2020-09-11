rpy = ['roll_deg', 'pitch_deg', 'yaw_deg']
rpy_ref = ['rollref_deg', 'pitchref_deg', 'yawref_deg']
rpy_speed = ['rollspeed_degperrsec', 'pitchspeed_degperrsec', 'yawspeed_degperrsec']
fmot = ["f{}_kg".format(i + 1) for i in range(6)]
zlmn = ['Z', 'L', 'M', 'N']

class Format:
    end = '\033[0m'
    underline = '\033[4m'
import pandas as pd
import numpy as np
import pathlib
import joblib
import sklearn.pipeline
import sklearn.ensemble
import sklearn.preprocessing
import time
# =============================================================================

version = "7.2"

sims_train = 500  # cantidad de simulaciones por cada rotor
n = 6
fs = 100
orden = 1

# Parametros del svm:
c = 1e-3
tole = 1e-3  # default
# ratio = 1e-4
# pesos_clases = {0:ratio,1:1, 2:1,3:1,4:1,5:1,6:1}
pesos_clases = 'balanced'
iter = -1
state = 42  # / None # Quiero comparar repitiendo resultados
cachito = 2000
diezmo = 6  # decimacion de lo datos

# clf = sklearn.svm.LinearSVC(random_state = state, dual = False, max_iter = iter, tol = tole, class_weight=pesos_clases,C=c)
clf = sklearn.svm.SVC(C=c, probability=True, kernel='rbf', cache_size=cachito, degree=orden, tol=tole,
                      class_weight=pesos_clases, max_iter=iter, random_state=state)

FEATURES = [rpy + fmot]
names = ['rpy+fmot']

Fs = int(np.floor(fs))
time_fail = 0.5
time_all = 10
time_cut = 0.15
rotores = 6
labelnumber = 1

datapoints = int(fs * time_all + 1)
failpoints = int(fs * time_fail)
cutpoints = int(fs * time_cut)
factor = round(200 / fs)

for k, features in enumerate(FEATURES):

    time_init = time.time()

    len_features = len(features) * orden
    # training_data = np.empty((int(np.round((datapoints-cutpoints-n+1)*sims_train*rotores/diezmo)),n*len_features+1))
    training_data = np.empty(
        (int(np.round((datapoints - cutpoints - n + 2) / diezmo)) * sims_train * rotores, n * len_features + 1))
    dfs = []

    for r in range(rotores):

        ddir = pathlib.Path("../Data/Train/fail{}".format(r + 1))  # me paro en la carpeta

        for s in range(sims_train):

            fn = ddir / "run{}.csv".format(s + 1)  # y busco el csv
            assert (fn.exists())
            dfs = pd.read_csv(fn)

            matrix = dfs[features].values[::factor, :]
            matrix0 = dfs[features].values[::factor, :]

            for _ in range(orden - 1):  # expando potencias de los features
                matrix0 = matrix0 * matrix[:, :len(features)]
                matrix = np.append(matrix, matrix0, axis=1)

            matrix = np.append(matrix, np.zeros((n - 1, len_features)), axis=0)  # n-1 filas al final
            matrix = np.append(matrix, np.zeros((matrix.shape[0], (n - 1) * len_features + labelnumber)),
                               axis=1)  # n-1 columnas de features + flags

            for j in range(n - 1):  # armo la ventana
                matrix[j + 1:j + 1 + datapoints, len_features * (j + 1):len_features * (j + 2)] = matrix[:datapoints,
                                                                                                  :len_features]

            matrix = matrix[n - 1:datapoints, :]  # (n-1) primeras filas
            matrix[-failpoints:, -labelnumber] = int(r + 1)  # Flag
            matrix = np.append(matrix[:-failpoints, :], matrix[(-failpoints + cutpoints):, :], axis=0)  # cutpoints.

            matrix = matrix[::diezmo, :]
            training_data[(s + r * sims_train) * matrix.shape[0]:(s + 1 + r * sims_train) * matrix.shape[0],
            :] = matrix  # meto el vuelo en el set de datos

    # In[3]: entrenamiento de un clasificador Forrest Tree.

    lab_enc = sklearn.preprocessing.LabelEncoder()
    training_outputs = lab_enc.fit_transform(
        training_data[:, -1])  # esto transforma los enteros en labels (fail0, fail1,..., fail6)

    clf = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), clf)  # Normaliza los features

    print('------------------------------------------------------------------\n')
    print(Format.underline + 'Entrenamiento v{}: {}'.format(version, time.ctime(time_init)) + Format.end)
    clf.fit(training_data[:, :-labelnumber], training_outputs)  # Entrenamiento

    # In[4]: guardo el clasificador
    joblib.dump(clf, 'Clasificadores/SVM({})_n{}_f{}_tr{}_{}'.format(names[k], n, Fs, sims_train, version))

    # In[5]: calculo el tiempo del entrenador:
    intervalo = time.time() - time_init  # TOC
    horas = int(np.floor(intervalo / 3600))
    minutos = int(np.floor((intervalo - horas * 3600) / 60))
    segundos = round(intervalo - horas * 3600 - minutos * 60)
    print('Tiempo de entrenamiento para n{}_fs{}: {}:{}:{} (hh:mm:ss)'.format(n, Fs, horas, minutos, segundos))
    print('------------------------------------------------------------------')
# print('Fin del entrenamiento: {}'.format(time.ctime(time.time())))
# print('\a')
