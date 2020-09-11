import pandas as pd
import numpy as np
import pathlib
import joblib

F = [50, 200 / 3, 100, 100, 100, 100, 100, 200]
N = [6, 6, 3, 6, 9, 12, 15, 6]

K = np.size(N)

for k in range(K):

    n = N[k]
    fs = F[k]
    Fs = int(np.floor(fs))  # para fs=66.66
    n_e = 20

    sims = 1000  # puedo guardar menos si quiero
    time_all = 20

    datapoints = int(fs * time_all + 1)
    factor = round(200 / fs)

    rotores = 6
    labelnumber = 6

    fmot = ["f{}_kg".format(i + 1) for i in range(6)]
    rpy = ['roll_deg', 'pitch_deg', 'yaw_deg']
    rpy_ref = ['rollref_deg', 'pitchref_deg', 'yawref_deg']
    features = rpy + rpy_ref + fmot
    len_features = len(features)

    # In[2]: valores para cargar el clasificador por su nombre, no son parametros en este script:

    sims_train = 1500
    Fs = int(np.floor(fs))  # para fs=66.66

    # Cargo el clasificador:

    clf = joblib.load('Clasificadores/RF_n{}_f{}'.format(n, Fs))

    # In[3]: clasificacion de las muestras:

    # path_in  = 'Data/Train/' # si quiero chequear overfitting
    path_in = '../Data/Test'  # de donde tomo las muestras de prueba
    path_out = 'Results/Testing/RF_n{}_f{}'.format(n, Fs)  # donde vuelco los resultados de clasificacion

    dfs = []
    outputs = np.zeros((datapoints, labelnumber))  # reservo memoria
    padding = np.zeros((n - 1, labelnumber))  # plantilla para el padding de flags iniciales

    print('Clasificando para n={} a {}Hz'.format(n, Fs))

    for r in range(rotores):

        # seteo las rutas de lectura de vuelos y escritura de resultados:
        ddir_in = pathlib.Path(path_in + "/fail{}".format(r + 1))
        ddir_out = pathlib.Path(path_out + "/fail{}".format(r + 1))

        print('Clasificando vuelos de motor{}'.format(r + 1))

        for s in range(sims):

            fn_out = ddir_out / "run{}".format(s + 1)
            fn_in = ddir_in / "run{}.csv".format(s + 1)
            assert (fn_in.exists())
            dfs = pd.read_csv(fn_in)

            matrix = dfs[features].values[::factor, :]
            matrix = np.append(matrix, np.zeros((n - 1, len_features)), axis=0)
            matrix = np.append(matrix, np.zeros((datapoints + n - 1, (n - 1) * len_features)), axis=1)

            for j in range(n - 1):
                matrix[j + 1:j + 1 + datapoints, len_features * (j + 1):len_features * (j + 2)] = matrix[:datapoints,
                                                                                                  :len_features]

            matrix = matrix[:datapoints, :]  # descarto las ultimas (n-1) filas

            predict = clf.predict_proba(matrix)  # hago la prediccion del vuelo

            for j in range(labelnumber):
                outputs[:, j] = predict[j][:, 1]  # Guardo las salidas:

            outputs[:n - 1, :] = padding  # Padding en los primeros flags

            pd.DataFrame(outputs).to_csv(fn_out, index=True)  # Guardo el archivo