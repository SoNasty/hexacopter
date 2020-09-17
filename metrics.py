import pandas as pd
import numpy as np
import pathlib
from tqdm import tqdm
from scipy.integrate import trapz

# In[1]: Declaro parametros a usar.
F=[50,200/3,100,100,100,100,100,200]
N=[ 6,  6,    3,  6,  9,  12, 15, 6]

K = np.size(N)

for k in range(K):
    
    n = N[k]
    fs = F[k]
    n_e = 20
    
    sims = 1000
    time_fail = 0.5
    time_all = 20
    max_delay = 0.4 # 400ms es el maximo tiempo de deteccion
    
    rotores = 6
    labelnumber = 6
    factor = round(200/fs)
    
    datapoints = int(fs*time_all+1)
    failpoints = int(fs*time_fail)
    maxpoints = int(max_delay*fs)
        
    Fs = int(np.floor(fs))
    path = 'Results/Testing/RF_n{}_f{}'.format(n,Fs)
    
    steps = n_e
    P = np.zeros((steps))
    R = np.zeros((steps))
    TP = np.zeros((steps))
    FP = np.zeros(steps)
    FN = np.zeros(steps)
    
    threshold = np.zeros((steps))
    falses = np.zeros((steps))
    delay = np.zeros((steps))
    sigma_delay = np.zeros((steps))
    dist_threshold = np.zeros((steps)) #quiero medir el treshold mas cercano al (1,1) del PR.
    
    maximo = np.zeros((steps))
    minimo = np.zeros((steps))
    
    print('Calculando metricas para n={} a {}Hz'.format(n,Fs))
    for i in tqdm(range(steps)):
    
        d = []
        threshold[i] = i/steps
    
        for r in range (rotores):
    
            ddir = pathlib.Path(path+"/fail{}".format(r+1))
    
            for s in range(sims):
    
                fn = ddir/"run{}".format(s+1)
                assert(fn.exists())
    
                outputs = pd.read_csv(fn).values[:,-labelnumber:]>threshold[i] # necesito tirar el indice del csv
                
                fail_time = np.argmax(outputs[-failpoints:,r]) # busco si levanta el flag correcto y cuando lo hace
                outputs[-failpoints:,r] = 0 #plancho el flag

                if fail_time == 0 or fail_time >= maxpoints: # si no detecto falla o la detecta muy tarde
                    
                    if (np.sum(outputs))>0: # FP
                        FP[i]+=1 #si hay algun flag perdido por ahi
                        
                    else:
                        if threshold[i]>0.2:
                            FN[i]+=1 #si estan todos en cero
                        
                else: #si levanto el flag a tiempo, me fijo que paso con los otros flags                   
                    
                    if np.sum(outputs[:-(failpoints-fail_time),:])>0: #si alguien levanto antes
                        FP[i]+=1
                    else:           #si levanto primero
                        TP[i]+=1
                        d = np.append(d,fail_time) #guarda el delay
                    

    
        if TP[i]==0: # corrijo indeterminaciones
            d = 0
            R[i] = 1
    
        delay[i] = np.mean(d)*(1000/fs)
        sigma_delay[i] = np.std(d)*(1000/fs)

        maximo[i] = np.max(d)*(1000/fs)
        minimo[i] = np.min(d)*(1000/fs)
    
       #precision and recall: se deberia contemplar casos donde no entra aca, y hay que corregir indeterminacion
        if ((TP[i]+FP[i])>0 and (TP[i]+FN[i])>0):
            P[i] = TP[i]/(TP[i]+FP[i]) #cuando subo el treshold, bajan los FP, sube P.
            R[i] = TP[i]/(TP[i]+FN[i]) #cuando subo el treshold, suben los FN, baja R
    
        falses[i] = FP[i]/(rotores*sims*time_all/3600)    # Falsos por hora de vuelo
    
    
        dist_threshold[i]=np.sqrt((P[i]-1)**2+(R[i]-1)**2)

    #INDETERMINACIONES Y EXTREMOS
    P = np.concatenate(([0],P,[1]))
    R = np.concatenate(([1],R,[0]))
    area = trapz(R,P)*100


    #save to file
    np.save('Results/Metricas/P_n{}_fs{}'.format(n,Fs), P)
    np.save('Results/Metricas/R_n{}_fs{}'.format(n,Fs), R)
    np.save('Results/Metricas/falses_n{}_fs{}'.format(n,Fs), falses)
    np.save('Results/Metricas/delay_n{}_fs{}'.format(n,Fs), delay)
    np.save('Results/Metricas/sigma_delay_n{}_fs{}'.format(n,Fs), sigma_delay)
    np.save('Results/Metricas/area_n{}_fs{}'.format(n,Fs), area)
    np.save('Results/Metricas/maximo_n{}_fs{}'.format(n,Fs), maximo)
    np.save('Results/Metricas/minimo_n{}_fs{}'.format(n,Fs), minimo)
    np.save('Results/Metricas/dist_treshold_n{}_fs{}'.format(n,Fs),dist_threshold)
