import pandas as pd
import numpy as np
import pathlib

# In[1]: Declaro parametros a usar.

n = 6
fs = 200
n_e = 20

sims = 1500
time_fail = 0.5
time_all = 30
max_delay = 0.4 # 400ms es el maximo tiempo de deteccion

rotores = 6
labelnumber = 6
factor = round(200/fs)

datapoints = int(fs*time_all+1)
failpoints = int(fs*time_fail)
maxpoints = int(max_delay*fs)
    
Fs = int(np.floor(fs))
path = 'Results/Outputs/RF_n{}_fs{}'.format(n,Fs)

P = 0
R = 0
TP = 0
FP = 0
FN = 0

threshold = 0.95
falses = 0
delay = 0
sigma_delay = 0
dist_threshold = 0

maximo = 0
minimo = 0

print('Calculando metricas para n={} a {}Hz'.format(n,Fs))

d = []


for r in range (rotores):

    ddir = pathlib.Path(path+"/fail{}".format(r+1))

    for s in range(sims):

        fn = ddir/"run{}".format(s+1)
        assert(fn.exists())

        outputs = pd.read_csv(fn).values[:,-labelnumber:]>threshold # necesito tirar el indice del csv
        
        fail_time = np.argmax(outputs[-failpoints:,r]) # busco si levanta el flag correcto y cuando lo hace
        outputs[-failpoints:,r] = 0 #plancho el flag

        if fail_time == 0 or fail_time >= maxpoints: # si no detecto falla o la detecta muy tarde
            
            if (np.sum(outputs))>0: # FP
                FP+=1 #si hay algun flag perdido por ahi
                
            else:
                if threshold>0.2:
                    FN+=1 #si estan todos en cero
                
        else: #si levanto el flag a tiempo, me fijo que paso con los otros flags                   
            
            if np.sum(outputs[:-(failpoints-fail_time),:])>0: #si alguien levanto antes
                FP+=1
            else:           #si levanto primero
                TP+=1
                d = np.append(d,fail_time) #guarda el delay
                

      
if TP==0: # corrijo indeterminaciones
    d = 0
    R = 1
    
delay = np.mean(d)*(1000/fs)
sigma_delay = np.std(d)*(1000/fs)

maximo = np.max(d)*(1000/fs)
minimo = np.min(d)*(1000/fs)

   #precision and recall: se deberia contemplar casos donde no entra aca, y hay que corregir indeterminacion
if ((TP+FP)>0 and (TP+FN)>0):
    P = TP/(TP+FP) #cuando subo el treshold, bajan los FP, sube P.
    R = TP/(TP+FN) #cuando subo el treshold, suben los FN, baja R

falses = FP/(rotores*sims*time_all/3600)    # Falsos por hora de vuelo


dist_threshold=np.sqrt((P-1)**2+(R-1)**2)

        
print('Resultados n{}_fs{}: '.format(n,Fs))  
#print('Threshold:{}'.format(threshold))          
#print('TP:{}'.format(TP))
#print('FP:{}'.format(FP))        
#print('FN:{}'.format(FN))

print('Media: {}'.format(delay))
print('Desvio: {}'.format(sigma_delay))
print('Maximo: {}'.format(maximo))
print('Minimo: {}'.format(minimo))


R=TP/(TP+FN)
P=TP/(TP+FP)
        
dist_threshold =np.sqrt((P-1)**2+(R-1)**2)
