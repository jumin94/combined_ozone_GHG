#Lo primero que hace es cargar todos los datos de viento
#Luego la diferencia de VB date.
#Debería entregar un array con todos los valores del índice para cada año.

import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import glob
import os, fnmatch
from datetime import datetime
import math
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import statsmodels.api as sm


#Abro los datos--------------------------------------------
ruta = '/home/julia.mindlin/Trabajo/CMIP6_ozone/timeseries' 
models = [
    'ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CAMS-CSM1-0', 
    'CanESM5', 'CESM2_', 'CESM2-WACCM', 'CNRM-CM6-1', 'CNRM-ESM2-1',
    'EC-Earth3', 'FGOALS-g3', 'HadGEM3-GC31-LL', 'INM-CM4-8', 
    'INM-CM5-0', 'MIROC6', 'MIROC-ES2L', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 
    'MRI-ESM2-0', 'NESM3', 'NorESM2-LM', 'NorESM2-MM', 'UKESM1-0-LL'
    ]
os.chdir(ruta)
os.getcwd()

class my_dictionary(dict):
    # __init__ function 
    def __init__(self):
        self = dict()
    # Function to add key:value 
    def add(self, key, value):
        self[key] = value

def abro_dato(models:list):
    """Abre el dato

    Args:
        model (list): modelos a analizar

    Returns:
        dict: DataFrames con series temporales
    """
    dic = {}
    listOfFiles = os.listdir(ruta)
    for model in models:
        dic[model] = []
        pattern = "*"+model+"*"
        for entry in listOfFiles:
            if fnmatch.fnmatch(entry,pattern):
                print(ruta+'/'+entry)
                df = pd.read_csv(ruta+'/'+entry)
                dic[model] = df
    return dic

dato_ua = abro_dato(models)

#Maneja los dias para que queden por encima de 300, para tener una serie consistente
def convert_day(a):
    for i in range(len(a)):
        if a[i] < 200 and a[i]!=0:
            a[i] = a[i] + 365
        else:
            a[i] = a[i]
    for j in range(len(a)):
        if a[j] == 0:
            media = np.mean(a[j-20:j]); std = np.std(a[j-20:j])
            a[j] = random.randint(-int(std),int(std)) + media
        else:
            a[j] = a[j]
    return a 

dic_vb = {}
for model in models:
    dic_vb[model] = []
    index = dato_ua[model].keys()[2:]
    for i in range(len(index)):
        dat = convert_day(dato_ua[model][index[i]])
        clima = np.mean(dat[:20])
        anom = dat - clima
        dic_vb[model].append(anom)
            		
#Regresiones-------------------------------------------------------------
#Genero array de años
years = np.arange(1950,2015,1)
years_futuro = np.arange(2015,2099,1)

#Abro los regressors (temperatura en sup y EESC_polar)
GW = pd.read_csv('/home/julia.mindlin/Trabajo/CMIP6_ozone/GW_timeseries.csv')
EESC_pol = pd.read_csv('/home/julia.mindlin/Trabajo/CMIP6_ozone/GW_EESC_polar_ozoneloss.csv')
Years = EESC_pol['Years']
EESC = EESC_pol['EESC_polar']

#Funciones para calcular regresiones
def regresion_modelo(modelo,k):
        regresores = pd.DataFrame({'GW':GW[modelo][:149],'EESC':EESC_pol['EESC_polar'][:149],'GW_EESC':GW[modelo][:149]*EESC_pol['EESC_polar'][:149]})
        #y = sm.add_constant(regresores.values)
        reg = linear_model.LinearRegression()
        x = dic_vb[modelo][k].values
        res = sm.OLS(x,regresores).fit()
        a = res.params[0]
        b = res.params[1]
        c = res.params[2]
        r2 = res.rsquared
        c_i = res.conf_int(alpha=0.05,cols=None)
        return a, b, c, r2, c_i

def regresiones(vb_models,models):
    dic = {}
    for model in models:
        dic[model] = {}
        for k in range(len(vb_models[model])):
            dic[model][k] = []
            a,b,c,r2,c_i = regresion_modelo(model,k)
            dic[model][k].append(a); dic[model][k].append(b); dic[model][k].append(c)
            dic[model][k].append(r2); dic[model][k].append(c_i)
    return dic

regresiones_todas = regresiones(dic_vb,models)

dic = {}
for model in models:
    print(model)
    GW_a = np.array([])
    EESC_b = np.array([])
    GW_EESC_c = np.array([])
    R2 = np.array([])
    N = len(regresiones_todas[model])
    if N == 1:
        GWa = regresiones_todas[model][0][0]
        EESCb = regresiones_todas[model][0][1]
        GW_EESC = regresiones_todas[model][0][2]
        r2 = regresiones_todas[model][0][3]
        std_GW = regresiones_todas[models[0]][0][4][1]['GW'] - regresiones_todas[models[0]][0][4][0]['GW']
        std_EESC = regresiones_todas[models[0]][0][4][1]['EESC'] - regresiones_todas[models[0]][0][4][0]['EESC']
        std_GW_EESC = regresiones_todas[models[0]][0][4][1]['GW_EESC'] - regresiones_todas[models[0]][0][4][0]['GW_EESC']
        dic[model] = [GWa, std_GW, EESCb, std_EESC,GW_EESC,std_GW_EESC,r2]
    else:
        for k in range(N):
            GW_a = np.append(regresiones_todas[model][k][0],GW_a)
            EESC_b = np.append(regresiones_todas[model][k][1],EESC_b)
            GW_EESC_c = np.append(regresiones_todas[model][k][2],GW_EESC_c)
            R2 = np.append(regresiones_todas[model][k][3],R2)
            mean_GW = np.mean(GW_a)
            std_GW = np.std(GW_a)
            mean_EESC = np.mean(EESC_b)
            std_EESC = np.std(EESC_b)
            mean_GW_EESC = np.mean(GW_EESC_c)
            std_GW_EESC = np.std(GW_EESC_c)
            r2 = np.mean(R2)
            dic[model] = [mean_GW, std_GW, mean_EESC, std_EESC,mean_GW_EESC,std_GW_EESC,r2]

    
#Guardamos los datos en DataFrames
anos = np.append(years,years_futuro)
Regresiones = {
    models[0]:dic[models[0]],models[1]:dic[models[1]],models[2]:dic[models[2]],
    models[3]:dic[models[3]],models[4]:dic[models[4]],models[5]:dic[models[5]],
    models[6]:dic[models[6]],models[7]:dic[models[7]],models[8]:dic[models[8]],
    models[9]:dic[models[9]],models[10]:dic[models[10]],models[11]:dic[models[11]],
    models[12]:dic[models[12]],models[13]:dic[models[13]],models[14]:dic[models[14]],
    models[15]:dic[models[15]],models[16]:dic[models[16]],models[17]:dic[models[17]],
    models[18]:dic[models[18]],models[19]:dic[models[19]],models[20]:dic[models[20]],
    models[21]:dic[models[21]],models[22]:dic[models[22]]
    }
Reg = pd.DataFrame(Regresiones)
GW.insert(0,"Years", anos,True)
Reg.to_csv('/home/julia.mindlin/Trabajo/CMIP6_ozone/Regresiones.csv', float_format='%g')

models_full = ['CNRM-CM6-1', 'CNRM-ESM2-1', 'MIROC6', 'MIROC-ES2L', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-MM']
anos = np.append(years,years_futuro)
VB = {'Year': anos,models_full[0]:dic[models_full[0]],models_full[1]:dic[models_full[1]],models_full[2]:dic[models_full[2]],models_full[3]:dic[models_full[3]],models_full[4]:dic[models_full[4]],models_full[5]:dic[models_full[5]],models_full[6]:dic[models_full[6]],models_full[7]:dic[models_full[7]]}
VB = pd.DataFrame(VB)
VB.insert(0,"Years", anos,True)
VB.to_csv('/home/julia.mindlin/Trabajo/CMIP6_ozone/VB_timeseries.csv', float_format='%g')

