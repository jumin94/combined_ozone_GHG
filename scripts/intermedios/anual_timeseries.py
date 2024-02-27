#Lo primero que hace es cargar todos los datos de viento
#Luego la diferencia de VB date.
#Debería entregar un array con todos los valores del índice para cada año.

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

#Abro los datos--------------------------------------------
ruta = '/pikachu/datos/julia.mindlin/CMIP6_ensambles' #/historical/mon/tas/past'
vars = ['ta']
models = [
    'ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CAMS-CSM1-0', 'CanESM5',
    'CESM2_', 'CESM2-WACCM', 'CNRM-CM6-1', 'CNRM-ESM2-1', 'EC-Earth3', 'FGOALS-g3',
    'GFDL-ESM4', 'HadGEM3-GC31-LL', 'INM-CM4-8', 'INM-CM5-0', 'MIROC6', 'MIROC-ES2L',
    'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NESM3', 'NorESM2-LM', 'NorESM2-MM',
    'UKESM1-0-LL'
    ]

ruta1_hist = '/pikachu/datos/CMIP6_backup/Data_used/historical/day/ua'
ruta2_hist = '/pikachu/datos4/CMIP6/historical/day/ua'
ruta1_ssp = '/pikachu/datos2/CMIP6/ssp585/day/ua'
ruta2_ssp = '/pikachu/datos/CMIP6_backup/Data_used/ssp585/day/ua'
models = [
    'ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CAMS-CSM1-0', 
    'CanESM5', 'CESM2_', 'CESM2-WACCM', 'CNRM-CM6-1', 'CNRM-ESM2-1',
    'EC-Earth3', 'FGOALS-g3', 'GFDL-ESM4', 'HadGEM3-GC31-LL', 'INM-CM4-8', 
    'INM-CM5-0', 'MIROC6', 'MIROC-ES2L', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 
    'MRI-ESM2-0', 'NESM3', 'NorESM2-LM', 'NorESM2-MM', 'UKESM1-0-LL'
    ]
#Tomo todos los miembros que tengo para cada modelo
#Como estan en discos distintos, los cargo por separado 
models1_h = ['CESM2_', 'CESM2-WACCM', 'EC-Earth3', 'GFDL-ESM4', 
    'INM-CM5-0', 'MPI-ESM1-2-HR']
models2_h = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR',
    'CAMS-CSM1-0', 'CanESM5', 'CNRM-CM6-1',
    'CNRM-ESM2-1', 'INM-CM4-8', 'MIROC6', 'MIROC-ES2L',  
    'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NESM3',
    'NorESM2-LM', 'NorESM2-MM', 'HadGEM3-GC31-LL', 'UKESM1-0-LL'
    ]
models2_s = [
    'CESM2_', 'FGOALS-g3', 'INM-CM4-8', 'INM-CM5-0', 'MIROC6', 'MIROC-ES2L', 'MPI-ESM1-2-HR',
    'MRI-ESM2-0', 'NESM3', 'NorESM2-LM', 'NorESM2-MM', 'UKESM1-0-LL', 'HadGEM3-GC31-LL'
    ]
models1_s = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CAMS-CSM1-0', 'CanESM5',
    'CESM2-WACCM', 'CNRM-CM6-1', 'CNRM-ESM2-1', 'EC-Earth3', 'MPI-ESM1-2-LR', 'GFDL-ESM4',
    'HadGEM3-GC31-LL']


scenarios = ['historical', 'ssp585']
os.chdir(ruta)
os.getcwd()

class my_dictionary(dict):
    # __init__ function 
    def __init__(self):
        self = dict()
    # Function to add key:value 
    def add(self, key, value):
        self[key] = value

def cargo_todo(scenarios:list,models:list,vars:list):
    """Esta funcion carga todo

    Args:
        scenarios (list): experimento CMIP
        models (list): lista de modelos
        vars (list): variables a cargar

    Returns:
        dict: rutas a los datos
    """
    dic = {}
    dic['ssp585'] = {}
    dic['historical'] = {}
    for scenario in dic.keys():
        for var in vars:
            listOfFiles = os.listdir(ruta+'/'+scenario+'/'+var)
            dic[scenario][var] = {}
            for model in models:
                dic[scenario][var][model] = []
                pattern = "*"+model+"*"+scenario+"*"
                for entry in listOfFiles:
                    if fnmatch.fnmatch(entry,pattern):
                        print(ruta+'/'+scenario+'/'+var+'/'+entry)
                        dic[scenario][var][model].append(ruta+'/'+scenario+'/'+var+'/'+entry)
    return dic

years = np.arange(1950,2015,1); years_futuro = np.arange(2015,2099,1)
dato = cargo_todo(scenarios,models,vars)

#Genera la serie de temperatura global en superficie
dic_ta = {}
for model in models:
    print(model)
    dic_ta[model] = np.zeros(65)
    ta_hist = xr.open_dataset(dato[scenarios[0]]['ta'][model][0])
    ta_hist = ta_hist.sel(plev=slice(26000,24000))
    ta_h = ta_hist.ta
    ta_h.attrs = ta_hist.ta.attrs
    #ta_h = ta_h.isel(plev=8)
    ta_h = ta_h.sel(lat=slice(-15,15))
    ta_h = ta_h.mean(dim='lat')
    ta_h = ta_h.mean(dim='lon')
    cont = 0
    for year in years:
        a = str(year)+'-12'
        b = str(year+1)+'-02'
        ua = ta_h.sel(time=slice(a,b))
        TA = ua.mean(dim='time')
        dic_ta[model][cont] = TA
        cont+=1

for model in models:
    print(model)
    ta_rcp = xr.open_dataset(dato[scenarios[1]]['ta'][model][0])
    #ta_rcp = ta_rcp.sel(plev=slice(26000,24000))
    ta_r = ta_rcp.ta
    ta_r.attrs = ta_rcp.ta.attrs
    ta_r = ta_r.sel(plev=slice(26000,24000))
    ta_r = ta_r.sel(lat=slice(-15,15))
    ta_r = ta_r.mean(dim='lat')
    ta_r = ta_r.mean(dim='lon')
    for year in years_futuro:
        a = str(year)+'-12'
        b = str(year+1)+'-02'
        ua = ta_r.sel(time=slice(a,b))
        TA = ua.mean(dim='time')
        dic_ta[model] = np.append(dic_ta[model],TA)

#Guardamos los datos en DataFrames
anos = np.append(years,years_futuro)
TW = {'Year': anos,models[0]:dic_ta[models[0]],models[1]:dic_ta[models[1]],
    models[2]:dic_ta[models[2]],models[3]:dic_ta[models[3]],
    models[4]:dic_ta[models[4]],models[5]:dic_ta[models[5]],
    models[6]:dic_ta[models[6]],models[7]:dic_ta[models[7]],
    models[8]:dic_ta[models[8]],models[9]:dic_ta[models[9]],
    models[10]:dic_ta[models[10]],models[11]:dic_ta[models[11]],
    models[12]:dic_ta[models[12]],models[13]:dic_ta[models[13]],
    models[14]:dic_ta[models[14]],models[15]:dic_ta[models[15]],
    models[16]:dic_ta[models[16]],models[17]:dic_ta[models[17]],
    models[18]:dic_ta[models[18]],models[19]:dic_ta[models[19]],
    models[20]:dic_ta[models[20]],models[21]:dic_ta[models[21]],
    models[22]:dic_ta[models[22]],models[23]:dic_ta[models[23]]}
TW = pd.DataFrame(TW)
TW.insert(0,"Years", anos,True)
TW.to_csv('/home/julia.mindlin/Trabajo/CMIP6_ozone/TW_timeseries.csv', float_format='%g')



dic_tas = {}
for model in models:
    print(model)
    dic_tas[model] = np.zeros(65)
    ta_hist = xr.open_dataset(dato[scenarios[0]]['tas'][model][0])
    if model == 'KACE-1-0-G':
        ta_h = ta_hist.ta
        ta_h.attrs = ta_hist.ta.attrs
    else:
        ta_h = ta_hist.tas
        ta_h.attrs = ta_hist.tas.attrs
        lats = np.cos(ta_h.lat.values*np.pi/180)
        s = sum(lats)
        ta_h = ta_h.mean(dim='lon')
        ta_h = ta_h.fillna(ta_h[-1]-1)
        cont = 0
    for year in years:
        a = str(year)+'-12'
        b = str(year+1)+'-02'
        tas = ta_h.sel(time=slice(a,b))
        tas = tas.mean(dim='time')
        TAS = sum(tas*lats)/s
        dic_tas[model][cont] = TAS
        cont+=1

for model in models:
    print(model)
    ta_rcp = xr.open_dataset(dato[scenarios[1]]['tas'][model][0])
    ta_r = ta_rcp.tas
    ta_r.attrs = ta_rcp.tas.attrs
    lats = np.cos(ta_r.lat.values*np.pi/180)
    s = sum(lats)
    ta_r = ta_r.mean(dim='lon')
    ta_r = ta_r.fillna(ta_r[-1]-1)
    ta_r = sum(ta_r*lats)/s
    for year in years_futuro:
        a = str(year)+'-12'
        b = str(year+1)+'-02'
        tas = ta_r.sel(time=slice(a,b))
        TAS = tas.mean(dim='time')
        dic_tas[model] = np.append(dic_tas[model],TAS)


#Vortex breakdown date time series
def cargo_todo_ua(scenarios, ruta_h1, models_h1, 
ruta_h2, models_h2, ruta_s1, models_s1, ruta_s2,
models_s2):
    """Guarda los datos de las dos rutas

    Args:
        scenarios (list): [description]
        ruta_h1 (str): [description]
        models_h1 (list): [description]
        ruta_h2 (str): [description]
        models_h2 (list): [description]
        ruta_s1 (str): [description]
        models_s1 (list): [description]
        ruta_s2 (str): [description]
        models_s2 (list): [description]

    Returns:
        dict: todas las rutas a todos los archivos
    """
    dic = {}
    dic['ssp585'] = {}
    dic['historical'] = {}
    for scenario in dic.keys():
        if scenario == 'historical':
            listOfFiles1 = os.listdir(ruta_h1+'/')
            listOfFiles2 = os.listdir(ruta_h2+'/')
            dic[scenario] = {}
            for model in models_h1:
                dic[scenario][model] = []
                pattern = "*day*"+model+"*"+scenario+"*"
                for entry in listOfFiles1:
                    if fnmatch.fnmatch(entry,pattern):
                        dic[scenario][model].append(ruta_h1+'/'+entry)
            for model in models_h2:
                dic[scenario][model] = []
                pattern = "*day*"+model+"*"+scenario+"*"
                for entry in listOfFiles2:
                    if fnmatch.fnmatch(entry,pattern):
                        dic[scenario][model].append(ruta_h2+'/'+entry) 
        else:
            listOfFiles1 = os.listdir(ruta_s1+'/')
            listOfFiles2 = os.listdir(ruta_s2+'/')
            dic[scenario] = {}
            for model in models_s1:
                dic[scenario][model] = []
                pattern = "*day*"+model+"*"+scenario+"*"
                for entry in listOfFiles1:
                    if fnmatch.fnmatch(entry,pattern):
                        dic[scenario][model].append(ruta_s1+'/'+entry)
            for model in models_s2:
                dic[scenario][model] = []
                pattern = "*day*"+model+"*"+scenario+"*"
                for entry in listOfFiles2:
                    if fnmatch.fnmatch(entry,pattern):
                        dic[scenario][model].append(ruta_s2+'/'+entry)
    return dic

dato_ua = cargo_todo_ua(scenarios, ruta1_hist, 
        models1_h, ruta2_hist, models2_h, ruta1_ssp, models1_s, 
        ruta2_ssp, models2_s)
#Diccionario con las rutas de los archivos
dato = cargo_todo(scenarios,models,vars)

#Funcion para calcular dia juliano
def julian(date):
    """Calcula el dia juliano para una fecha

    Args:
        date ([type]): [description]

    Returns:
        [type]: [description]
    """
    try:
        yday = datetime.utcfromtimestamp(date.tolist()/1e9).timetuple().tm_yday
    except (AttributeError,TypeError):
        yday = date.timetuple().tm_yday
    return yday

#Genero array de años
years = np.arange(1950,2015,1)
years_futuro = np.arange(2015,2099,1)

#Genero un diccionario 
dic = {}
vb_h = []
for model in models:
    print(model)
    dic[model] = np.array([])
    if model in models2:
        var_hist = xr.open_dataset(dato[scenarios[0]]['ua/day'][model][0])
        var_hist = var_hist.sel(plev=slice(5100, 4900))
        var_h = var_hist.ua
        var_h.attrs = var_hist.ua.attrs
        var_h = var_h.isel(plev=0)
        var_h = var_h.sel(lat=slice(-60,-50))
        var_hu = var_h.mean(dim='lat')
    else:
        var_hist = xr.open_dataset(dato[scenarios[0]]['ua/day'][model][0])
        var_hist = var_hist.sel(plev=slice(5100, 4900))
        var_h = var_hist.ua
        var_h.attrs = var_hist.ua.attrs
        var_h = var_h.isel(plev=0)
        var_h = var_h.sel(lat=slice(-60,  -50))
        var_h = var_h.mean(dim='lat')
        var_hu = var_h.mean(dim='lon')
    for year in years:
        a = str(year)+'-10'
        b = str(year)+'-12'
        ua = var_hu.sel(time=slice(a, b))
        T = ua.time
        for i in range(len(ua)):
            if not ua[i] > 15:  #Encuentra la fecha del breakdown
                vb1 = julian(T.values[i]) 
                print(vb1, year)
                dic[model] = np.append(dic[model], vb1) #Agrega al array
                break

#Genera la serie para el experimento ssp585
for model in models:
    print(model)
    var_rcp = xr.open_dataset(dato[scenarios[1]]['ua/day'][model][0])
    var_rcp = var_rcp.sel(plev=slice(5100,4900))
    var_r = var_rcp.ua
    var_r.attrs = var_rcp.ua.attrs
    var_r = var_r.isel(plev=0)
    var_r = var_r.sel(lat=slice(-60,-50))
    var_r = var_r.mean(dim='lat')
    var_ru = var_r.mean(dim='lon')
    for year in years_futuro:
        a = str(year)+'-10'
        b = str(year+1)+'-02'
        ua = var_ru.sel(time=slice(a,b))
        T = ua.time
        for i in range(len(ua)):
            if not (ua[i] > 15):
                vb1 = julian(T.values[i])
                print(vb1,year)
                dic[model] = np.append(dic[model],vb1)
                break

#Maneja los dias para que queden por encima de 300, para tener una serie consistente
def convert_day(a):
    for i in range(len(a)):
        if a[i] < 200:
            a[i] = a[i] + 365
        else:
            a[i] = a[i]
    return a 

dic_conv = {}
for model in models:
    dic_conv[model] = convert_day(dic[model])
    ref_bd_date = np.mean(dic_conv[model][:20])
    dic_conv[model] = dic_conv[model] - ref_bd_date


models_full = ['CNRM-CM6-1', 'CNRM-ESM2-1', 'MIROC6', 'MIROC-ES2L', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-MM']
anos = np.append(years,years_futuro)
VB = {'Year': anos,models_full[0]:dic[models_full[0]],models_full[1]:dic[models_full[1]],models_full[2]:dic[models_full[2]],models_full[3]:dic[models_full[3]],models_full[4]:dic[models_full[4]],models_full[5]:dic[models_full[5]],models_full[6]:dic[models_full[6]],models_full[7]:dic[models_full[7]]}
VB = pd.DataFrame(VB)
VB.insert(0,"Years", anos,True)
VB.to_csv('/home/julia.mindlin/Trabajo/CMIP6_ozone/VB_timeseries.csv', float_format='%g')

