
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
vars = ['ua/day','ta','tas']
models = ['ACCESS-CM2','ACCESS-ESM1-5','BCC-CSM2-MR','CAMS-CSM1-0','CanESM5','CESM2_','CESM2-WACCM','CNRM-CM6-1','CNRM-ESM2-1','EC-Earth3','FGOALS-g3','HadGEM3-GC31-LL','INM-CM4-8','INM-CM5-0','MIROC6','MIROC-ES2L','MPI-ESM1-2-HR','MPI-ESM1-2-LR','MRI-ESM2-0','NESM3','NorESM2-LM','NorESM2-MM','UKESM1-0-LL']
models2 = ['IPSL-CM6A-LR'] #,'UKESM1-0-LL']
scenarios = ['historical','ssp585']
scenarios2 = ['historical']
os.chdir(ruta)
os.getcwd()

class my_dictionary(dict):
    # __init__ function 
    def __init__(self):
        self = dict()
    # Function to add key:value 
    def add(self, key, value):
        self[key] = value

def cargo_todo(scenarios,models,vars):
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
                        #dato = xr.open_dataset(ruta+'/'+scenario+'/'+var+'/'+entry)
                        print(ruta+'/'+scenario+'/'+var+'/'+entry)
                        #if var == 'ta':
                        #	dato = dato.sel(plev=slice(26000,24000))	
                        #else:
                        #	dato = dato.sel(plev=slice(5100,4900))
                        dic[scenario][var][model].append(ruta+'/'+scenario+'/'+var+'/'+entry)
        return dic

#Diccionario con las rutas de los archivos
dato = cargo_todo(scenarios,models,vars)

#Funcion para calcular dia juliano
def julian(date):
    try:
        yday = datetime.utcfromtimestamp(date.tolist()/1e9).timetuple().tm_yday
    except AttributeError or TypeError:
        yday = date.timetuple().tm_yday
    return yday

#Genero array de dias
years = np.arange(1950,2015,1)
years_futuro = np.arange(2015,2099,1)

#Genero un diccionario 
dic = {}
vb_h = []
for model in models:
    print(model)
    dic[model] = np.array([])
    var_hist = xr.open_dataset(dato[scenarios[0]]['ua/day'][model][0])
    var_hist = var_hist.sel(plev=slice(5100,4900))
    var_h = var_hist.ua
    var_h.attrs = var_hist.ua.attrs
    var_h = var_h.isel(plev=0)
    var_h = var_h.sel(lat=slice(-60,-50))
    var_h = var_h.mean(dim='lat')
    var_hu = var_h.mean(dim='lon')
    for year in years:
        a = str(year)+'-10'
        b = str(year)+'-12'
        ua = var_hu.sel(time=slice(a,b))
        T = ua.time
        for i in range(len(ua)):
            if not (ua[i] > 19): #Encuentra la fecha del breakdown
                vb1 = julian(T.values[i]) 
                print(vb1,year)
                dic[model] = np.append(dic[model],vb1) #Agrega al array
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
                if not (ua[i] > 19):
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

#Genera la serie de temperatura global en superficie
dic_ta = {}
for model in models:
	print(model)
	dic_ta[model] = np.zeros(65)
	ta_hist = xr.open_dataset(dato[scenarios[0]]['ta'][model][0])
	#ta_hist = ta_hist.sel(plev=slice(26000,24000))
	ta_h = ta_hist.ta
	ta_h.attrs = ta_hist.ta.attrs
	ta_h = ta_h.isel(plev=0)
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
        ta_r = ta_r.isel(plev=0)
        ta_r = ta_r.sel(lat=slice(-15,15))
        ta_r = ta_r.mean(dim='lat')
        ta_r = ta_r.mean(dim='lon')
        for year in years_futuro:
            a = str(year)+'-12'
            b = str(year+1)+'-02'
            ua = ta_r.sel(time=slice(a,b))
            TA = ua.mean(dim='time')
            dic_ta[model] = np.append(dic_ta[model],TA)

dic_tas = {}
for model in models:
    print(model)
    dic_tas[model] = np.zeros(65)
    ta_hist = xr.open_dataset(dato[scenarios[0]]['tas'][model][0])
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

#Guardamos los datos en DataFrames
anos = np.append(years,years_futuro)
GW = {'Year': anos,models[0]:dic_ta[models[0]],models[1]:dic_ta[models[1]],models[2]:dic_ta[models[2]],models[3]:dic_ta[models[3]],models[4]:dic_ta[models[4]],models[5]:dic_ta[models[5]],models[6]:dic_ta[models[6]],models[7]:dic_ta[models[7]],models[8]:dic_ta[models[8]],models[9]:dic_ta[models[9]],models[10]:dic_ta[models[10]],models[11]:dic_ta[models[11]],models[12]:dic_ta[models[12]],models[13]:dic_ta[models[13]],models[14]:dic_ta[models[14]],models[15]:dic_ta[models[15]],models[16]:dic_ta[models[16]],models[17]:dic_ta[models[17]],models[18]:dic_ta[models[18]],models[19]:dic_ta[models[19]],models[20]:dic_ta[models[20]],models[21]:dic_ta[models[21]],models[22]:dic_ta[models[22]],models[23]:dic_ta[models[23]],models[24]:dic_ta[models[24]]}
GW = pd.DataFrame(GW)
GW.insert(0,"Years", anos,True)
GW.to_csv('/home/julia.mindlin/Trabajo/CMIP6_ozone/GW_timeseries.csv', float_format='%g')

models_full = ['CNRM-CM6-1','CNRM-ESM2-1','MIROC6','MIROC-ES2L','MPI-ESM1-2-HR','MPI-ESM1-2-LR','MRI-ESM2-0','NorESM2-MM']
anos = np.append(years,years_futuro)
VB = {'Year': anos,models_full[0]:dic[models_full[0]],models_full[1]:dic[models_full[1]],models_full[2]:dic[models_full[2]],models_full[3]:dic[models_full[3]],models_full[4]:dic[models_full[4]],models_full[5]:dic[models_full[5]],models_full[6]:dic[models_full[6]],models_full[7]:dic[models_full[7]]}
VB = pd.DataFrame(VB)
VB.insert(0,"Years", anos,True)
VB.to_csv('/home/julia.mindlin/Trabajo/CMIP6_ozone/VB_timeseries.csv', float_format='%g')

