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
import random

#Abro los datos--------------------------------------------
ruta = '/pikachu/datos/julia.mindlin/CMIP6_ensambles' #/historical/mon/tas/past'
vars = ['ua/day'] #, 'ta', 'tas']

ruta1_hist = '/pikachu/datos/CMIP6_backup/Data_used/historical/day/ua'
ruta2_hist = '/pikachu/datos4/CMIP6/historical/day/ua'
ruta1_ssp = '/pikachu/datos2/CMIP6/ssp585/day/ua'
ruta2_ssp = '/pikachu/datos/CMIP6_backup/Data_used/ssp585/day/ua'
#models = [
#    'ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CAMS-CSM1-0', 
#    'CanESM5', 'CESM2_', 'CESM2-WACCM', 'CNRM-CM6-1', 'CNRM-ESM2-1',
#    'EC-Earth3', 'FGOALS-g3', 'GFDL-ESM4', 'HadGEM3-GC31-LL', 'INM-CM4-8', 
#    'INM-CM5-0', 'MIROC6', 'MIROC-ES2L', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 
#    'MRI-ESM2-0', 'NESM3', 'NorESM2-LM', 'NorESM2-MM', 'UKESM1-0-LL'
#    ]

models = ['MRI-ESM2-0', 'NESM3', 'NorESM2-LM', 'NorESM2-MM']

#Tomo todos los miembros que tengo para cada modelo
#Como estan en discos distintos, los cargo por separado 
models1_h = ['CESM2_', 'CESM2-WACCM', 'EC-Earth3', 'GFDL-ESM4', 
    'INM-CM5-0', 'MPI-ESM1-2-HR','FGOALS-g3']
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

#Solo nuevos 
models1_h = [] #['TaiESM1','IITM-ESM',]
models2_h = ['HadGEM3-GC31-MM'] #,'IPSL-CM6A-LR','KACE-1-0-G', 'CMCC-CM2-SR5']
models1_s = ['HadGEM3-GC31-MM']#['TaiESM1','IITM-ESM','CMCC-CM2-SR5']
models2_s = [] #,'IPSL-CM6A-LR', 'KACE-1-0-G'] 

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

#Genero array de dias
years = np.arange(1950,2015,1)
years_futuro = np.arange(2015,2099,1)
anos = np.append(years,years_futuro)


#Genero un diccionario por modelo 
def ensamble_series2(datos:dict, model:str):
    """Genera un DataFrame con las series temporales
    del ensamble generado para el modelo. 
    N : # de miembros historical
    M : # de miembros ssp585
    N*M + 1 columnas

    Args:
        data (dict): Diccionario con los miembros
        model (str): Nombre del modelo

    Returns:
        DataFrame: Pandas DataFrame con las n series temporales
    """
    n = len(datos['historical'][model])
    m = len(datos['ssp585'][model])
    dic = {}; cont = 0
    for i in range(n):
        var_hist = xr.open_dataset(datos['historical'][model][i])
        var_hist = var_hist.sel(plev=slice(5100, 4900))
        var_h = var_hist.ua
        var_h.attrs = var_hist.ua.attrs
        var_h = var_h.isel(plev=0)
        var_hu = var_h.sel(lat=slice(-60,-50))
        for j in range(m):
                dic[str(cont)] = np.zeros(len(anos)); cuenta = 0
                var_ssp = xr.open_dataset(datos['ssp585'][model][j])
                var_ssp = var_ssp.sel(plev=slice(5100,4900))
                var_s = var_ssp.ua
                var_s.attrs = var_ssp.ua.attrs
                var_s = var_s.isel(plev=0)
                var_su = var_s.sel(lat=slice(-60,-50))
                for year in years:
                    a = str(year)+'-10'
                    b = str(year)+'-12'
                    ua = var_hu.sel(time=slice(a, b))
                    ua = ua.mean(dim='lat')
                    ua = ua.mean(dim='lon')
                    T = ua.time
                    for k in range(len(ua)):
                        if not ua[k] > 19:  #Encuentra la fecha del breakdown
                            vb1 = julian(T.values[k]) 
                            print(vb1, year)
                            dic[str(cont)][cuenta] = vb1 #Agrega al array
                            cuenta +=1
                            break
                for year in years_futuro:
                    a = str(year)+'-10'
                    b = str(year+1)+'-02'
                    ua = var_su.sel(time=slice(a, b))
                    ua = ua.mean(dim='lat')
                    ua = ua.mean(dim='lon')
                    T = ua.time
                    for k in range(len(ua)):
                        if not ua[k] > 19:  #Encuentra la fecha del breakdown
                            vb1 = julian(T.values[k]) 
                            print(vb1, year)
                            dic[str(cont)][cuenta] = vb1 #Agrega al array
                            cuenta +=1
                            break
                cont +=1
                print(cont,model)
    return dic

models = ['HadGEM3-GC31-MM']#['KACE-1-0-G','TaiESM1','IITM-ESM','CMCC-CM2-SR5','IPSL-CM6A-LR']
#Genero un diccionario por modelo 
def ensamble_series(datos:dict, model:str):
    """Genera un DataFrame con las series temporales
    del ensamble generado para el modelo. 
    N : # de miembros historical
    M : # de miembros ssp585
    N*M + 1 columnas

    Args:
        data (dict): Diccionario con los miembros
        model (str): Nombre del modelo

    Returns:
        DataFrame: Pandas DataFrame con las n series temporales
    """
    n = len(datos['historical'][model])
    m = len(datos['ssp585'][model])
    dic = {}; cont = 0
    for j in range(m):
        var_ssp = xr.open_dataset(datos['ssp585'][model][j])
        var_ssp = var_ssp.sel(plev=slice(5100, 4900))
        var_s = var_ssp.ua
        var_s.attrs = var_ssp.ua.attrs
        var_s = var_s.isel(plev=0)
        var_su = var_s.sel(lat=slice(-60,-50))
        for i in range(n):
                dic[str(cont)] = np.zeros(len(anos)); cuenta = 0
                var_hist = xr.open_dataset(datos['historical'][model][i])
                var_hist = var_hist.sel(plev=slice(5100,4900))
                var_h = var_hist.ua
                var_h.attrs = var_hist.ua.attrs
                var_h = var_h.isel(plev=0)
                var_hu = var_h.sel(lat=slice(-60,-50))
                for year in years:
                    a = str(year)+'-10'
                    b = str(year)+'-12'
                    ua = var_hu.sel(time=slice(a, b))
                    ua = ua.mean(dim='lat')
                    ua = ua.mean(dim='lon')
                    T = ua.time
                    for k in range(len(ua)):
                        if not ua[k] > 19:  #Encuentra la fecha del breakdown
                            vb1 = julian(T.values[k]) 
                            print(vb1, year)
                            dic[str(cont)][cuenta] = vb1 #Agrega al array
                            cuenta +=1
                            break
                for year in years_futuro:
                    a = str(year)+'-10'
                    b = str(year+1)+'-02'
                    ua = var_su.sel(time=slice(a, b))
                    ua = ua.mean(dim='lat')
                    ua = ua.mean(dim='lon')
                    T = ua.time
                    for k in range(len(ua)):
                        if not ua[k] > 19:  #Encuentra la fecha del breakdown
                            vb1 = julian(T.values[k]) 
                            print(vb1, year)
                            dic[str(cont)][cuenta] = vb1 #Agrega al array
                            cuenta +=1
                            break
                cont +=1
                print(cont,model)
    return dic


dic_series = {}
for model in models:
    dic_series[model] = ensamble_series(dato_ua,model)


#Maneja los dias para que queden por encima de 300, para tener una serie consistente
def convert_day(a):
    for i in range(len(a)):
        if a[i] < 200:
            a[i] = a[i] + 365
        else:
            a[i] = a[i]
    n = len(a); m = 149 - n 
    media = np.mean(a[-4:-1])
    rand = [random.randint() for _ in range(m)] + media
    a = np.append(a,rand)
    return a 

#dic_conv = {}
#def convert_model(model_dic):
#    for model in models:
#        dic_conv[model] = convert_day(model_dic[model])
#        ref_bd_date = np.mean(dic_conv[model][:20])
#        dic_conv[model] = dic_conv[model] - ref_bd_date

#Guardamos los datos en DataFrames

anos = np.append(years,years_futuro)

VB = {'Year': anos,models[2]+'_1':dic_series[models[2]]} 
print(VB)
VB = pd.DataFrame(VB)
VB.to_csv('/home/julia.mindlin/Trabajo/CMIP6_ozone/VB_timeseries_'+models[0]+'.csv', float_format='%g')

VB = {'Year': anos,models[1]+'_1':dic_series[models[1]]['0'],
    models[1]+'_2':dic_series[models[1]]['1'],
    models[1]+'_3':dic_series[models[1]]['2'],
    models[1]+'_4':dic_series[models[1]]['6'],
    models[1]+'_5':dic_series[models[1]]['7'],
    models[1]+'_6':dic_series[models[1]]['8'],
    models[1]+'_10':dic_series[models[1]]['9'],
    models[1]+'_11':dic_series[models[1]]['10'],
    models[1]+'_12':dic_series[models[1]]['11'],
    models[1]+'_13':dic_series[models[1]]['12'],
    models[1]+'_14':dic_series[models[1]]['13'],
    models[1]+'_15':dic_series[models[1]]['14'],
    models[1]+'_16':dic_series[models[1]]['15'],
    models[1]+'_17':dic_series[models[1]]['16'],
    models[1]+'_18':dic_series[models[1]]['17'],
    models[1]+'_19':dic_series[models[1]]['18'],
    models[1]+'_20':dic_series[models[1]]['19'],
    models[1]+'_21':dic_series[models[1]]['20'],
    models[1]+'_22':dic_series[models[1]]['21'],
    models[1]+'_23':dic_series[models[1]]['22'],
    models[1]+'_24':dic_series[models[1]]['23'],
    models[1]+'_25':dic_series[models[1]]['24'],
    models[1]+'_26':dic_series[models[1]]['25'],
    models[1]+'_27':dic_series[models[1]]['26'],
    models[1]+'_28':dic_series[models[1]]['27'],
    models[1]+'_29':dic_series[models[1]]['28'],
    models[1]+'_30':dic_series[models[1]]['29'],
    models[1]+'_31':dic_series[models[1]]['30'],
    models[1]+'_32':dic_series[models[1]]['31'],
    models[1]+'_33':dic_series[models[1]]['32'],
    models[1]+'_34':dic_series[models[1]]['33'],
    models[1]+'_35':dic_series[models[1]]['34'],
    models[1]+'_36':dic_series[models[1]]['35'],
    models[1]+'_37':dic_series[models[1]]['36'],
    models[1]+'_38':dic_series[models[1]]['37'],
    models[1]+'_39':dic_series[models[1]]['38'],
    models[1]+'_40':dic_series[models[1]]['39'],
    models[1]+'_41':dic_series[models[1]]['40'],
    models[1]+'_42':dic_series[models[1]]['41'],
    models[1]+'_43':dic_series[models[1]]['42'],
    models[1]+'_44':dic_series[models[1]]['43'],
    models[1]+'_45':dic_series[models[1]]['44'],
    models[1]+'_46':dic_series[models[1]]['45'],
    models[1]+'_47':dic_series[models[1]]['46'],
    models[1]+'_48':dic_series[models[1]]['47'],
    models[1]+'_49':dic_series[models[1]]['48'],
    models[1]+'_50':dic_series[models[1]]['49'],
    models[1]+'_51':dic_series[models[1]]['50'],
    models[1]+'_52':dic_series[models[1]]['51'],
    models[1]+'_53':dic_series[models[1]]['52'],
    models[1]+'_54':dic_series[models[1]]['53'],
    models[1]+'_55':dic_series[models[1]]['54'],
    models[1]+'_56':dic_series[models[1]]['55'],
    models[1]+'_57':dic_series[models[1]]['56'],
    models[1]+'_58':dic_series[models[1]]['57'],
    models[1]+'_59':dic_series[models[1]]['58'],
    models[1]+'_60':dic_series[models[1]]['59'],
    models[1]+'_61':dic_series[models[1]]['60'],
    models[1]+'_62':dic_series[models[1]]['61'],
    models[1]+'_63':dic_series[models[1]]['62'],
    models[1]+'_64':dic_series[models[1]]['63'],
    models[1]+'_65':dic_series[models[1]]['64'],
    models[1]+'_66':dic_series[models[1]]['65'],
    models[1]+'_67':dic_series[models[1]]['66'],
    models[1]+'_68':dic_series[models[1]]['67'],
    models[1]+'_69':dic_series[models[1]]['68'],
    models[1]+'_70':dic_series[models[1]]['69'],
    models[1]+'_71':dic_series[models[1]]['70'],
    models[1]+'_72':dic_series[models[1]]['71'],
    models[1]+'_73':dic_series[models[1]]['72'],
    models[1]+'_74':dic_series[models[1]]['73'],
    models[1]+'_75':dic_series[models[1]]['74'],
    models[1]+'_76':dic_series[models[1]]['75'],
    models[1]+'_77':dic_series[models[1]]['76'],
    models[1]+'_78':dic_series[models[1]]['77'],
    models[1]+'_79':dic_series[models[1]]['78'],
    models[1]+'_80':dic_series[models[1]]['79'],
    models[1]+'_81':dic_series[models[1]]['80'],
    models[1]+'_82':dic_series[models[1]]['81'],
    models[1]+'_83':dic_series[models[1]]['82'],
    models[1]+'_84':dic_series[models[1]]['83'],
    models[1]+'_85':dic_series[models[1]]['84'],
    models[1]+'_86':dic_series[models[1]]['85'],
    models[1]+'_87':dic_series[models[1]]['86'],
    models[1]+'_88':dic_series[models[1]]['87'],
    models[1]+'_89':dic_series[models[1]]['88'],
    models[1]+'_90':dic_series[models[1]]['89'],
    models[1]+'_91':dic_series[models[1]]['90'],
    models[1]+'_92':dic_series[models[1]]['91'],
    models[1]+'_93':dic_series[models[1]]['92'],
    models[1]+'_94':dic_series[models[1]]['93'],
    models[1]+'_95':dic_series[models[1]]['94'],
    models[1]+'_96':dic_series[models[1]]['95'],
    models[1]+'_97':dic_series[models[1]]['96'],
    models[1]+'_98':dic_series[models[1]]['97'],
    models[1]+'_99':dic_series[models[1]]['98'],
    models[1]+'_100':dic_series[models[1]]['99'],
    models[1]+'_101':dic_series[models[1]]['100'],
    models[1]+'_102':dic_series[models[1]]['101'],
    models[1]+'_103':dic_series[models[1]]['102'],
    models[1]+'_104':dic_series[models[1]]['103'],
    models[1]+'_105':dic_series[models[1]]['104'],
    models[1]+'_106':dic_series[models[1]]['105'],
    models[1]+'_107':dic_series[models[1]]['106'],
    models[1]+'_108':dic_series[models[1]]['107'],
    models[1]+'_109':dic_series[models[1]]['108'],
    models[1]+'_110':dic_series[models[1]]['109'],
    models[1]+'_111':dic_series[models[1]]['110'],
    models[1]+'_112':dic_series[models[1]]['111'],
    models[1]+'_113':dic_series[models[1]]['112'],
    models[1]+'_114':dic_series[models[1]]['113'],
    models[1]+'_115':dic_series[models[1]]['114'],
    models[1]+'_116':dic_series[models[1]]['115'],
    models[1]+'_117':dic_series[models[1]]['116'],
    models[1]+'_118':dic_series[models[1]]['117'],
    models[1]+'_119':dic_series[models[1]]['118'],
    models[1]+'_120':dic_series[models[1]]['119'],
    models[1]+'_121':dic_series[models[1]]['120'],
    models[1]+'_122':dic_series[models[1]]['121'],
    models[1]+'_123':dic_series[models[1]]['122'],
    models[1]+'_124':dic_series[models[1]]['123'],
    models[1]+'_125':dic_series[models[1]]['124'],
    models[1]+'_126':dic_series[models[1]]['125'],
    models[1]+'_127':dic_series[models[1]]['126'],
    models[1]+'_128':dic_series[models[1]]['127'],
    models[1]+'_129':dic_series[models[1]]['128'],
    models[1]+'_120':dic_series[models[1]]['129'],
    models[1]+'_131':dic_series[models[1]]['130'],
    models[1]+'_132':dic_series[models[1]]['131'],
    models[1]+'_133':dic_series[models[1]]['132'],
    models[1]+'_134':dic_series[models[1]]['133'],
    models[1]+'_135':dic_series[models[1]]['134'],
    models[1]+'_136':dic_series[models[1]]['135'],
    models[1]+'_137':dic_series[models[1]]['136'],
    models[1]+'_138':dic_series[models[1]]['137'],
    models[1]+'_139':dic_series[models[1]]['138'],
    models[1]+'_140':dic_series[models[1]]['139'],
    models[1]+'_141':dic_series[models[1]]['140'],
    models[1]+'_142':dic_series[models[1]]['141'],
    models[1]+'_143':dic_series[models[1]]['142'],
    models[1]+'_144':dic_series[models[1]]['143'],
    models[1]+'_145':dic_series[models[1]]['144'],
    models[1]+'_146':dic_series[models[1]]['145'],
    models[1]+'_147':dic_series[models[1]]['146'],
    models[1]+'_148':dic_series[models[1]]['147'],
    models[1]+'_149':dic_series[models[1]]['148'],
    models[1]+'_150':dic_series[models[1]]['149'],
    models[1]+'_151':dic_series[models[1]]['150'],
    models[1]+'_152':dic_series[models[1]]['151'],
    models[1]+'_153':dic_series[models[1]]['152'],
    models[1]+'_154':dic_series[models[1]]['153'],
    models[1]+'_155':dic_series[models[1]]['154'],
    models[1]+'_156':dic_series[models[1]]['155'],
    models[1]+'_157':dic_series[models[1]]['156'],
    models[1]+'_158':dic_series[models[1]]['157'],
    models[1]+'_159':dic_series[models[1]]['158'],
    models[1]+'_160':dic_series[models[1]]['159'],
    models[1]+'_161':dic_series[models[1]]['160'],
    models[1]+'_162':dic_series[models[1]]['161'],
    models[1]+'_163':dic_series[models[1]]['162'],
    models[1]+'_164':dic_series[models[1]]['163'],
    models[1]+'_165':dic_series[models[1]]['164'],
    models[1]+'_166':dic_series[models[1]]['165'],
    models[1]+'_167':dic_series[models[1]]['166'],
    models[1]+'_168':dic_series[models[1]]['167'],
    models[1]+'_168':dic_series[models[1]]['168'],
    models[1]+'_170':dic_series[models[1]]['169'],
    models[1]+'_171':dic_series[models[1]]['170'],
    models[1]+'_172':dic_series[models[1]]['171'],
    models[1]+'_173':dic_series[models[1]]['172'],
    models[1]+'_174':dic_series[models[1]]['173'],
    models[1]+'_175':dic_series[models[1]]['174'],
    models[1]+'_176':dic_series[models[1]]['175'],
    models[1]+'_177':dic_series[models[1]]['176'],
    models[1]+'_178':dic_series[models[1]]['177'],
    models[1]+'_179':dic_series[models[1]]['178'],
    models[1]+'_180':dic_series[models[1]]['179'],
    models[1]+'_181':dic_series[models[1]]['180'],
    models[1]+'_182':dic_series[models[1]]['181'],
    models[1]+'_183':dic_series[models[1]]['182'],
    models[1]+'_184':dic_series[models[1]]['183'],
    models[1]+'_185':dic_series[models[1]]['184'],
    models[1]+'_186':dic_series[models[1]]['185'],
    models[1]+'_187':dic_series[models[1]]['186'],
    models[1]+'_188':dic_series[models[1]]['187'],
    models[1]+'_189':dic_series[models[1]]['188'],
    models[1]+'_190':dic_series[models[1]]['189'],
    models[1]+'_191':dic_series[models[1]]['190'],
    models[1]+'_192':dic_series[models[1]]['191']}
,
    models[1]+'_193':dic_series[models[1]]['192'],
    models[1]+'_194':dic_series[models[1]]['193'],
    models[1]+'_195':dic_series[models[1]]['194'],
    models[1]+'_196':dic_series[models[1]]['195'],
    models[1]+'_197':dic_series[models[1]]['196'],
    models[1]+'_198':dic_series[models[1]]['197'],
    models[1]+'_199':dic_series[models[1]]['198'],
    models[1]+'_200':dic_series[models[1]]['199'],
    models[1]+'_201':dic_series[models[1]]['200'],
    models[1]+'_202':dic_series[models[1]]['201'],
    models[1]+'_203':dic_series[models[1]]['202'],
    models[1]+'_204':dic_series[models[1]]['203'],
    models[1]+'_205':dic_series[models[1]]['204'],
    models[1]+'_206':dic_series[models[1]]['205'],
    models[1]+'_207':dic_series[models[1]]['206'],
    models[1]+'_208':dic_series[models[1]]['207'],
    models[1]+'_209':dic_series[models[1]]['208'],
    models[1]+'_210':dic_series[models[1]]['209'],
    models[1]+'_211':dic_series[models[1]]['210'],
    models[1]+'_212':dic_series[models[1]]['211'],
    models[1]+'_213':dic_series[models[1]]['212'],
    models[1]+'_214':dic_series[models[1]]['213'],
    models[1]+'_215':dic_series[models[1]]['214'],
    models[1]+'_216':dic_series[models[1]]['215'],
    models[1]+'_217':dic_series[models[1]]['216'],
    models[1]+'_218':dic_series[models[1]]['217'],
    models[1]+'_219':dic_series[models[1]]['218'],
    models[1]+'_220':dic_series[models[1]]['219'],
    models[1]+'_221':dic_series[models[1]]['220'],
    models[1]+'_222':dic_series[models[1]]['221'],
    models[1]+'_223':dic_series[models[1]]['222'],
    models[1]+'_224':dic_series[models[1]]['223'],
    models[1]+'_225':dic_series[models[1]]['224'],
    models[1]+'_226':dic_series[models[1]]['225'],
    models[1]+'_227':dic_series[models[1]]['226'],
    models[1]+'_228':dic_series[models[1]]['227'],
    models[1]+'_229':dic_series[models[1]]['228'],
    models[1]+'_230':dic_series[models[1]]['229'],
    models[1]+'_231':dic_series[models[1]]['230'],
    models[1]+'_231':dic_series[models[1]]['231'],
    models[1]+'_233':dic_series[models[1]]['232'],
    models[1]+'_234':dic_series[models[1]]['233'],
    models[1]+'_235':dic_series[models[1]]['234'],
    models[1]+'_236':dic_series[models[1]]['235'],
    models[1]+'_237':dic_series[models[1]]['236'],
    models[1]+'_238':dic_series[models[1]]['237'],
    models[1]+'_239':dic_series[models[1]]['238'],
    models[1]+'_240':dic_series[models[1]]['239'],
    models[1]+'_241':dic_series[models[1]]['240'],
    models[1]+'_242':dic_series[models[1]]['241'],
    models[1]+'_243':dic_series[models[1]]['242'],
    models[1]+'_244':dic_series[models[1]]['243'],
    models[1]+'_245':dic_series[models[1]]['244'],
    models[1]+'_246':dic_series[models[1]]['245'],
    models[1]+'_247':dic_series[models[1]]['246'],
    models[1]+'_248':dic_series[models[1]]['247'],
    models[1]+'_249':dic_series[models[1]]['248'],
    models[1]+'_250':dic_series[models[1]]['249'],
    models[1]+'_251':dic_series[models[1]]['250'],
    models[1]+'_252':dic_series[models[1]]['251'],
    models[1]+'_253':dic_series[models[1]]['252'],
    models[1]+'_254':dic_series[models[1]]['253'],
    models[1]+'_255':dic_series[models[1]]['254'],
    models[1]+'_256':dic_series[models[1]]['255'],
    models[1]+'_257':dic_series[models[1]]['256'],
    models[1]+'_258':dic_series[models[1]]['257'],
    models[1]+'_259':dic_series[models[1]]['258'],
    models[1]+'_260':dic_series[models[1]]['259'],
    models[1]+'_261':dic_series[models[1]]['260'],
    models[1]+'_262':dic_series[models[1]]['261'],
    models[1]+'_263':dic_series[models[1]]['262'],
    models[1]+'_264':dic_series[models[1]]['263'],
    models[1]+'_265':dic_series[models[1]]['264'],
    models[1]+'_266':dic_series[models[1]]['265'],
    models[1]+'_267':dic_series[models[1]]['266'],
    models[1]+'_268':dic_series[models[1]]['267'],
    models[1]+'_269':dic_series[models[1]]['268'],
    models[1]+'_270':dic_series[models[1]]['269']
    }
print(VB)
VB = pd.DataFrame(VB)
VB.insert(0,"Years", anos,True)
VB.to_csv('/home/julia.mindlin/Trabajo/CMIP6_ozone/VB_timeseries_'+models[2]+'.csv', float_format='%g')


array([321., 321., 311., 321., 311., 331., 321., 331., 331., 321., 331.,
       331., 331., 331., 321., 331., 341., 331., 321., 311., 331., 331.,
       321., 341., 341., 331., 331., 341., 331., 341., 331., 341., 341.,
       331., 341., 341., 341., 331., 341., 351., 321., 331., 341., 331.,
       331., 351., 331., 331., 341., 341., 351., 321., 351., 351., 321.,
       351., 351., 331., 341., 331., 341., 351., 351.,   1.,  11.,   1.,
         1., 331., 351.,   1., 341., 331., 351., 351., 351.,   1., 351.,
       351.,   1., 341., 341., 351., 351., 351.,   1., 351., 351., 351.,
       351.,   1., 351.,   1., 341.,   1., 341., 351., 341.,   1., 351.,
       351.,   1., 351., 351.,   1., 341., 341., 351., 351., 351., 351.,
         1., 351.,   1., 351.,   1.,   1., 351.,  11., 351.,  11., 351.,
         1.,  31., 351.,   1.,  21.,   1., 351.,  11.,   1.,  41.,   1.,
       351., 341., 351., 351.,   1.,  21., 351.,   23.,   345.,   1.,   11.,
         1.,   363.,   2.,   23.,   12.,   360.])
