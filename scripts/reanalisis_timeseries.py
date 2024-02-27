#Calculo los mismos indices que para CMIP6 para ERA5: VBdelay 

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


ruta = '/datos/julia.mindlin/ERA5'
ua_dato = xr.open_dataset(ruta+'/era5.u_component_of_wind.day_T42.nc')
ta_dato = xr.open_dataset(ruta+'/era5.mon.mean_T42.nc')

#Funciones

def julian(date):
    """Calcula el dia juliano a partir de una fecha

    Args:
        date ([type]): fecha varia el tipo segun el calendario

    Returns:
        int: dia juliano
    """
    try:
        yday = datetime.utcfromtimestamp(date.tolist()/1e9).timetuple().tm_yday
    except AttributeError or TypeError:
        yday = date.timetuple().tm_yday
    return yday

years = np.arange(1979,2019,1)
tas_index = np.array([])
ta_sup = ta_dato.t
ta_sup.attrs = ta_dato.t.attrs
ta_sup = ta_sup.isel(lev=36)
lats = np.cos(ta_sup.lat.values*np.pi/180)
s = sum(lats)
ta_sup = ta_sup.mean(dim='lon')
ta_sup = ta_sup.fillna(ta_sup[-1]-1)
cont = 0
for year in years:
    a = str(year)+'-12'
    b = str(year+1)+'-02'
    tas = ta_sup.sel(time=slice(a,b))
    tas = tas.mean(dim='time')
    TAS = sum(tas*lats)/s
    print(year,TAS)
    tas_index = np.append(tas_index,TAS)

a = 10
TAS = {'Year': years, 'GW': tas_index}

ua_50 = ua_dato.sel(expver=1)
ua_50 = ua_50.u
ua_50.attrs = ua_dato.u.attrs
ua_50 = ua_50.sel(lat=slice(-50,-60))
ua_50 = ua_50.mean(dim='lat')
ua_50 = ua_50.mean(dim='lon')
VB_date = np.array([])
for year in years:
    a = str(year)+'-10'
    b = str(year)+'-12'
    ua = ua_50.sel(time=slice(a,b))
    T = ua.time
    print(ua,T)
    for i in range(len(ua)):
        if not (ua[i] > 19): #Encuentra la fecha del breakdown
            vb1 = julian(T.values[i])
            print(vb1,year)
            VB_date = np.append(VB_date,vb1) #Agrega al array
            break

def convert_day(a):
        for i in range(len(a)):
                if a[i] < 200:
                        a[i] = a[i] + 365
                else:
                        a[i] = a[i]
        return a


VB = {'Year': years,'ERA5':VB_date}
VB = pd.DataFrame(VB)
VB.to_csv('/home/julia.mindlin/Trabajo/CMIP6_ozone/reanalisis_VB_timeseries.csv', float_format='%g')

TAS = pd.DataFrame(TAS)
TAS.to_csv('/home/julia.mindlin/Trabajo/CMIP6_ozone/reanalisis_GW_timeseries.csv', float_format='%g')


