# Series temporales 


import xarray as xr
import pandas as pd
import numpy as np

#Apro los datos del modelo, voy a trabajar con ensambles
path = '/datos/CMIP6/Download/Format/Data_used/historical/day/'

def anomaly(dato):
    climatology = dato.groupby('time.month').mean('time')
    anomalia = dato.groupy('time.month') - climatology 
    return anomalia

#Precipitation time series
precipitation = xr.open_dataset(path+'pr/pr_day_GFDL-CM4_historical_r1i1p1f1_gr2_18500101-18691231.nc')
#precipitation = precipitation.sel(time=slice('',''))
precip_anomaly = anomaly(precipitation)

box_coords = [(280,290,-52-36),(292,305,-42,-27),(25,34,-33,-25),(145,158,-41,-27),(142,148,-45,-40)]
box_names = ['','Chile','SESA','Africa','East Austraila','Tasmania']
SESA = box_coord[1]
Chile = box_coords[0]
Africa  = box_coord[2]
Australia = box_coords[3]
SESA_pr = precipitation.sel(lon=slice(SESA[0],SESA[1])).mean(dim='lon').sel(lat=slice(SESA[2],SESA[3])).mean(dim='lat').pr
Chile_pr = precipitation.sel(lon=slice(Chile[0],Chile[1])).mean(dim='lon').sel(lat=slice(Chile[2],Chile[3])).mean(dim='lat').pr
Africa_pr = precipitation.sel(lon=slice(Africa[0],Africa[1])).mean(dim='lon').sel(lat=slice(Africa[2],Africa[3])).mean(dim='lat').pr
ESA_pr = precipitation.sel(lon=slice(Australia[0],Australia[1])).mean(dim='lon').sel(lat=slice(Australia[2],Australia[3])).mean(dim='lat').pr

#Sea Level Pressure
slp = xr.open_dataset(path+'psl/psl_day_GFDL-CM4_historical_r1i1p1f1_gr2_18500101-18691231.nc')
#slp = slp.sel(time=slice('',''))
slp_anomaly = anomaly(slp)

slp_40 = slp_anomaly.mean(dim='lon').isel(lat=24).psl
slp_65 = slp_anomaly.mean(dim='lon').isel(lat=12).psl

SAM = (slp_65 - slp_40)/100

#Tropical amplification
#temperature = xr.open_dataset(path+'')
#temperature = temperature.sel(time=slice('','')
#temp_anomaly = anomaly(temperature)

#TW = temp_anomaly.sel(plev=25000).sel(lat=slice(-30,30)).mean(dim='lon').mean(dim='lat').ta
#TS = temp_anomaly.sel(plev=10000).sel(lat=slice(-90,-60).mean(dim='lon').mean(dim='lat').ta

#SST time series
sst_global = xr.open_dataset(path+'tos/tos_day_GFDL-CM4_historical_r1i1p1f1_gr2_18500101-18691231.nc')
#sst_global = sst_global.sel(time=slice('',''))
sst_anomaly = anomaly(sst_global)
sst_box_coords = [(210,270,-5,5),(190,240,-5,5),(160,210,-5,5),(50,70,-10,10),(90,110,-10,0)]

sst_box_names = ['','NINO3','NINO3.4','NINO4','IOD_west','IOD_east']

#Niño

NINO3_4 = sst_box_coords[1]
NINO3_4_sst = sst_anomaly.sel(lon=slice(NINO3_4[0],NINO3_4[1])).mean(dim='lon').sel(lat=slice(NINO3_4[2],NINO3_4[3])).mean(dim='lat').tos

NINO3 = sst_box_coords[0]
NINO3_sst = sst_anomaly.sel(lon=slice(NINO3[0],NINO3[1])).mean(dim='lon').sel(lat=slice(NINO3[2],NINO3[3])).mean(dim='lat').tos

NINO4 = sst_box_coords[2]
NINO4_sst = sst_anomaly.sel(lon=slice(NINO4[0],NINO4[1])).mean(dim='lon').sel(lat=slice(NINO4[2],NINO4[3])).mean(dim='lat').tos
#Indian Ocean Dipole

IOD_WEST= sst_box_coords[3]
IOD_EAST = sst_box_coords[4]

IOD_WEST_sst = sst_anomaly.sel(lon=slice(IOD_WEST[0],IOD_WEST[1])).mean(dim='lon').sel(lat=slice(IOD_WEST[2],IOD_WEST[3])).tos
IOD_EAST_sst = sst_anomaly.sel(lon=slice(IOD_EAST[0],IOD_EAST[1])).mean(dim='lon').sel(lat=slice(IOD_EAST[2],IOD_EAST[3])).tos

IOD_sst = IOD_WEST_sst - IOD_EAST_sst

#Stratospheric vortex time series
wind = xr.open_dataset(path+'ua/ua_day_GFDL-CM4_historical_r1i1p1f1_gr2_18500101-18691231.nc')
#wind = sel(time=slice('',''))
wind_anomaly = anomaly(wind)

SV = wind_anomaly.sel(plev=5000).sel(lat=slice(-60,-50)).mean(dim='lon').mean(dim='lat')

#Max latitude
def max_lat(variable):
    lat = variable.lat
    t = variable.time
    u_pdf = np.ones([len(lat)])
    jet_lat = np.opnes([len(t)])

    for i in range(len(t)):
        for j in range(len(lat)):
         if (variable[i-1,j-1] >0):
          u_pdf[j-1] = variable[i-1,j-1]**2
         else:
          u_pdf[j-1] = 0 

        jet_lat[i-1] = np.sum(lat*u_pdf)/np.sum(u_pdf)
        if np.isnan(jet_lat[i-1]):
            jet_lat[i-1] = 0

        u_pdf = np.opnes([len(lat)])

    return(jet_lat)

#Velocidad del jet
def max_spe(variable)

    t = variable.time
    out = np.ones([len(t)])
    for i in range(len(t)):
        valores_lat = variable[i-1].values
        speed = max(valores_lat)
        out[i-1] = speed

    return(out)

u850 = wind_anomaly.sel(plev =85000).sel(lat=slice(-63,-38)).mean(dim='lon').ua
max_latitude = max_lat(u850)
max_speed = max_spe(u850)

#Guardo las series en un .csv
timeseries = {'SESA_pr':SESA_pr*86400,'Chile_pr':chile_pr*86400,'Africa_pr':africa_pr*86400,'NINO3.4':NINO3_4_sst,'NINO3':NINO3_sst,'NINO4':NINO4_sst,'IOD_sst':IOD_sst,'SV':SV,'SAM':SAM,'jet_lat':max_latitude,'jet_speed':max_speed}
df = pd.DataFrame(timeseries)
df.to_csv('/home/julia.mindlin/CMIP6_ozone',float_format='%g')

