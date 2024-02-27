#Modelo lineal de VB basado en GW y EESC
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import math
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import os, fnmatch
import glob


GW = pd.read_csv('/home/julia.mindlin/Trabajo/CMIP6_ozone/GW_timeseries.csv')
VB = pd.read_csv('/home/julia.mindlin/Trabajo/CMIP6_ozone/VB_timeseries.csv')
EESC_pol = pd.read_csv('/home/julia.mindlin/Trabajo/CMIP6_ozone/GW_EESC_polar_ozoneloss.csv')
Years = EESC_pol['Years']
EESC = EESC_pol['EESC_polar']
models = ['CNRM-CM6-1','CNRM-ESM2-1','MIROC6','MIROC-ES2L','MPI-ESM1-2-HR','MPI-ESM1-2-LR','MRI-ESM2-0','NorESM2-MM']


def regresion_modelo(modelo):
        regresores = pd.DataFrame({'GW':GW[modelo][:149],'EESC':EESC_pol['EESC_polar'][:149]})
        #y = sm.add_constant(regresores.values)
        reg = linear_model.LinearRegression()
        x = VB[modelo].values
        res = sm.OLS(x,regresores).fit()
        a = res.params[0]
        b = res.params[1]
        r2 = res.rsquared
        c_i = res.conf_int(alpha=0.05,cols=None)
        return a, b, r2, c_i         
 

a, b, r2, c_i = regresion_modelo(models[0]) 

dic = {}
for model in models:
        dic[model] = []
        a,b,r2,c_i = regresion_modelo(model)
        dic[model].append(a) , dic[model].append(b),
        dic[model].append(r2) , dic[model].append(c_i)


def modelo_lineal(modelo):
	VB_modelado_ci_low = GW[modelo]*dic[modelo][3][0]['GW'] + EESC*dic[modelo][3][0]['EESC']
	VB_modelado_ci_high = GW[modelo]*dic[modelo][3][1]['GW'] + EESC*dic[modelo][3][1]['EESC']
	VB_modelado = GW[modelo]*dic[modelo][0] + EESC*dic[modelo][1]
	return VB_modelado_ci_low, VB_modelado, VB_modelado_ci_high
 

plt.clf()
plt.close('all')
fig = plt.figure(figsize=(40,10))
fig.subplots_adjust(hspace = .9, wspace = .2)
 
for k in range(len(models)):
         VB_modelado_ci_low, VB_modelado, VB_modelado_ci_high = modelo_lineal(models[k])
         ax = fig.add_subplot(2,8,k+1)
         ax.plot(Years[:149],VB[models[k]][:149],label=str(models[k]))
         ax.plot(Years[:149],VB_modelado[:149],label='statistical model for'+str(models[k]))
         ax.fill_between(Years[:149],VB_modelado_ci_low[:149],VB_modelado_ci_high[:149],alpha=0.5)
         ax.set_xlabel('Years',fontsize=15)
         if k == 0:
                 ax.set_ylabel('Vortex Breakdown Date [Julian day]',fontsize=14)
         ax.set_yticks(np.arange(280,360,10))
         ax.set_xticks(np.arange(1950,2130,30))
         ax.tick_params(axis='both', labelsize=14)
         ax.legend()
         ax1 = fig.add_subplot(2,8,k+9)
         ax1.plot(Years[:149],GW[models[k]][:149],label=str(models[k]))
         ax1.set_xlabel('Years',fontsize=15)
         if k == 0 :
                 ax1.set_ylabel('Surface Temperature [K]',fontsize=14)
         ax1.set_yticks(np.arange(290,310,5))
         ax1.set_xticks(np.arange(1950,2130,30))
         ax1.tick_params(axis='both', labelsize=14)
         ax1.legend()
 
fig.savefig('/home/julia.mindlin/Trabajo/CMIP6_ozone/modelos_lineales3.png')


