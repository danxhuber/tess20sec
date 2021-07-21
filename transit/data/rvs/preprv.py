import numpy as np
import pandas as pd
from astropy.io import ascii
import bin
import matplotlib.pyplot as plt
import pdb

aat = ascii.read(
    "https://exoplanetarchive.ipac.caltech.edu/data/ExoData/0026/0026394/data/UID_0026394_RVC_001.tbl"
)
binx_aat,biny_aat,erry_aat=bin.bin_time_err(aat['JD'],aat['Radial_Velocity'],aat['Radial_Velocity_Uncertainty'],1.)

harps = pd.read_csv(
    "https://raw.githubusercontent.com/exoplanet-dev/case-studies/main/data/pi_men_harps_rvs.csv",
    skiprows=1,
)
harps = harps.rename(lambda x: x.strip().strip("#"), axis=1)
harps_post = np.array(harps.date > "2015-07-01", dtype=int)

binx_harps,biny_harps,erry_harps=bin.bin_time_err(harps["bjd"],harps["rv"],harps["e_rv"],1.)
harps_pre = np.where(binx_harps < 2457204.5)[0]
harps_post = np.where(binx_harps > 2457204.5)[0]

#plt.clf()
#plt.plot(harps["bjd"],harps["rv"],'.')
#plt.plot(binx_harps,biny_harps,'.')

espresso=ascii.read('espresso.tsv')
breakp=2458375.-2450000.
um1=np.where(espresso['Time'] <= breakp)[0]
um2=np.where(espresso['Time'] > breakp)[0]
print(len(um1),len(um2))
binx_espresso_1,biny_espresso_1,erry_espresso_1=bin.bin_time_err(espresso['Time'][um1],
	espresso['RV'][um1],espresso['e_RV'][um1],1.)
binx_espresso_2,biny_espresso_2,erry_espresso_2=bin.bin_time_err(espresso['Time'][um2],
	espresso['RV'][um2],espresso['e_RV'][um2],1.)	
	

#plt.clf()
#plt.errorbar(espresso['Time'],espresso['RV'],yerr=espresso['e_RV'],fmt='.')
#plt.errorbar(binx,biny,yerr=erry,fmt='o')

coralie=ascii.read('coralie.tsv')
um=np.where(coralie['Dataset'] == 'CORALIE-98')[0]
binx_coralie_98,biny_coralie_98,erry_coralie_98=bin.bin_time_err(coralie['Time'][um],
	coralie['RV'][um],coralie['e_RV'][um],1.)
um=np.where(coralie['Dataset'] == 'CORALIE-14')[0]
binx_coralie_14,biny_coralie_14,erry_coralie_14=bin.bin_time_err(coralie['Time'][um],
	coralie['RV'][um],coralie['e_RV'][um],1.)
um=np.where(coralie['Dataset'] == 'CORALIE-07')[0]
binx_coralie_07,biny_coralie_07,erry_coralie_07=bin.bin_time_err(coralie['Time'][um],
	coralie['RV'][um],coralie['e_RV'][um],1.)

'''
plt.clf()
plt.ion()
plt.plot(binx_coralie_98,biny_coralie_98,'o')
plt.plot(binx_coralie_07,biny_coralie_07,'o')
plt.plot(binx_coralie_14,biny_coralie_14,'o')
'''

t = np.concatenate((binx_aat, binx_harps, binx_coralie_98+2450000, binx_coralie_07+2450000, 
	binx_coralie_14+2450000, binx_espresso_1+2450000, binx_espresso_2+2450000))
rv = np.concatenate((biny_aat, biny_harps, biny_coralie_98, biny_coralie_07, 
	biny_coralie_14, biny_espresso_1, biny_espresso_2))
rv_err = np.concatenate((erry_aat, erry_harps, erry_coralie_98, erry_coralie_07, 
	erry_coralie_14, erry_espresso_1 , erry_espresso_2))
inst_id = np.concatenate((np.zeros(len(binx_aat), dtype=int), 
	np.zeros(len(binx_harps[harps_pre]), dtype=int)+1, 
	np.zeros(len(binx_harps[harps_post]), dtype=int)+2, 
	np.zeros(len(binx_coralie_98), dtype=int)+3,
	np.zeros(len(binx_coralie_07), dtype=int)+4,
	np.zeros(len(binx_coralie_14), dtype=int)+5,
	np.zeros(len(binx_espresso_1), dtype=int)+6,
	np.zeros(len(binx_espresso_2), dtype=int)+7))

inds = np.argsort(t)
t = np.ascontiguousarray(t[inds], dtype=float)
rv = np.ascontiguousarray(rv[inds], dtype=float)
rv_err = np.ascontiguousarray(rv_err[inds], dtype=float)
inst_id = np.ascontiguousarray(inst_id[inds], dtype=int)

inst_names = ["aat", "harps_pre", "harps_post", "coralie_98", "coralie_07", 
	"coralie_14", "espresso_1", "espresso_2"]
num_inst = len(inst_names)


for i, name in enumerate(inst_names):
    m = inst_id == i
    plt.errorbar(
        t[m], rv[m] - np.min(rv[m]), yerr=rv_err[m], fmt=".", label=name
    )

plt.legend(fontsize=10)
plt.xlabel("BJD")
_ = plt.ylabel("radial velocity [m/s]")

ascii.write([t,rv,rv_err,inst_id],'pimen_rvs_binned.csv',delimiter=',',names=['t','rv','rv_err','inst_id'])