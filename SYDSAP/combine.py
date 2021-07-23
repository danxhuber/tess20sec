import numpy as np
import os, sys
from astropy.io import ascii, fits
import matplotlib.pyplot as plt

dat1=ascii.read('piMen_S27_20s_SYDSAP.csv')
dat2=ascii.read('piMen_S28_20s_SYDSAP.csv',delimiter=',')
#dat3=ascii.read('piMen_S31_20s_SYDSAP.csv')

time=np.concatenate((dat1['col1'],dat2['col1']))
flux=np.concatenate((dat1['col3'],dat2['col3']/np.median(dat2['col3'])))
flux_err=np.concatenate((dat1['col4'],dat2['col4']/np.median(dat2['col3'])))

plt.ion()
plt.clf()
plt.plot(time,flux,'.')

ascii.write([time,time,flux,flux_err],'piMen_20s_S272831_SYDSAP.csv',names=['time','time2','flux','flux_err'],delimiter=',')


old=ascii.read('piMen_20s_SYDSAP.csv')

plt.clf()
plt.plot(old['time'],old['flux'],'.')


plt.clf()

dat1=ascii.read('piMen_S28_120s_SYDSAP.csv')
dat2=ascii.read('piMen_newsectors/piMen_S28_120s_SYDSAP.csv')

plt.clf()
plt.ion()
plt.plot(dat1['time'],dat1['flux']/np.median(dat1['flux']),'.')
plt.plot(dat2['time'],dat2['flux']/np.median(dat2['flux']),'.')