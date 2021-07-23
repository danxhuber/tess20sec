import numpy as np
import os, sys, pdb
from astropy.io import ascii, fits
import matplotlib.pyplot as plt
from echelle import interact_echelle 
from echelle import plot_echelle
from echelle import echelle as calc_echelle
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit

plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.major.width'] = 3
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['font.size']=16
plt.rcParams['mathtext.default']='regular'
plt.rcParams['lines.markersize']=8
plt.rcParams['xtick.major.pad']='6'
plt.rcParams['ytick.major.pad']='8'
plt.rcParams['ytick.minor.visible'] = 'True'
plt.rcParams['xtick.minor.visible'] = 'True'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['ytick.right'] = 'True'
plt.rcParams['xtick.top'] = 'True'

def func(x, a, b):
    return a + b*x

#dnu_pimen=116.8
#dnu_gampav=120.0
#dnu_zetatuc=126.0

star='pimen'
#star='gammapav'
#star='zetatuc'

print(star)

if (star == 'gammapav'):
	# gamma Pav = TIC265488188
	tit='gamma Pav = TIC265488188'
	dat=ascii.read('265488188_bgcorr.txt')
	freqs=ascii.read('gammapav-freq.csv')
	usel=0
	lol=2000
	upl=3250

if (star == 'zetatuc'):
	# zeta Tuc = TIC425935521
	tit='zeta Tuc = TIC425935521'
	dat=ascii.read('425935521_bgcorr.txt')
	freqs=ascii.read('zetatuc-freq.csv')
	usel=0
	lol=2200
	upl=3400

if (star == 'pimen'):
	# pi Men = TIC261136679
	tit='pi Men = TIC261136679'
	dat=ascii.read('261136679_bgcorr.txt')
	dat=ascii.read('../data/processed/piMen_new/piMen_20sec_PS.txt')
	freqs=ascii.read('pimen-freq.csv')
	usel=0
	lol=2200
	upl=3000

um=np.where(freqs['l'] == usel)[0]

popt, pcov = curve_fit(func, np.arange(len(um)),freqs['freq'][um], sigma=1./(freqs['err'][um]**2))
#f=np.polyfit(np.arange(len(um)),freqs['freq'][um],1)
dnu=popt[1]
print('dnu:',dnu)

freq=np.array(dat['col1'])
amp=np.array(dat['col2'])
gauss_kernel = Gaussian1DKernel(2)
pssm = convolve(amp, gauss_kernel)
echx, echy, echz=calc_echelle(freq,pssm, dnu,fmin=1000, fmax=5000)


alf=0.5

plt.ion()
plt.clf()

plt.subplot(2,1,1)
plt.plot(freq,amp,color='grey')
plt.plot(freq,pssm,color='black',lw=2)
plt.xlim([lol,upl])

for i in range(0,len(freqs)):
    if (freqs['l'][i] == 0):
    	plt.axvspan(freqs['freq'][i]-freqs['err'][i],freqs['freq'][i]+freqs['err'][i],color='red',alpha=0.5)
    if (freqs['l'][i] == 1):
    	plt.axvspan(freqs['freq'][i]-freqs['err'][i],freqs['freq'][i]+freqs['err'][i],color='blue',alpha=0.5)
    if (freqs['l'][i] == 2):
    	plt.axvspan(freqs['freq'][i]-freqs['err'][i],freqs['freq'][i]+freqs['err'][i],color='green',alpha=0.5)

plt.title(tit)
plt.ylim([0,20])

plt.subplot(2,1,2)
plt.imshow(echz,aspect="auto",extent=(echx.min(), echx.max(), echy.min(), echy.max()),origin="lower",cmap='BuPu',interpolation='None')
um=np.where(freqs['l'] == 0)[0]
plt.errorbar(freqs['freq'][um] % dnu,freqs['freq'][um],xerr=freqs['err'][um],fmt='o',color='red',fillstyle='none',lw=2)
um=np.where(freqs['l'] == 1)[0]
plt.errorbar(freqs['freq'][um] % dnu,freqs['freq'][um],xerr=freqs['err'][um],fmt='o',color='blue',fillstyle='none',lw=2)
um=np.where(freqs['l'] == 2)[0]
plt.errorbar(freqs['freq'][um] % dnu,freqs['freq'][um],xerr=freqs['err'][um],fmt='o',color='green',fillstyle='none',lw=2)

plt.ylim([lol,upl])
plt.xlabel('Frequency mod '+str(dnu)[0:6]+' $\mu$Hz')
plt.tight_layout()



#plt.savefig(star+'-final.png',dpi=150)












