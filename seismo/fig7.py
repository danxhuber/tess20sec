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

plt.rcParams['xtick.major.size'] = 9
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['xtick.minor.size'] = 6
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.major.size'] = 9
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['ytick.minor.size'] = 6
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['font.size']=16
plt.rcParams['mathtext.default']='regular'
plt.rcParams['lines.markersize']=8
plt.rcParams['xtick.major.pad']='3'
plt.rcParams['ytick.major.pad']='3'
plt.rcParams['ytick.minor.visible'] = 'True'
plt.rcParams['xtick.minor.visible'] = 'True'
plt.rcParams['xtick.direction'] = 'inout'
plt.rcParams['ytick.direction'] = 'inout'
plt.rcParams['ytick.right'] = 'True'
plt.rcParams['xtick.top'] = 'True'

# use a color-blind friendly palette
# orange, red, light blue, dark blue
colors=['#FF9408','#DC4D01','#00A9E0','#016795']

def func(x, a, b):
    return a + b*x

nams=['gammapav','zetatuc','pimen']

fig = plt.figure(figsize=(15, 6))

plt.ion()
plt.clf()

for i in range(0,len(nams)):

	star=nams[i]

	if (star == 'gammapav'):
		# gamma Pav = TIC265488188
		tit='gamma Pav = TIC265488188'
		dat=ascii.read('peakbagging/265488188_bgcorr.txt')
		freqs=ascii.read('peakbagging/gammapav-freq.csv')
		usel=0
		lol=2000
		upl=3250

	if (star == 'zetatuc'):
		# zeta Tuc = TIC425935521
		tit='zeta Tuc = TIC425935521'
		dat=ascii.read('peakbagging/425935521_bgcorr.txt')
		freqs=ascii.read('peakbagging/zetatuc-freq.csv')
		usel=0
		lol=2200
		upl=3400

	if (star == 'pimen'):
		# pi Men = TIC261136679
		tit='pi Men = TIC261136679'
		dat=ascii.read('peakbagging/261136679_bgcorr.txt')
		freqs=ascii.read('peakbagging/pimen-freq.csv')
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

	plt.subplot(1,3,i+1)
	plt.imshow(echz,aspect="auto",extent=(echx.min(), echx.max(), echy.min(), echy.max()),origin="lower",cmap='Greys',interpolation='None')
	um=np.where(freqs['l'] == 0)[0]
	#plt.errorbar(freqs['freq'][um] % dnu,freqs['freq'][um],xerr=freqs['err'][um],fmt='o',color=colors[3],fillstyle='none',lw=2)
	plt.plot(freqs['freq'][um] % dnu,freqs['freq'][um],'o',color=colors[3])

	#plt.plot(freqs['freq'][um] % dnu,freqs['freq'][um],fmt='o',color=colors[3],lw=10)
	um=np.where(freqs['l'] == 1)[0]
	#plt.errorbar(freqs['freq'][um] % dnu,freqs['freq'][um],xerr=freqs['err'][um],fmt='o',color=colors[1],fillstyle='none',lw=2)
	plt.plot(freqs['freq'][um] % dnu,freqs['freq'][um],'s',color=colors[1])

	um=np.where(freqs['l'] == 2)[0]
	plt.plot(freqs['freq'][um] % dnu,freqs['freq'][um],'D',color=colors[0])
	#plt.errorbar(freqs['freq'][um] % dnu,freqs['freq'][um],xerr=freqs['err'][um],fmt='o',color=colors[0],fillstyle='none',lw=2)

	plt.ylim([lol,upl])
	plt.xlabel('Frequency mod '+str(dnu)[0:6]+' $\mu$Hz')
	if (i == 0):
		plt.ylabel('Frequency ($\mu$Hz)')
		plt.annotate("$\\gamma$ Pav (F9V)", xy=(0.1, 0.9), xycoords="axes fraction",fontsize=20,color='black')

	if (star == 'zetatuc'):
		plt.annotate("$\\zeta$ Tuc (F9.5V)", xy=(0.1, 0.9), xycoords="axes fraction",fontsize=20,color='black')
		
	if (star == 'pimen'):
		plt.annotate("$\\pi$ Men (G0V)", xy=(0.1, 0.9), xycoords="axes fraction",fontsize=20,color='black')

plt.subplots_adjust(wspace=0.25,hspace=0.2,left=0.07,right=0.99,bottom=0.11,top=0.97)
plt.savefig('fig7.png',dpi=150)

plt.show()	
#plt.savefig('fig-echelle.pdf',dpi=150)

