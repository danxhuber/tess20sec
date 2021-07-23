import matplotlib.pyplot as plt
from astropy.stats import LombScargle
from astropy.io import ascii
import numpy as np
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel
from astropy.stats import LombScargle
from scipy.signal import savgol_filter as savgol
import pdb, fnmatch
from astropy.io import fits
import os, glob
from astropy.stats import mad_std
import pandas as pd

# subroutine to perform rough sigma clipping
def sigclip(x,y,subs,sig):
    keep = np.zeros_like(x)
    start=0
    end=subs
    nsubs=int((len(x)/subs)+1)
    for i in range(0,nsubs):        
        me=np.mean(y[start:end])
        sd=np.std(y[start:end])
        good=np.where((y[start:end] > me-sig*sd) & (y[start:end] < me+sig*sd))[0]
        keep[start:end][good]=1
        start=start+subs
        end=end+subs
    return keep


# sigma clipping box in hours
fw=5.

# high-pass filter width in days
hp=1.


# pi Men: SYDSAP data
fname='piMen_20sec'
fn='../../SYDSAP/piMen_20s_SYDSAP.csv'
#fn='../../lightcurves/piMen_newsectors/piMen_20s_S27283134_SYDSAP.csv'

fname='piMen_2min'
fn='../../SYDSAP/piMen_120s_SYDSAP.csv'
#fn='raw/piMen_new/piMen_120s_S27283134_SYDSAP.csv'

# zeta Tuc: SYDSAP
fname='zetaTuc_20sec'
fn='../../SYDSAP/zetTuc_S28_20s_SYDSAP.csv'

fname='zetaTuc_2min'
fn='../../SYDSAP/zetTuc_S28_120s_SYDSAP.csv'


# gamma Pav: SPOC data 
fname='gammaPav_2min'
fn='../../sample/mastDownload/TESS/tess2020186164531-s0027-0000000265488188-0189-s/tess2020186164531-s0027-0000000265488188-0189-s_lc.fits'
fname='gammaPav_20sec'
fn='../../sample/mastDownload/TESS/tess2020186164531-s0027-0000000265488188-0189-a_fast/tess2020186164531-s0027-0000000265488188-0189-a_fast-lc.fits'

if (fnmatch.fnmatch(fname,'*gammaPav*')):
	x=fits.open(fn)
	data=x[1].data
	um=np.where((data['QUALITY'] == 0.) & (np.isfinite(data['PDCSAP_FLUX'])))[0]
	time=data['TIME'][um]
	flux=data['PDCSAP_FLUX'][um]/np.median(data['PDCSAP_FLUX'][um])
	good=np.isfinite(flux)
	time=time[good]
	fluxs=flux[good]
else:
	if (fnmatch.fnmatch(fname,'piMen_2min')):
		dat=ascii.read(fn,delimiter=',')
		time=np.array(dat['time'])
		um=np.where(time < 2061.)[0]
		dat['flux'][um]=dat['flux'][um]/np.median(dat['flux'][um])
		um=np.where(time > 2061.)[0]
		dat['flux'][um]=dat['flux'][um]/np.median(dat['flux'][um])
		fluxs=np.array(dat['flux'])
	else:
		dat=ascii.read(fn,delimiter=',')
		time=np.array(dat['time'])
		fluxs=np.array(dat['flux']/np.median(dat['flux']))

# median time sampling in minutes
df=np.median(time[1::]-time[0:-1])*24.*60.

if (fnmatch.fnmatch(fname,'*piMen*')):
	ph=time % 6.2679
	out=np.where((ph < 2.9) | (ph > 3.04))
	time=time[out]
	fluxs=fluxs[out]

plt.ion()
plt.clf()
plt.plot(time,fluxs,'.')

# sigma clipping
res=sigclip(time,fluxs,np.int(5./(df/60.)),5)
good = np.where(res == 1)[0]
time=time[good]
fluxs=fluxs[good]

plt.plot(time,fluxs,'.')


boxsize=hp/(df/60./24.)
box_kernel = Box1DKernel(boxsize)
if (int(boxsize) % 2 == 0):
	boxsize=int(boxsize)+1
else:
	boxsize=int(boxsize)
smoothed_flux = savgol(fluxs,boxsize,1,mode='mirror')
plt.plot(time,smoothed_flux)

fluxs=fluxs/(smoothed_flux)
plt.clf()
plt.plot(time,fluxs,'.')

nyq=1./(2.*df/60./24.)
#nyq=1./(4./60./24.)

freq, amp = LombScargle(time,fluxs).autopower(method='fast',samples_per_peak=1,maximum_frequency=nyq)
freq = 1000.*freq/86.4
bin = freq[1]-freq[0]
amp = 2.*amp*np.var(fluxs*1e6)/(np.sum(amp)*bin)
gauss_kernel = Gaussian1DKernel(2)
pssm = convolve(amp, gauss_kernel)

plt.clf()

plt.subplot(2,1,1)
plt.plot(time,fluxs,'.')

plt.subplot(2,1,2)
plt.plot(freq,amp)
plt.plot(freq,pssm)
plt.xlim([1.,nyq/0.0864])
plt.xlabel('frequency (muHz)')
plt.ylabel('power density')
plt.title(fname)

#ascii.write([time,fluxs],'lc_processed/'+fname+'_LC.txt',format='no_header')
#ascii.write([freq,amp],'lc_processed/'+fname+'_PS.txt',format='no_header')








dat1=ascii.read('piMen_20sec_LC.txt')
dat2=ascii.read('piMen_2min_LC.txt')

plt.ion()
plt.clf()
plt.plot(dat1['col1'],dat1['col2'],'.',label='20sec')
plt.plot(dat2['col1'],dat2['col2'],'.',label='2min')
plt.legend()




dat1=ascii.read('425935521.2min.ts.txt')
dat2=ascii.read('zetaTuc_2min_LC.txt')

plt.ion()
plt.clf()
plt.plot(dat2['col1'],dat2['col2'],'.',label='SYD')
plt.plot(dat1['col1'],dat1['col2'],'.',label='SPOC')
plt.legend()


dat1=ascii.read('425935521.2min.ps.txt')
dat2=ascii.read('zetaTuc_2min_PS.txt')

dat1=ascii.read('425935521.ps.txt')
dat2=ascii.read('zetaTuc_20sec_PS.txt')


plt.ion()
plt.clf()
plt.plot(dat2['col1'],dat2['col2'])
plt.plot(dat1['col1'],dat1['col2'])




################ OLD

plt.clf()
plt.plot(freq,amp)
plt.plot(freq,pssm)
plt.xlim([1000,4000])


# zeta Tuc 20-sec cadence
dat=ascii.read('data_SYDSAP/zetTuc_S28_20s_SYDSAP.csv',delimiter=',')
time=np.array(dat['time'])
fluxs=np.array(dat['flux']/np.median(dat['flux']))

plt.ion()
plt.clf()
plt.plot(time,fluxs)

# sigma clipping
res=sigclip(time,fluxs,1000,5)
good = np.where(res == 1)[0]
plt.plot(time[good],fluxs[good])
time=time[good]
fluxs=fluxs[good]

plt.clf()
plt.plot(time,fluxs)

width=1.0
boxsize=width/(20./60./60./24.)
box_kernel = Box1DKernel(boxsize)
smoothed_flux = savgol(fluxs,int(boxsize)-1,1,mode='mirror')
plt.plot(time,smoothed_flux)


fluxs=fluxs/(smoothed_flux)
plt.clf()
plt.plot(time,fluxs,'.')

nyq=1./(2.*20./60./60./24.)
#nyq=1./(4./60./24.)

freq, amp = LombScargle(time,fluxs).autopower(method='fast',samples_per_peak=1,maximum_frequency=nyq)
freq = 1000.*freq/86.4
bin = freq[1]-freq[0]
amp = 2.*amp*np.var(fluxs*1e6)/(np.sum(amp)*bin)
gauss_kernel = Gaussian1DKernel(12)
pssm = convolve(amp, gauss_kernel)

plt.clf()
plt.loglog(freq,amp)
plt.plot(freq,pssm)
plt.xlim([100.,nyq/0.0864])
plt.xlabel('frequency (muHz)')
plt.ylabel('power density')

ascii.write([time,fluxs],'data_SYDSAP/zetaTuc_LC.txt',format='no_header')
ascii.write([freq,amp],'data_SYDSAP/zetaTuc_PS.txt',format='no_header')




# pi men 20-sec cadence
dat=ascii.read('data_SYDSAP/piMen_20s_SYDSAP.csv',delimiter=',')
dat=ascii.read('data_SYDSAP/piMen_20s_S272831_SYDSAP.csv',delimiter=',')

time=np.array(dat['time'])
fluxs=np.array(dat['flux'])

plt.ion()
plt.clf()
plt.plot(time,fluxs)

# sigma clipping
res=sigclip(time,fluxs,1000,5)
good = np.where(res == 1)[0]
plt.plot(time[good],fluxs[good])
time=time[good]
fluxs=fluxs[good]

ph=time % 6.2679
out=np.where((ph < 2.9) | (ph > 3.04))
time=time[out]
fluxs=fluxs[out]

plt.clf()
plt.plot(time,fluxs)

width=1.0
boxsize=width/(20./60./60./24.)
box_kernel = Box1DKernel(boxsize)
smoothed_flux = savgol(fluxs,int(boxsize)-1,1,mode='mirror')
plt.plot(time,smoothed_flux)


fluxs=fluxs/(smoothed_flux)
plt.clf()
plt.plot(time,fluxs,'.')

nyq=1./(2.*20./60./60./24.)
#nyq=1./(4./60./24.)

freq, amp = LombScargle(time,fluxs).autopower(method='fast',samples_per_peak=1,maximum_frequency=nyq)
freq = 1000.*freq/86.4
bin = freq[1]-freq[0]
amp = 2.*amp*np.var(fluxs*1e6)/(np.sum(amp)*bin)
gauss_kernel = Gaussian1DKernel(12)
pssm = convolve(amp, gauss_kernel)

plt.clf()
plt.loglog(freq,amp)
plt.plot(freq,pssm)
plt.xlim([10.,nyq/0.0864])
plt.xlabel('frequency (muHz)')
plt.ylabel('power density')

ascii.write([time,fluxs],'data_SYDSAP/piMen_LC.txt',format='no_header')
ascii.write([freq,amp],'data_SYDSAP/piMen_PS.txt',format='no_header')




# pi men 2-min cadence
dat=ascii.read('data_SYDSAP/piMen_120s_SYDSAP.csv',delimiter=',')
time=np.array(dat['time'])
um=np.where(time < 2061.)[0]
dat['flux'][um]=dat['flux'][um]/np.median(dat['flux'][um])
um=np.where(time > 2061.)[0]
dat['flux'][um]=dat['flux'][um]/np.median(dat['flux'][um])
fluxs=np.array(dat['flux'])

plt.ion()
plt.clf()
plt.plot(time,fluxs)

# sigma clipping
#res=sigclip(time,fluxs,1000,5)

res=sigclip(time,fluxs,80,5)
good = np.where(res == 1)[0]
plt.plot(time[good],fluxs[good])
time=time[good]
fluxs=fluxs[good]

ph=time % 6.2679
out=np.where((ph < 2.9) | (ph > 3.04))
time=time[out]
fluxs=fluxs[out]

plt.clf()
plt.plot(time,fluxs)

width=1.0
boxsize=width/(2./60./24.)
box_kernel = Box1DKernel(boxsize)
smoothed_flux = savgol(fluxs,int(boxsize)-1,1,mode='mirror')
plt.plot(time,smoothed_flux)


fluxs=fluxs/(smoothed_flux)
plt.clf()
plt.plot(time,fluxs,'.')

#nyq=1./(2.*20./60./60./24.)
nyq=1./(4./60./24.)

freq, amp = LombScargle(time,fluxs).autopower(method='fast',samples_per_peak=1,maximum_frequency=nyq)
freq = 1000.*freq/86.4
bin = freq[1]-freq[0]
amp = 2.*amp*np.var(fluxs*1e6)/(np.sum(amp)*bin)
gauss_kernel = Gaussian1DKernel(12)
pssm = convolve(amp, gauss_kernel)

plt.clf()
plt.loglog(freq,amp)
plt.plot(freq,pssm)
plt.xlim([100.,nyq/0.0864])
plt.xlabel('frequency (muHz)')
plt.ylabel('power density')

ascii.write([time,fluxs],'data_SYDSAP/piMen_LC_2min.txt',format='no_header')
ascii.write([freq,amp],'data_SYDSAP/piMen_PS_2min.txt',format='no_header')
