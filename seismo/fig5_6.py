import numpy as np
import os, sys
from astropy.io import ascii
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel

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


### gamma Pav
slow=ascii.read('data/lc_processed/gammaPav_2min_PS.txt')
fast=ascii.read('data/lc_processed/gammaPav_20sec_PS.txt')

#slow2=ascii.read('data/processed/425935521.2min.ps.txt')
slow2=ascii.read('data/lc_processed/zetaTuc_2min_PS.txt')
fast2=ascii.read('data/lc_processed/zetaTuc_20sec_PS.txt')

slow3=ascii.read('data/lc_processed/piMen_2min_PS.txt')
fast3=ascii.read('data/lc_processed/piMen_20sec_PS.txt')

lim1=1500
lim2=3800
wd=1.0

lowl=0
upl=19.99

fss=18
fss2=16

alf=1.0

fig = plt.figure(figsize=(15, 6))

plt.ion()
plt.clf()

ax1 = plt.subplot(2,3,1)

fres=slow['col1'][1]-slow['col1'][0]
gauss_kernel = Gaussian1DKernel(int(wd/fres))
pssm = convolve(np.array(slow['col2']), gauss_kernel)
ax1.plot(slow['col1'],slow['col2'],color='grey',alpha=0.25)
ax1.plot(slow['col1'],pssm,color=colors[3],alpha=alf,lw=1.25)

ax1.set_xlim([lim1,lim2])
#ax1.set_xlabel('Frequency ($\mu$Hz)')
#ax1.set_ylabel('Power ((m/s)$^2$)')
ax1.set_ylim([lowl,20])

ax1.annotate("2-min", xy=(0.03, 0.82), xycoords="axes fraction",fontsize=fss2,color=colors[3])
ax1.set_xticklabels([])
#plt.title("$\\gamma$ Pav (F9V, V=4.2)")
ax1.annotate("$\\gamma$ Pav (F9V)", xy=(0.6, 0.82), xycoords="axes fraction",fontsize=fss,color='black')


ax2 = plt.subplot(2,3,4)

fres=fast['col1'][1]-fast['col1'][0]
gauss_kernel = Gaussian1DKernel(int(wd/fres))
pssm = convolve(np.array(fast['col2']), gauss_kernel)
ax2.plot(fast['col1'],fast['col2'],color='grey',alpha=0.25)
ax2.plot(fast['col1'],pssm,color=colors[1],alpha=alf,lw=1.25)
ax2.set_xlim([lim1,lim2])
ax2.set_xlabel('Frequency ($\mu$Hz)',labelpad=5)
ax2.set_ylabel('                                   Power Density (ppm$^{2}$ $\mu$Hz$^{-1}$)',labelpad=5)
ax2.annotate("20-sec", xy=(0.03, 0.82), xycoords="axes fraction",fontsize=fss2,color=colors[1])
ax2.set_ylim([lowl,upl])


### zeta Tuc
lim1=1500
lim2=3800
lowl=0
upl=19.99

ax1 = plt.subplot(2,3,2)
fres=slow2['col1'][1]-slow2['col1'][0]
gauss_kernel = Gaussian1DKernel(int(wd/fres))
pssm = convolve(np.array(slow2['col2']), gauss_kernel)
ax1.plot(slow2['col1'],slow2['col2'],color='grey',alpha=0.25)
ax1.plot(slow2['col1'],pssm,color=colors[3],alpha=alf,lw=1.25)

ax1.set_xlim([lim1,lim2])
#ax1.set_xlabel('Frequency ($\mu$Hz)')
#ax1.set_ylabel('Power ((m/s)$^2$)')
ax1.set_ylim([lowl,30])

ax1.annotate("2-min", xy=(0.03, 0.82), xycoords="axes fraction",fontsize=fss2,color=colors[3])
ax1.annotate("$\\zeta$ Tuc (F9.5V)", xy=(0.55, 0.82), xycoords="axes fraction",fontsize=fss,color='black')

ax1.set_xticklabels([])
#plt.title("$\\gamma$ Pav (F9V, V=4.2)")


ax2 = plt.subplot(2,3,5)
fres=fast2['col1'][1]-fast2['col1'][0]
gauss_kernel = Gaussian1DKernel(int(wd/fres))
pssm = convolve(np.array(fast2['col2']), gauss_kernel)
ax2.plot(fast2['col1'],fast2['col2'],color='grey',alpha=0.25)
ax2.plot(fast2['col1'],pssm,color=colors[1],alpha=alf,lw=1.25)
ax2.set_xlim([lim1,lim2])
ax2.set_xlabel('Frequency ($\mu$Hz)',labelpad=5)
ax2.annotate("20-sec", xy=(0.03, 0.82), xycoords="axes fraction",fontsize=fss2,color=colors[1])
ax2.set_ylim([lowl,upl])


### pi Men
lim1=1500
lim2=3800

ax1 = plt.subplot(2,3,3)
fres=slow3['col1'][1]-slow3['col1'][0]
gauss_kernel = Gaussian1DKernel(int(wd/fres))
pssm = convolve(np.array(slow3['col2']), gauss_kernel)
ax1.plot(slow3['col1'],slow3['col2'],color='grey',alpha=0.25)
ax1.plot(slow3['col1'],pssm,color=colors[3],alpha=alf,lw=1.25)

ax1.set_xlim([lim1,lim2])
#ax1.set_xlabel('Frequency ($\mu$Hz)')
#ax1.set_ylabel('Power ((m/s)$^2$)')
ax1.set_ylim([lowl,100])

ax1.annotate("2-min", xy=(0.03, 0.82), xycoords="axes fraction",fontsize=fss2,color=colors[3])
ax1.annotate("$\\pi$ Men (G0V)", xy=(0.55, 0.82), xycoords="axes fraction",fontsize=fss,color='black')

ax1.set_xticklabels([])
#plt.title("$\\gamma$ Pav (F9V, V=4.2)")


ax2 = plt.subplot(2,3,6)
fres=fast3['col1'][1]-fast3['col1'][0]
gauss_kernel = Gaussian1DKernel(int(wd/fres))
pssm = convolve(np.array(fast3['col2']), gauss_kernel)
ax2.plot(fast3['col1'],fast3['col2'],color='grey',alpha=0.25)
ax2.plot(fast3['col1'],pssm,color=colors[1],alpha=alf,lw=1.25)
ax2.set_xlim([lim1,lim2])
ax2.set_xlabel('Frequency ($\mu$Hz)',labelpad=5)
ax2.annotate("20-sec", xy=(0.03, 0.82), xycoords="axes fraction",fontsize=fss2,color=colors[1])
ax2.set_ylim([lowl,upl])

plt.subplots_adjust(wspace=0.16,hspace=0.0,left=0.06,right=0.99,bottom=0.13,top=0.97)
plt.savefig('fig5.png',dpi=150)


fig = plt.figure(figsize=(13, 7))
plt.rcParams['font.size']=18

##################################

lim1=100
lim2=25000
wd=1.0

lws=3
alfs=0.6

lowl=0
upl=20

fss=22

xax=np.arange(1.,26000.,10.)
nyq=4100.
cor = (np.sin(np.pi/2. * xax/nyq)/(np.pi/2. * xax/nyq))**2
nyq2=1./(2.*20./60./60./24.)/0.0864
cor2 = (np.sin(np.pi/2. * xax/nyq2)/(np.pi/2. * xax/nyq2))**2


plt.ion()
plt.clf()

ax1 = plt.subplot(3,1,1)
fres=slow['col1'][1]-slow['col1'][0]
gauss_kernel = Gaussian1DKernel(int(wd/fres))
pssm = convolve(np.array(fast['col2']), gauss_kernel)
#ax1.plot(slow['col1'],pssm,color='blue',alpha=0.75,lw=1.25)
ax1.plot(fast['col1'],pssm,color='grey',alpha=0.75,lw=1.25)
ax1.set_xlim([lim1,lim2])
ax1.set_yscale('log')
ax1.set_xscale('log')
#ax1.annotate("2-min", xy=(0.03, 0.82), xycoords="axes fraction",fontsize=18,color='blue')
ax1.set_xticklabels([])
#plt.title("$\\gamma$ Pav (F9V, V=4.2)")
ax1.annotate("$\\gamma$ Pav (F9V)", xy=(0.02, 0.15), xycoords="axes fraction",fontsize=fss,color='black')
fres=fast['col1'][1]-fast['col1'][0]
gauss_kernel = Gaussian1DKernel(int(wd/fres))
pssm = convolve(np.array(fast['col2']), gauss_kernel)
#ax1.plot(fast['col1'],fast['col2'],color='grey',alpha=0.25)
#ax1.annotate("20-sec", xy=(0.03, 0.82), xycoords="axes fraction",fontsize=18,color='red')
#ax1.plot([4166,4166],[0.001,100],color='red',ls='dashed',alpha=alfs,lw=lws)
ax1.set_ylim([0.15,20])

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.plot(xax,cor,label='2min',lw=lws,alpha=alfs,color=colors[1])
ax2.plot(xax,cor2,ls='dashed',label='20sec',lw=lws,alpha=alfs,color=colors[3])
ax2.set_ylim([0.001,1.1])
ax2.set_xscale('log')
ax2.set_xticklabels([])
ax1.set_xlim([lim1,lim2])


ax1 = plt.subplot(3,1,2)
fres=slow2['col1'][1]-slow2['col1'][0]
gauss_kernel = Gaussian1DKernel(int(wd/fres))
pssm = convolve(np.array(fast2['col2']), gauss_kernel)
#ax1.plot(slow2['col1'],pssm,color='blue',alpha=0.75,lw=1.25)
ax1.set_xlim([lim1,lim2])
#ax1.set_xlabel('Frequency ($\mu$Hz)')
#ax1.set_ylabel('Power ((m/s)$^2$)')
#ax1.set_ylim([lowl,upl])
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_xticklabels([])
ax1.annotate("$\\zeta$ Tuc (F9.5V)", xy=(0.02, 0.15), xycoords="axes fraction",fontsize=fss,color='black')
fres=fast2['col1'][1]-fast2['col1'][0]
gauss_kernel = Gaussian1DKernel(int(wd/fres))
pssm = convolve(np.array(fast2['col2']), gauss_kernel)
#ax1.plot(fast2['col1'],fast2['col2'],color='grey',alpha=0.25)
ax1.plot(fast2['col1'],pssm,color='grey',alpha=0.75,lw=1.25)
ax1.set_xlim([lim1,lim2])
#ax1.annotate("20-sec", xy=(0.03, 0.82), xycoords="axes fraction",fontsize=18,color='red')
#ax1.set_ylim([lowl,upl])
ax1.set_ylabel('Power Density (ppm$^{2}$ $\mu$Hz)',labelpad=5)
#ax1.plot([4166,4166],[0.001,100],color='red',ls='dashed',alpha=alfs,lw=lws)
ax1.set_ylim([0.15,20])

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.plot(xax,cor,label='2min',lw=lws,alpha=alfs,color=colors[1])
ax2.plot(xax,cor2,ls='dashed',label='20sec',lw=lws,alpha=alfs,color=colors[3])
ax2.set_ylabel('Fractional Power Attenuation')
ax2.set_ylim([0.001,1.1])
ax2.set_xscale('log')
ax2.set_xticklabels([])


ax1 = plt.subplot(3,1,3)
fres=slow3['col1'][1]-slow3['col1'][0]
gauss_kernel = Gaussian1DKernel(int(wd/fres))
pssm = convolve(np.array(slow3['col2']), gauss_kernel)
#ax1.plot(slow3['col1'],pssm,color='blue',alpha=0.75,lw=1.25)
ax1.set_xlim([lim1,lim2])
#ax1.set_xlabel('Frequency ($\mu$Hz)')
#ax1.set_ylabel('Power ((m/s)$^2$)')
#ax1.set_ylim([lowl,upl])
ax1.set_yscale('log')
ax1.set_xscale('log')
#ax1.annotate("2-min", xy=(0.03, 0.82), xycoords="axes fraction",fontsize=18,color='blue')
#plt.title("$\\gamma$ Pav (F9V, V=4.2)")
ax1.annotate("$\\pi$ Men (G0V)", xy=(0.02, 0.15), xycoords="axes fraction",fontsize=fss,color='black')
fres=fast3['col1'][1]-fast3['col1'][0]
gauss_kernel = Gaussian1DKernel(int(wd/fres))
pssm = convolve(np.array(fast3['col2']), gauss_kernel)
#ax1.plot(fast3['col1'],fast3['col2'],color='grey',alpha=0.25)
ax1.plot(fast3['col1'],pssm,color='grey',alpha=0.75,lw=1.25)
ax1.set_xlim([lim1,lim2])
ax1.set_xlabel('Frequency ($\mu$Hz)',labelpad=5)
#ax1.annotate("20-sec", xy=(0.03, 0.82), xycoords="axes fraction",fontsize=18,color='red')
#ax1.plot([4166,4166],[0.001,100],color='red',ls='dashed',alpha=alfs,lw=lws)
ax1.set_ylim([1,20])

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.plot(xax,cor,label='2min',lw=lws,alpha=alfs,color=colors[1])
ax2.plot(xax,cor2,ls='dashed',label='20sec',lw=lws,alpha=alfs,color=colors[3])
#ax2.set_ylabel('Fractional Attenuation')
ax2.set_ylim([0.001,1.1])
ax2.set_xscale('log')
#plt.savefig('fig-compcadence-log.pdf',dpi=150)

plt.subplots_adjust(hspace=0.0,wspace=0.2,left=0.08,bottom=0.11,right=0.93,top=0.98)

plt.savefig('fig6.png',dpi=150)





'''
lim1=100
lim2=20000

plt.ion()
plt.clf()

ax1 = plt.subplot(2,1,1)
#ax1.plot(freq_xtd,amp_xtd)
ax1.plot(slow['col1'],slow['col2'],color='blue',alpha=0.75,lw=1.25)

ax1.set_xlim([lim1,lim2])
#ax1.set_xlabel('Frequency ($\mu$Hz)')
#ax1.set_ylabel('Power ((m/s)$^2$)')
ax1.set_ylim([0.001,100])
ax1.set_yscale('log')
ax1.set_xscale('log')

ax1.annotate("S27 2-min cadence", xy=(0.03, 0.88), xycoords="axes fraction",fontsize=18,color='blue')
ax1.set_xticklabels([])
plt.title("$\\gamma$ Pav (F9V, V=4.2)")


ax2 = plt.subplot(2,1,2)
ax2.plot(fast['col1'],(fast['col2']),color='red',alpha=0.75,lw=1.25)
ax2.set_xlim([lim1,lim2])
ax2.set_xlabel('Frequency ($\mu$Hz)',labelpad=5)
ax2.set_ylabel('                                   Power (Arbitrary Units)',labelpad=5)
ax2.annotate("S27 20-sec cadence", xy=(0.03, 0.88), xycoords="axes fraction",fontsize=18,color='red')
ax2.set_ylim([0.001,100])

ax2.set_yscale('log')
ax2.set_xscale('log')

plt.savefig('gammapav.png',dpi=150)


gauss_kernel = Box1DKernel(4)
pssm = convolve(np.array(fast['col2']), gauss_kernel)

freqs=ascii.read('freqs.txt')

plt.clf()
plt.plot(fast['col1'],fast['col2'],color='grey',alpha=0.5)
plt.plot(fast['col1'],pssm,color='black')

plt.xlim([2000,3500])
for i in range(0,len(freqs)):
	if (freqs['col2'][i] == 0):
		plt.plot([freqs['col1'][i],freqs['col1'][i]],[0,30],ls='dashed',color='red')
	if (freqs['col2'][i] == 1):
		plt.plot([freqs['col1'][i],freqs['col1'][i]],[0,30],ls='dashed',color='blue')
	if (freqs['col2'][i] == 2):
		plt.plot([freqs['col1'][i],freqs['col1'][i]],[0,30],ls='dashed',color='green')



slow=ascii.read('data/265488188.2min.ts.txt')
fast=ascii.read('data/265488188.ts.txt')

time=slow['col1']
flux=slow['col2']
nyq=1./(2.*2./60./24.)
freq, amp = LombScargle(time,flux).autopower(method='fast',samples_per_peak=1,maximum_frequency=nyq)
fres=freq[1]-freq[0]
print(fres)
freq = 1000.*freq/86.4
bin = freq[1]-freq[0]
amp = 2.*amp*np.var(flux*1e6)/(np.sum(amp)*bin)

plt.clf()
plt.plot(freq,amp)

time=fast['col1']
flux=fast['col2']
nyq=1./(2.*20./60./60./24.)
freq, amp = LombScargle(time,flux).autopower(method='fast',samples_per_peak=1,maximum_frequency=nyq)
fres=freq[1]-freq[0]
print(fres)
freq = 1000.*freq/86.4
bin = freq[1]-freq[0]
amp = 2.*amp*np.var(flux*1e6)/(np.sum(amp)*bin)

plt.plot(freq,amp)

'''
