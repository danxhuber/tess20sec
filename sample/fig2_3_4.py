import numpy as np
import os, sys
from astropy.io import ascii, fits
import matplotlib.pyplot as plt
import pandas as pd
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel
import glob, pdb
from scipy.signal import savgol_filter as savgol
from astropy.stats import mad_std
import bin

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
plt.rcParams['xtick.major.pad']='6'
plt.rcParams['ytick.major.pad']='8'
plt.rcParams['ytick.minor.visible'] = 'True'
plt.rcParams['xtick.minor.visible'] = 'True'
plt.rcParams['xtick.direction'] = 'inout'
plt.rcParams['ytick.direction'] = 'inout'
plt.rcParams['ytick.right'] = 'True'
plt.rcParams['xtick.top'] = 'True'

# use a color-blind friendly palette
# orange, red, light blue, dark blue
colors=['#FF9408','#DC4D01','#00A9E0','#016795']

def bin_set(xdata,ydata,bins):

    nbins=len(bins)-1
    
    resx=np.zeros(nbins)
    resy=np.zeros(nbins)
    erry=np.zeros(nbins)
    
    for i in range(0,len(bins)-1):
        um=np.where( (xdata >= bins[i]) & (xdata < bins[i+1]))[0]
        
        resx[i]=(bins[i]+bins[i+1])/2.
        resy[i]=np.median(ydata[um])
        erry[i]=np.std(ydata[um])/np.sqrt(len(um))
        #print(bins[i],np.median(ydata[um]),np.std(ydata[um]),len(um))
        #print bins[i],bins[i+1],resx[i],resy[i],len(um)
        #print xdata[um]
        #break
        
    return resx,resy,erry
    

def pixel_cost(x, mask_size='new'):
	""" The number of pixels in the aperture. Use when calculating instrumental
	noise with TESS in calc_noise().
	Kewargs
	mask_size ('conservative' or 'new'): the mask size to use for the pixel cost.
	"""

	if mask_size == 'conservative':
		N = np.ceil(10.0**-5.0 * 10.0**(0.4*(20.0-x)))
		npix_aper = 10*(N+10)

	if mask_size == 'new':
		# updated number of pixels equation from Tiago on 22.03.18
		npix_aper = np.ceil(10**(0.8464 - 0.2144 * (x - 10.0)))

	total = np.cumsum(npix_aper)
	per_cam = 26*4 # to get from the total pixel cost to the cost per camera at a given time, divide by this
	pix_limit = 1.4e6 # the pixel limit per camera at a given time

	return npix_aper


def calc_noise(imag, exptime, teff, e_lng = 0, e_lat = 30, g_lng = 96, g_lat = -30, subexptime = 2.0, npix_aper = 10, \
frac_aper = 0.76, e_pix_ro = 10, geom_area = 60.0, pix_scale = 21.1, sys_limit = 0):

    omega_pix = pix_scale**2.0
    n_exposures = exptime/subexptime

    # electrons from the star
    megaph_s_cm2_0mag = 1.6301336 + 0.14733937*(teff-5000.0)/5000.0
    e_star = 10.0**(-0.4*imag) * 10.0**6 * megaph_s_cm2_0mag * geom_area * exptime * frac_aper
    e_star_sub = e_star*subexptime/exptime

    # e/pix from zodi
    dlat = (abs(e_lat)-90.0)/90.0
    vmag_zodi = 23.345 - (1.148*dlat**2.0)
    e_pix_zodi = 10.0**(-0.4*(vmag_zodi-22.8)) * (2.39*10.0**-3) * geom_area * omega_pix * exptime

    # e/pix from background stars
    dlat = abs(g_lat)/40.0*10.0**0

    dlon = g_lng
    q = np.where(dlon>180.0)
    if len(q[0])>0:
    	dlon[q] = 360.0-dlon[q]

    dlon = abs(dlon)/180.0*10.0**0
    p = [18.97338*10.0**0, 8.833*10.0**0, 4.007*10.0**0, 0.805*10.0**0]
    imag_bgstars = p[0] + p[1]*dlat + p[2]*dlon**(p[3])
    e_pix_bgstars = 10.0**(-0.4*imag_bgstars) * 1.7*10.0**6 * geom_area * omega_pix * exptime

    # compute noise sources
    noise_star = np.sqrt(e_star) / e_star
    noise_sky  = np.sqrt(npix_aper*(e_pix_zodi + e_pix_bgstars)) / e_star
    noise_ro   = np.sqrt(npix_aper*n_exposures)*e_pix_ro / e_star
    noise_sys  = 0.0*noise_star + sys_limit/(1*10.0**6)/np.sqrt(exptime/3600.0)

    noise1 = np.sqrt(noise_star**2.0 + noise_sky**2.0 + noise_ro**2.0)
    noise2 = np.sqrt(noise_star**2.0 + noise_sky**2.0 + noise_ro**2.0 + noise_sys**2.0)
    
    return noise_star,noise_sky,noise_ro,noise1,noise2


fig = plt.figure(figsize=(13, 7.5))

# keeping only qflag = 0
dat=ascii.read('data/scatter-all.csv')

# default in light curve + cosmic rays
#dat=ascii.read('data/scatter/scatter-all-hard.csv')

# rejecting only qflag = 1,2,4,8,32,128 (default in lightkurve)
#dat=ascii.read('data/scatter/scatter-all-default.csv')


um=np.where((dat['tmags'] > 5) & (dat['tmags'] < 16.5))[0]
dat=dat[um]


alf=0.5
alf2=1.0
alf3=0.4
lowm=5
mss=3
ls=10

tmagarr=np.arange(lowm,16,0.1)

npix_aper=pixel_cost(dat['tmags'])
noise_star,noise_sky,noise_ro,noise1,noise2=(calc_noise(dat['tmags'],120.,5800.))
noisevals=noise1*1.4*1e6

upn=np.where((dat['fast_2min'] > noisevals) | (dat['slow_2min'] > noisevals))[0]
lon=np.where((dat['fast_2min'] < noisevals) & (dat['slow_2min'] < noisevals))[0]

plt.ion()
plt.clf()
plt.subplot(2,2,1)
plt.semilogy(dat['tmags'][upn],dat['slow_2min'][upn],'D',color='grey',label='',alpha=alf,fillstyle='none',ms=mss)
plt.semilogy(dat['tmags'][upn],dat['fast_2min'][upn],'o',color='grey',label='',alpha=alf,fillstyle='none',ms=mss)
plt.semilogy(dat['tmags'][lon],dat['fast_2min'][lon],'o',label='20s binned to 2m',color=colors[2],alpha=alf,fillstyle='none',ms=mss)
plt.semilogy(dat['tmags'][lon],dat['slow_2min'][lon],'D',label='2m',color=colors[1],alpha=alf,fillstyle='none',ms=mss)
plt.ylabel("Time-Domain Scatter (ppm)")
plt.xlim([lowm,16])
plt.ylim([10,1e5])
plt.annotate("(a)", xy=(0.05, 0.85), xycoords="axes fraction",fontsize=24,color='black')
lgnd2 = plt.legend(loc='lower right',numpoints=1,handletextpad=0.25,prop={'size':16},handlelength=1.0)
lgnd2.legendHandles[0]._legmarker.set_markersize(ls)
lgnd2.legendHandles[1]._legmarker.set_markersize(ls)


plt.subplot(2,2,3)
plt.plot(dat['tmags'][lon],dat['fast_2min'][lon]/dat['slow_2min'][lon],'.',label='2-min light curves',color=colors[3],alpha=alf3,zorder=-32)
xdat=dat['tmags'][lon]
ydat=dat['fast_2min'][lon]/dat['slow_2min'][lon]
um=np.where(ydat < 1.1)[0]
xdat=xdat[um]
ydat=ydat[um]
bs=np.arange(5.5,17.5,1)
binx,biny,binz=bin_set(xdat,ydat,bs)
#plt.plot(binx,biny,yerr=binz,fmt='-o',color=colors[0],ms=10)
plt.plot(binx,biny,color='black',lw=3)
plt.plot(binx,biny,'-o',color=colors[0],ms=10)
plt.plot(binx,biny,'o',color='black',ms=10,mfc='none')


plt.plot([lowm,20],[1,1],ls='dashed',color=colors[1],lw=3,alpha=alf2)
#plt.plot([lowm,20],[np.sqrt(0.8),np.sqrt(0.8)],ls='dotted',color='green',lw=3,alpha=alf2)
plt.ylabel("Scatter$_{20s}$/Scatter$_{2m}$")
plt.xlabel("TESS Magnitude")
plt.ylim([0.55,1.2])
plt.xlim([lowm,16])
plt.annotate("(c)", xy=(0.05, 0.85), xycoords="axes fraction",fontsize=24,color='black')

'''
xdat=dat['tmags'][lon]
ydat=dat['fast_2min'][lon]/dat['slow_2min'][lon]
fit,cov=np.polyfit(xdat,ydat,3,cov=True)
p = np.poly1d(fit)
xax=np.arange(6,16,0.1)
yax=p(xax)
np.set_printoptions(suppress=True)
print(p)
print(np.sqrt(np.diag(cov)))
tyax=0.000308*xax**3 - 0.0133*xax**2 + 0.199*xax - 0.015
'''

noise_star,noise_sky,noise_ro,noise1,noise2=(calc_noise(dat['tmags'],3600.,5800.))
noisevals=noise1*1.4*1e6
upn=np.where((dat['fast_1hr'] > noisevals) | (dat['slow_1hr'] > noisevals))[0]
lon=np.where((dat['fast_1hr'] < noisevals) & (dat['slow_1hr'] < noisevals))[0]

plt.subplot(2,2,2)
plt.plot(dat['tmags'][upn],dat['fast_1hr'][upn],'D',color='grey',label='',alpha=alf,fillstyle='none',ms=mss)
plt.semilogy(dat['tmags'][upn],dat['slow_1hr'][upn],'o',color='grey',label='',alpha=alf,fillstyle='none',ms=mss)
plt.plot(dat['tmags'][lon],dat['fast_1hr'][lon],'o',label='20s binned to 1h',color=colors[2],alpha=alf,fillstyle='none',ms=mss)
plt.semilogy(dat['tmags'][lon],dat['slow_1hr'][lon],'D',label='2m binned to 1h',color=colors[1],alpha=alf,fillstyle='none',ms=mss)
plt.annotate("(b)", xy=(0.05, 0.85), xycoords="axes fraction",fontsize=24,color='black')
plt.ylabel("Time-Domain Scatter (ppm)")
plt.xlim([lowm,16])
plt.ylim([10,1e5])
lgnd2 = plt.legend(loc='lower right',numpoints=1,handletextpad=0.25,prop={'size':16},handlelength=1.0)
lgnd2.legendHandles[0]._legmarker.set_markersize(ls)
lgnd2.legendHandles[1]._legmarker.set_markersize(ls)

plt.subplot(2,2,4)
plt.plot(dat['tmags'][lon],dat['fast_1hr'][lon]/dat['slow_1hr'][lon],'.',label='2-min light curves',color=colors[3],alpha=alf3,zorder=-32)
plt.plot([lowm,20],[1,1],ls='dashed',color=colors[1],lw=3,alpha=alf2)
#plt.plot([lowm,20],[np.sqrt(0.8),np.sqrt(0.8)],ls='dotted',color='green',lw=3,alpha=alf2)
xdat=dat['tmags'][lon]
ydat=dat['fast_1hr'][lon]/dat['slow_1hr'][lon]
um=np.where(ydat < 1.1)[0]
xdat=xdat[um]
ydat=ydat[um]
bs=np.arange(5.5,17.5,1)
binx2,biny2,binz2=bin_set(xdat,ydat,bs)
#plt.errorbar(binx2,biny2,yerr=binz2,fmt='-o',color=colors[0],ms=10)
plt.plot(binx,biny,color='black',lw=3)
plt.plot(binx,biny,'-o',color=colors[0],ms=10)
plt.plot(binx,biny,'o',color='black',ms=10,mfc='none')
plt.ylabel("Scatter$_{20s}$/Scatter$_{2m}$")
plt.xlabel("TESS Magnitude")
plt.ylim([0.55,1.2])
plt.xlim([lowm,16])
plt.annotate("(d)", xy=(0.05, 0.85), xycoords="axes fraction",fontsize=24,color='black')

#plt.tight_layout(pad=0.5,h_pad=0.5,w_pad=1.5)
plt.subplots_adjust(wspace=0.24,hspace=0.17,left=0.07,right=0.98,bottom=0.1,top=0.97)

plt.show()

for i in range(0,len(binx)):
	print(binx[i],'& $',round(biny[i],3),'\pm',round(binz[i],3),'$ & $',round(biny2[i],3),'\pm',round(binz2[i],3),'$')


'''
fit,cov=np.polyfit(xdat,ydat,3,cov=True)
p = np.poly1d(fit)
xax=np.arange(6,16,0.1)
yax=p(xax)
#um=np.where(xax > 12.5)[0]
#yax[um]=1
#plt.plot(xax,yax,color='orange',alpha=0.75,lw=5,zorder=30)
print(p)
print(np.sqrt(np.diag(cov)))
'''

plt.savefig('fig2.png',dpi=150)



fig = plt.figure(figsize=(7, 5))

dat=ascii.read('data/scatter-s34-parts.csv')
#um=np.where(dat['teff'] < 8000.)[0]
#dat=dat[um]

alf=0.8
alf2=1.0

tmagarr=np.arange(6,16,0.1)

npix_aper=pixel_cost(dat['tmags'])
noise_star,noise_sky,noise_ro,noise1,noise2=(calc_noise(dat['tmags'],120.,5800.))
noisevals=noise1*1.2*1e6

plt.ion()
plt.clf()

lon=np.where((dat['fast_2min_part1'] < noisevals) & (dat['slow_2min_part1'] < noisevals))[0]
xdat=dat['tmags'][lon]
ydat=dat['fast_2min_part1'][lon]/dat['slow_2min_part1'][lon]
um=np.where(ydat < 1.1)[0]
xdat=xdat[um]
ydat=ydat[um]

lon=np.where((dat['fast_2min_part2'] < noisevals) & (dat['slow_2min_part2'] < noisevals))[0]
plt.plot(dat['tmags'][lon],dat['fast_2min_part2'][lon]/dat['slow_2min_part2'][lon],'.',label='',color=colors[3],alpha=0.2)

bs=np.arange(5.5,17.5,1)
binx,biny,binz=bin_set(xdat,ydat,bs)
#plt.errorbar(binx,biny,yerr=binz,fmt='-o',color='crimson',ms=10)
#plt.plot(binx,biny,'-o',color='blue')
plt.plot(dat['tmags'][lon],dat['fast_2min_part1'][lon]/dat['slow_2min_part1'][lon],'.',label='',color=colors[0],alpha=0.2)
plt.plot(binx,biny,color='black',lw=3)
plt.errorbar(binx,biny,yerr=binz,fmt='.',color=colors[0])
plt.plot(binx,biny,'-o',color=colors[0],ms=10,label='Sector 34, Orbit 1')
plt.plot(binx,biny,'o',color='black',ms=10,mfc='none')

lon=np.where((dat['fast_2min_part2'] < noisevals) & (dat['slow_2min_part2'] < noisevals))[0]
xdat=dat['tmags'][lon]
ydat=dat['fast_2min_part2'][lon]/dat['slow_2min_part2'][lon]
um=np.where(ydat < 1.1)[0]
xdat=xdat[um]
ydat=ydat[um]
#plt.plot(dat['tmags'][lon],dat['fast_2min_part2'][lon]/dat['slow_2min_part2'][lon],'.',label='',color=colors[3],alpha=0.2)
binx2,biny2,binz2=bin_set(xdat,ydat,bs)
#plt.errorbar(binx2,biny2,yerr=binz2,fmt='-^',color=colors[3],ms=10,label='Sector 34, Orbit 2')
plt.plot(binx2,biny2,color='black',lw=3)
plt.errorbar(binx2,biny2,yerr=binz2,fmt='.',color=colors[3])
plt.plot(binx2,biny2,'-^',color=colors[3],label='Sector 34, Orbit 2')
plt.plot(binx2,biny2,'^',color='black',ms=10,mfc='none')

plt.plot([5,20],[1,1],ls='dashed',color=colors[1],lw=3,alpha=alf2)
#plt.plot([5,20],[np.sqrt(0.8),np.sqrt(0.8)],ls='dotted',color='green',lw=3,alpha=alf2)

plt.ylabel("Scatter$_{20s}$/Scatter$_{2m}$")
plt.xlabel("TESS Magnitude")
plt.ylim([0.6,1.1])
plt.xlim([5,16])
plt.legend(loc='lower right')
#plt.tight_layout()
plt.subplots_adjust(left=0.13,right=0.97,bottom=0.14,top=0.96)

plt.savefig('fig3.png',dpi=150)



# test influence of quality flags

fig = plt.figure(figsize=(7, 9))

dat=ascii.read('data/scatter-20sec-qflags.csv')
#dat=ascii.read('data/scatter/scatter-20secto2min-qflags.csv')
#dat=ascii.read('data/scatter/scatter-20sec-qflags-scatteredlight.csv')
#dat=ascii.read('data/scatter/scatter-20sec-qflags-scatteredlight-default.csv')
dat2=ascii.read('data/scatter-2min-qflags.csv')

ll=0.83
ul=1.04
lowm=5
upm=16.5
alf=0.1
lws=3
alf2=1.

plt.ion()
plt.clf()
plt.subplot(3,1,1)
plt.plot(dat['tmags'],dat['hardest']/dat['hard'],'.',color=colors[3],alpha=alf)
plt.plot(dat2['tmags'],dat2['hardest']/dat2['hard'],'.',color=colors[0],alpha=alf)
binx,biny,binz=bin_set(dat['tmags'],dat['hardest']/dat['hard'],bs)
plt.plot(binx,biny,color='black',lw=3)
plt.plot(binx,biny,'-o',color=colors[3],ms=10,label='20sec')
plt.plot(binx,biny,'o',color='black',ms=10,mfc='none')
binx,biny,binz=bin_set(dat2['tmags'],dat2['hardest']/dat2['hard'],bs)
plt.plot(binx,biny,color='black',lw=3)
plt.plot(binx,biny,'-^',color=colors[0],ms=10,label='2min')
plt.plot(binx,biny,'^',color='black',ms=10,mfc='none',lw=2)
plt.ylim([0,2])
plt.plot([1,20],[1,1],color=colors[1],ls='dashed',lw=lws,alpha=alf2)
plt.ylabel('$\sigma_{Hardest}$/$\sigma_{Hard}$')
plt.ylim([ll,ul])
plt.xlim([lowm,upm])
plt.annotate("(a)", xy=(0.85, 0.15), xycoords="axes fraction",fontsize=22,color='black')
plt.legend(loc='lower left',fontsize=14)

plt.subplot(3,1,2)
plt.plot(dat['tmags'],dat['hardest']/dat['default'],'.',color=colors[3],alpha=alf)
plt.plot(dat2['tmags'],dat2['hardest']/dat2['default'],'.',color=colors[0],alpha=alf)
binx,biny,binz=bin_set(dat['tmags'],dat['hardest']/dat['default'],bs)
plt.plot(binx,biny,color='black',lw=3)
plt.plot(binx,biny,'-o',color=colors[3],ms=10)
plt.plot(binx,biny,'o',color='black',ms=10,mfc='none')
binx,biny,binz=bin_set(dat2['tmags'],dat2['hardest']/dat2['default'],bs)
plt.plot(binx,biny,color='black',lw=3)
plt.plot(binx,biny,'-^',color=colors[0],ms=10)
plt.plot(binx,biny,'^',color='black',ms=10,mfc='none')
plt.ylim([0,2])
plt.plot([1,20],[1,1],color=colors[1],ls='dashed',lw=lws,alpha=alf2)
plt.ylabel('$\sigma_{Hardest}$/$\sigma_{Default}$')
plt.ylim([ll,ul])
plt.xlim([lowm,upm])
plt.annotate("(b)", xy=(0.85, 0.15), xycoords="axes fraction",fontsize=22,color='black')

plt.subplot(3,1,3)
plt.plot(dat['tmags'],dat['hard']/dat['default'],'.',color=colors[3],alpha=alf)
plt.plot(dat2['tmags'],dat2['hard']/dat2['default'],'.',color=colors[0],alpha=alf)
binx,biny,binz=bin_set(dat['tmags'],dat['hard']/dat['default'],bs)
plt.plot(binx,biny,color='black',lw=3)
plt.plot(binx,biny,'-o',color=colors[3],ms=10)
plt.plot(binx,biny,'o',color='black',ms=10,mfc='none')
binx,biny,binz=bin_set(dat2['tmags'],dat2['hard']/dat2['default'],bs)
plt.plot(binx,biny,color='black',lw=3)
plt.plot(binx,biny,'-^',color=colors[0],ms=10)
plt.plot(binx,biny,'^',color='black',ms=10,mfc='none')
plt.xlim([lowm,upm])
plt.ylim([ll,ul])
plt.xlabel('TESS Magnitude')
plt.plot([1,20],[1,1],color=colors[1],ls='dashed',lw=lws,alpha=alf2)
plt.ylabel('$\sigma_{Hard}$/$\sigma_{Default}$')
plt.annotate("(c)", xy=(0.85, 0.15), xycoords="axes fraction",fontsize=22,color='black')

plt.subplots_adjust(wspace=0.2,hspace=0.0,left=0.16,right=0.98,bottom=0.09,top=0.99)

plt.savefig('fig4.png',dpi=150)


#plt.savefig('../plots_paper/test2.png',dpi=150)






'''
# keeping only qflag = 0
dat=ascii.read('data/scatter/scatter-all.csv')

# default in light curve + ApertureCosmic + CollateralCosmic + Straylight
dat2=ascii.read('data/scatter/scatter-all-hard.csv')

# default in light curve + ApertureCosmic + CollateralCosmic
#dat2=ascii.read('data/scatter/scatter-all-defaultpluscr.csv')

# rejecting only qflag = 1,2,4,8,32,128 (default in lightkurve)
dat3=ascii.read('data/scatter/scatter-all-default.csv')

bs=np.arange(4.0,18.5,1)
bs=np.arange(5.5,17.5,1)

ll=0.83
ul=1.04
lowm=5
upm=16.5
alf=0.2
lws=3
alf2=1.

plt.ion()
plt.clf()
plt.subplot(3,1,1)
plt.plot(dat['tmags'],dat['fast_2min']/dat2['fast_2min'],'.',color=colors[3],alpha=alf)
binx,biny,binz=bin_set(dat['tmags'],dat['fast_2min']/dat2['fast_2min'],bs)
#plt.errorbar(binx,biny,yerr=binz,fmt='-o',color='darkorange',ms=10)
plt.plot(binx,biny,'-o',color=colors[0],ms=10)
plt.ylim([0,2])
plt.plot([1,20],[1,1],color=colors[1],ls='dashed',lw=lws,alpha=alf2)
plt.ylabel('$\sigma_{hardest}$/$\sigma_{hard}$')
plt.ylim([ll,ul])
plt.xlim([lowm,upm])
plt.annotate("(a)", xy=(0.85, 0.15), xycoords="axes fraction",fontsize=22,color='black')

plt.subplot(3,1,2)
plt.plot(dat['tmags'],dat['fast_2min']/dat3['fast_2min'],'.',color=colors[3],alpha=alf)
binx,biny,binz=bin_set(dat['tmags'],dat['fast_2min']/dat3['fast_2min'],bs)
#plt.errorbar(binx,biny,yerr=binz,fmt='-o',color=colors[0],ms=10)
plt.plot(binx,biny,'-o',color=colors[0],ms=10)
plt.ylim([0,2])
plt.plot([1,20],[1,1],color=colors[1],ls='dashed',lw=lws,alpha=alf2)
plt.ylabel('$\sigma_{hardest}$/$\sigma_{default}$')
plt.ylim([ll,ul])
plt.xlim([lowm,upm])
plt.annotate("(b)", xy=(0.85, 0.15), xycoords="axes fraction",fontsize=22,color='black')

plt.subplot(3,1,3)
plt.plot(dat['tmags'],dat2['fast_2min']/dat3['fast_2min'],'.',color=colors[3],alpha=alf)
binx,biny,binz=bin_set(dat['tmags'],dat2['fast_2min']/dat3['fast_2min'],bs)
#plt.errorbar(binx,biny,yerr=binz,fmt='-o',color='darkorange',ms=10)
plt.plot(binx,biny,'-o',color=colors[0],ms=10)
plt.xlim([lowm,upm])
plt.ylim([ll,ul])
plt.xlabel('TESS Magnitude')
plt.plot([1,20],[1,1],color=colors[1],ls='dashed',lw=lws,alpha=alf2)
plt.ylabel('$\sigma_{hard}$/$\sigma_{default}$')
plt.annotate("(c)", xy=(0.85, 0.15), xycoords="axes fraction",fontsize=22,color='black')

#plt.savefig('fig-compscatter-qflags-20sec.png',dpi=150)

plt.subplots_adjust(wspace=0.2,hspace=0.0,left=0.14,right=0.98,bottom=0.09,top=0.98)
#plt.tight_layout(pad=0.4,h_pad=0.0)
#plt.savefig('plots_paper/fig4.png',dpi=150)



plt.ion()
plt.clf()
plt.subplot(3,1,1)
plt.plot(dat['tmags'],dat['slow_2min']/dat2['slow_2min'],'.',color='grey',alpha=0.5)
binx,biny,binz=bin_set(dat['tmags'],dat['slow_2min']/dat2['slow_2min'],bs)
#plt.errorbar(binx,biny,yerr=binz,fmt='-o',color='darkorange',ms=10)
plt.plot(binx,biny,'-o',color='black',ms=10)
plt.ylim([0,2])
plt.plot([1,20],[1,1],color='red',ls='dashed')
plt.ylabel('$\sigma_{hardest}$/$\sigma_{hard}$')
plt.ylim([ll,ul])
plt.xlim([3,19])

plt.subplot(3,1,2)
plt.plot(dat['tmags'],dat['slow_2min']/dat3['slow_2min'],'.',color='grey',alpha=0.5)
binx,biny,binz=bin_set(dat['tmags'],dat['slow_2min']/dat3['slow_2min'],bs)
#plt.errorbar(binx,biny,yerr=binz,fmt='-o',color='darkorange',ms=10)
plt.plot(binx,biny,'-o',color='black',ms=10)
plt.ylim([0,2])
plt.plot([1,20],[1,1],color='red',ls='dashed')
plt.ylabel('$\sigma_{hardest}$/$\sigma_{default}$')
plt.ylim([ll,ul])
plt.xlim([3,19])

plt.subplot(3,1,3)
plt.plot(dat['tmags'],dat2['slow_2min']/dat3['slow_2min'],'.',color='grey',alpha=0.5)
binx,biny,binz=bin_set(dat['tmags'],dat2['slow_2min']/dat3['slow_2min'],bs)
#plt.errorbar(binx,biny,yerr=binz,fmt='-o',color='darkorange',ms=10)
plt.plot(binx,biny,'-o',color='black',ms=10)
plt.xlim([3,19])
plt.ylim([ll,ul])
plt.xlabel('TESS Magnitude')
plt.plot([1,20],[1,1],color='red',ls='dashed')
plt.ylabel('$\sigma_{hard}$/$\sigma_{default}$')

#plt.savefig('fig-compscatter-qflags-2min.png',dpi=150)



dat=ascii.read('data/scatter/scatter-20sec-qflags.csv')
dat2=ascii.read('data/scatter/scatter-20secto2min-qflags.csv')

plt.clf()
plt.plot(dat['tmags'],dat['hardest']/dat2['hardest'],'.',label='hardest')
#plt.plot(dat['tmags'],dat['hard']/dat2['hard'],'.',label='hard')
#plt.plot(dat['tmags'],dat['default']/dat2['default'],'.',label='default')
plt.legend()
plt.ylim([2,2.8])


ran=np.random.randn(1000)
ran2=np.random.randint(0,1000,900)

print(np.std(ran),np.std(ran[ran2]))

'''
