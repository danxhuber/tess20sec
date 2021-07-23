import numpy as np
import os, sys
from astropy.io import ascii, fits
import matplotlib.pyplot as plt
from echelle import interact_echelle 
from echelle import plot_echelle
from echelle import echelle as calc_echelle
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel
import matplotlib.gridspec as gridspec

plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.major.width'] = 3
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['font.size']=14
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


dnu_pimen=116.8
dnu_gampav=120.0
dnu_zetatuc=126.0

star='pimen'
#star='gammapav'
#star='zetatuc'

if (star == 'gammapav'):
	# gamma Pav = TIC265488188
	dnu=dnu_gampav
	dat=ascii.read('265488188_bgcorr.txt')
	hans=ascii.read('Hans-ed/gammapav.txt')
	hanso=ascii.read('Hans/gammapav.txt')
	derek=ascii.read('Derek-ed/GammaPav_freqs_Buzasi.txt')
	dereko=ascii.read('Derek/GammaPav_freqs_Buzasi.txt')
	joel_diamonds=ascii.read('Joel-ed/DIAMONDS/TIC265488188.txt')
	joelo_diamonds=ascii.read('Joel/DIAMONDS/TIC265488188.txt')
	joel_mle=ascii.read('Joel-ed/MLE/TIC265488188.txt')
	joelo_mle=ascii.read('Joel/MLE/TIC265488188.txt')
	joel_pbjam=ascii.read('Joel-ed/PBJam/TIC265488188.txt')
	joelo_pbjam=ascii.read('Joel/PBJam/TIC265488188.txt')
	othman=ascii.read('Othman-ed/TIC265488188_freqs_Benomar.txt')
	othmano=ascii.read('Othman/TIC265488188_freqs_Benomar.txt')
	rafa=ascii.read('Rafa/gammaPav_summary_MCMC.pkb')
	rafao=rafa

	lol=1400
	upl=3600


if (star == 'zetatuc'):
	# zeta Tuc = TIC425935521
	dnu=dnu_zetatuc
	dat=ascii.read('425935521_bgcorr.txt')
	hanso=ascii.read('Hans/zetatuc.txt')
	hans=ascii.read('Hans-ed/zetatuc.txt')
	dereko=ascii.read('Derek/ZetaTuc_freqs_Buzasi.txt')
	derek=ascii.read('Derek-ed/ZetaTuc_freqs_Buzasi.txt')
	joelo_diamonds=ascii.read('Joel/DIAMONDS/TIC425935521.txt')
	joel_diamonds=ascii.read('Joel-ed/DIAMONDS/TIC425935521.txt')
	joelo_mle=ascii.read('Joel/MLE/TIC425935521.txt')
	joel_mle=ascii.read('Joel-ed/MLE/TIC425935521.txt')
	joelo_pbjam=ascii.read('Joel/PBJam/TIC425935521.txt')
	joel_pbjam=ascii.read('Joel-ed/PBJam/TIC425935521.txt')
	othmano=ascii.read('Othman/TIC425935521_freqs_Benomar.txt')
	othman=ascii.read('Othman-ed/TIC425935521_freqs_Benomar.txt')
	rafao=ascii.read('Rafa/zetaTuc_summary_MCMC.pkb')
	rafa=ascii.read('Rafa-ed/zetaTuc_summary_MCMC.pkb')

	lol=1750
	upl=3700


if (star == 'pimen'):
	# pi Men = TIC261136679
	dnu=dnu_pimen
	dat=ascii.read('261136679_bgcorr.txt')
	hans=ascii.read('Hans-ed/pimen.txt')
	hanso=ascii.read('Hans/pimen.txt')
	derek=ascii.read('Derek-ed/PiMen_freqs_Buzasi.txt')
	dereko=ascii.read('Derek/PiMen_freqs_Buzasi.txt')
	joel_diamonds=ascii.read('Joel-ed/DIAMONDS/TIC261136679.txt')
	joelo_diamonds=ascii.read('Joel/DIAMONDS/TIC261136679.txt')
	joel_mle=ascii.read('Joel-ed/MLE/TIC261136679.txt')
	joelo_mle=ascii.read('Joel/MLE/TIC261136679.txt')
	joel_pbjam=ascii.read('Joel-ed/PBJam/TIC261136679.txt')
	joelo_pbjam=ascii.read('Joel/PBJam/TIC261136679.txt')
	othman=ascii.read('Othman-ed/TIC261136679_freqs_Benomar.txt')
	othmano=ascii.read('Othman/TIC261136679_freqs_Benomar.txt')
	rafa=ascii.read('Rafa-ed/piMen_summary_MCMC.pkb')
	rafao=ascii.read('Rafa/piMen_summary_MCMC.pkb')

	lol=2000
	upl=3200



freq=np.array(dat['col1'])
amp=np.array(dat['col2'])
gauss_kernel = Gaussian1DKernel(2)
pssm = convolve(amp, gauss_kernel)
echx, echy, echz=calc_echelle(freq,pssm, dnu,fmin=1000, fmax=5000)

alf=0.4

plt.ion()
plt.clf()

plt.subplots_adjust(wspace = .001)

gs = gridspec.GridSpec(2, 7)

ax1 = plt.subplot(gs[0:1, 0:7])
ax1.plot(freq,amp,color='grey',alpha=0.5)
ax1.plot(freq,pssm,color='black',lw=2)

ax1.plot([hans['col1'][0],hans['col1'][0]],[0,15],ls='dashed',color='red',label='Kjeldsen')
for i in range(1,len(hans)):
    ax1.plot([hans['col1'][i],hans['col1'][i]],[0,15],ls='dashed',color='red',label='')
for i in range(1,len(hanso)):
    ax1.plot([hanso['col1'][i],hanso['col1'][i]],[0,15],ls='dashed',color='red',label='',alpha=alf)

ax1.plot([derek['col1'][0],derek['col1'][0]],[0,15],ls='dashed',color='green',label='Buzasi')
for i in range(1,len(derek)):
    ax1.plot([derek['col1'][i],derek['col1'][i]],[0,15],ls='dashed',color='green',label='')
for i in range(1,len(dereko)):
    ax1.plot([dereko['col1'][i],dereko['col1'][i]],[0,15],ls='dashed',color='green',label='',alpha=alf)

ax1.plot([joel_diamonds['col1'][0],joel_diamonds['col1'][0]],[0,15],ls='dashed',color='magenta',label='Ong-DIAMONDS')
for i in range(1,len(joel_diamonds)):
    ax1.plot([joel_diamonds['col1'][i],joel_diamonds['col1'][i]],[0,15],ls='dashed',color='magenta')
for i in range(1,len(joelo_diamonds)):
    ax1.plot([joelo_diamonds['col1'][i],joelo_diamonds['col1'][i]],[0,15],ls='dashed',color='magenta',alpha=alf)

ax1.plot([joel_mle['col1'][0],joel_mle['col1'][0]],[0,15],ls='dashed',color='orange',label='Ong-MLE')
for i in range(1,len(joel_mle)):
    ax1.plot([joel_mle['col1'][i],joel_mle['col1'][i]],[0,15],ls='dashed',color='orange',label='')
for i in range(1,len(joelo_mle)):
    ax1.plot([joelo_mle['col1'][i],joelo_mle['col1'][i]],[0,15],ls='dashed',color='orange',label='',alpha=alf)

ax1.plot([joel_pbjam['col1'][0],joel_pbjam['col1'][0]],[0,15],ls='dashed',color='cyan',label='Ong-PBJam')
for i in range(1,len(joel_pbjam)):
    ax1.plot([joel_pbjam['col1'][i],joel_pbjam['col1'][i]],[0,15],ls='dashed',color='cyan',label='')
for i in range(1,len(joelo_pbjam)):
    ax1.plot([joelo_pbjam['col1'][i],joelo_pbjam['col1'][i]],[0,15],ls='dashed',color='cyan',label='',alpha=alf)

ax1.plot([othman['col1'][0],othman['col1'][0]],[0,15],ls='dashed',color='grey',label='Benomar')
for i in range(1,len(othman)):
    ax1.plot([othman['col1'][i],othman['col1'][i]],[0,15],ls='dashed',color='grey',label='')
for i in range(1,len(othmano)):
    ax1.plot([othmano['col1'][i],othmano['col1'][i]],[0,15],ls='dashed',color='grey',label='',alpha=alf)    

ax1.plot([rafa['col1'][0],rafa['col1'][0]],[0,15],ls='dashed',color='blue',label='Garcia')
for i in range(1,len(rafa)):
    ax1.plot([rafa['col1'][i],rafa['col1'][i]],[0,15],ls='dashed',color='blue',label='')

ax1.legend()
ax1.set_ylim([0,15])
ax1.set_xlim([lol,upl])

ax1 = plt.subplot(gs[1:2, 0:1])
ax1.imshow(echz,aspect="auto",extent=(echx.min(), echx.max(), echy.min(), echy.max()),origin="lower",cmap='BuPu',interpolation='None')
plt.plot(hanso['col1'] % dnu,hanso['col1'],'o',color='red',fillstyle='none',alpha=alf)
plt.plot(hans['col1'] % dnu,hans['col1'],'o',color='red',fillstyle='none',lw=2)
plt.ylim([lol,upl])
ax1.set_title('Kjeldsen')

ax1 = plt.subplot(gs[1:2, 1:2])
ax1.imshow(echz,aspect="auto",extent=(echx.min(), echx.max(), echy.min(), echy.max()),origin="lower",cmap='BuPu',interpolation='None')
plt.plot(dereko['col1'] % dnu,dereko['col1'],'o',color='green',fillstyle='none',alpha=alf)
plt.plot(derek['col1'] % dnu,derek['col1'],'o',color='green',fillstyle='none',lw=2)
plt.ylim([lol,upl])
ax1.set_yticklabels('')
ax1.set_title('Buzasi')

ax1 = plt.subplot(gs[1:2, 2:3])
ax1.imshow(echz,aspect="auto",extent=(echx.min(), echx.max(), echy.min(), echy.max()),origin="lower",cmap='BuPu',interpolation='None')
plt.plot(joelo_diamonds['col1'] % dnu,joelo_diamonds['col1'],'o',color='magenta',fillstyle='none',alpha=alf)
plt.plot(joel_diamonds['col1'] % dnu,joel_diamonds['col1'],'o',color='magenta',fillstyle='none',lw=2)
plt.ylim([lol,upl])
ax1.set_yticklabels('')
ax1.set_title('Ong-Diamonds')

ax1 = plt.subplot(gs[1:2, 3:4])
ax1.imshow(echz,aspect="auto",extent=(echx.min(), echx.max(), echy.min(), echy.max()),origin="lower",cmap='BuPu',interpolation='None')
plt.ylim([lol,upl])
ax1.set_yticklabels('')
plt.plot(joelo_mle['col1'] % dnu,joelo_mle['col1'],'o',color='orange',fillstyle='none',alpha=alf)
plt.plot(joel_mle['col1'] % dnu,joel_mle['col1'],'o',color='orange',fillstyle='none',lw=2)
ax1.set_title('Ong-MLE')

ax1 = plt.subplot(gs[1:2, 4:5])
ax1.imshow(echz,aspect="auto",extent=(echx.min(), echx.max(), echy.min(), echy.max()),origin="lower",cmap='BuPu',interpolation='None')
plt.ylim([lol,upl])
ax1.set_yticklabels('')
plt.plot(joelo_pbjam['col1'] % dnu,joelo_pbjam['col1'],'o',color='cyan',fillstyle='none',alpha=alf)
plt.plot(joel_pbjam['col1'] % dnu,joel_pbjam['col1'],'o',color='cyan',fillstyle='none',lw=2)
ax1.set_title('Ong-PBJam')

ax1 = plt.subplot(gs[1:2, 5:6])
ax1.imshow(echz,aspect="auto",extent=(echx.min(), echx.max(), echy.min(), echy.max()),origin="lower",cmap='BuPu',interpolation='None')
plt.ylim([lol,upl])
ax1.set_yticklabels('')
plt.plot(othmano['col1'] % dnu,othmano['col1'],'o',color='grey',fillstyle='none',alpha=alf)
plt.plot(othman['col1'] % dnu,othman['col1'],'o',color='grey',fillstyle='none',lw=2)
ax1.set_title('Benomar')

ax1 = plt.subplot(gs[1:2, 6:7])
ax1.imshow(echz,aspect="auto",extent=(echx.min(), echx.max(), echy.min(), echy.max()),origin="lower",cmap='BuPu',interpolation='None')
plt.ylim([lol,upl])
ax1.set_yticklabels('')
plt.plot(rafao['col1'] % dnu,rafao['col1'],'o',color='blue',fillstyle='none',lw=2,alpha=alf)
plt.plot(rafa['col1'] % dnu,rafa['col1'],'o',color='blue',fillstyle='none',lw=2)
ax1.set_title('Garcia')

#plt.tight_layout()
#plt.savefig(star+'-all.png',dpi=150)

