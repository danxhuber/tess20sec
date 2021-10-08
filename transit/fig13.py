import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii
from astropy import constants as c
from matplotlib.ticker import FormatStrFormatter

plt.rcParams['xtick.major.size'] = 9
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['xtick.minor.size'] = 6
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.major.size'] = 9
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['ytick.minor.size'] = 6
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['font.size']=18
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

dat=ascii.read('data/transit/vaneylen18.txt')

p=6.26789
rp=2.131

m=-0.09
a=0.37
parr=np.arange(1.,100.,0.1)
mody=m*np.log10(parr)+a

mody2=m*np.log10(parr)+0.41
mody3=m*np.log10(parr)+0.33
#mody3=m*np.log10(parr)+0.35
#0.41
#0.35
 
fig = plt.figure(figsize=(8, 5.5))

plt.ion()
plt.clf()
plt.errorbar(dat['p'],dat['rp'],fmt='.',yerr=dat['rpe'],color='grey')
plt.plot(dat['p'],dat['rp'],'o',color='grey')

plt.fill_between(parr,10**mody2,10**mody3,alpha=0.3,label='Seismic Radius Valley',color=colors[0])

plt.plot(p,rp,'^',color=colors[1],label='$\pi$ Men c (This Work)')
#plt.plot(parr,10**mody,lw=35,alpha=0.5)

plt.fill_between([p-0.2,p+0.25],[1.98,1.98],[2.19,2.19],alpha=0.5,color=colors[3],
	label='$\pi$ Men c (Literature)')
plt.legend(loc='lower right',fontsize=14)


plt.xscale('log')    
plt.yscale('log')      
plt.xlim([2,80])
plt.ylim([1,3])

plt.xlabel("Orbital Period (days)")
plt.ylabel("Planet Radius (Earth Radii)")
plt.tight_layout()
plt.yticks([1,2,3],labels=[1,2,3])
plt.xticks([2,10,80],labels=[2,10,80])

plt.tight_layout(pad=0.4)

plt.savefig('fig13.png',dpi=200)
















