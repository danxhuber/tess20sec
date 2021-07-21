import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams['xtick.major.size'] = 9
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['xtick.minor.size'] = 6
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.major.size'] = 9
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['ytick.minor.size'] = 6
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['font.size']=15
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

#dat=ascii.read('sample/scatter10.csv')
dat=ascii.read('data/scatter-all.csv')

ix,un=np.unique(dat['ticids'],return_index=True)
dat=dat[un]

plt.ion()
plt.clf()

fig = plt.figure(figsize=(6, 8))

upl=16

plt.clf()
gs = gridspec.GridSpec(2, 1)

ax0 = plt.subplot(gs[0, 0])
plt.scatter(dat['teff'],dat['rad'],c=dat['tmags'],marker='o',alpha=1., vmax=upl, cmap='cubehelix',s=12, rasterized=True)
#plt.legend(loc='best')
#plt.plot([8000,4000],[5,3.4],ls='dashed',color='royalblue')
plt.xlim([8000,2700])
plt.ylim([0.1,200])
plt.xlabel('Effective Temperature (K)')
plt.ylabel('Stellar Radius (Solar)')
plt.yscale('log')
plt.annotate("(a)", xy=(0.05, 0.1), xycoords="axes fraction",fontsize=24,color='black')
cbaxes = inset_axes(ax0, width="40%", height="5%", loc=2) 
plt.colorbar(cax=cbaxes, orientation='horizontal', label='Tmag')

ax1 = plt.subplot(gs[1, 0])
um=np.where(dat['teff'] < 8000.)[0]
plt.semilogy(dat['tmags'][um],dat['rad'][um],'.',color=colors[3], rasterized=True)
plt.xlabel('TESS Magnitude')
plt.ylabel('Stellar Radius (Solar)')
plt.xlim([2,16])
plt.annotate("(b)", xy=(0.05, 0.82), xycoords="axes fraction",fontsize=24,color='black')

plt.subplots_adjust(wspace=0.20,hspace=0.26,left=0.155,right=0.97,bottom=0.08,top=0.98)

plt.savefig('fig1.png',dpi=150)


#plt.savefig('fig-hrd-all-v2.png',dpi=150)

