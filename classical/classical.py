import numpy as np
from astropy.io import ascii
import astropy.constants as c

def calclogg(numax,teff):
	gsun = (c.G.cgs.value*c.M_sun.cgs.value/c.R_sun.cgs.value**2)
	return np.log10(gsun  * (numax/3100.) * (teff/5777.)**(0.5))

systefferr=0.02
sysfeherr=0.062

dat=ascii.read('gammapav.txt')
x,y=np.unique(dat['Ref'],return_index=True) 
dat=dat[y]
x,y=np.unique(dat['Teff'],return_index=True) 
dat=dat[y]
um=np.where(dat['Ref'] == '2018A&A...614A..55A')[0]
print('gamma Pav:')
print('# results',len(dat))
teffe=np.sqrt((systefferr*dat['Teff'][um[0]])**2 + np.std(dat['Teff'])**2)
#teffe=np.sqrt(50.**2 + np.std(dat['Teff'])**2)
teffe=np.sqrt(50.**2 + (systefferr*dat['Teff'][um[0]])**2)
fehe=np.sqrt((np.std(dat['Fe_H']))**2 + sysfeherr**2)
print('teff:',dat['Teff'][um[0]],np.std(dat['Teff']),teffe)
print('feh:',dat['Fe_H'][um[0]],np.std(dat['Fe_H']),fehe)
teff=dat['Teff'][um[0]]
numax=2717.
print('logg:',calclogg(numax,teff))
print('median teff:',np.median(dat['Teff']),np.std(dat['Teff']))
print('median feh:',np.median(dat['Fe_H']),np.std(dat['Fe_H']))

# alpha elements from Bensby 2005
feh=-0.73
mgh=-0.51
cah=-0.63
tih=-0.67
sih=-0.58
afeh=np.mean([mgh,cah,tih,sih])-feh
print('[alpha/fe]:',np.mean([mgh,cah,tih,sih])-feh,np.std(np.array((mgh,cah,tih,sih))-feh))

feh=dat['Fe_H'][um[0]]
fa = 10**afeh
mh = feh + np.log10(0.694*fa+0.306)
print('[M/H]',mh)


print(' ')
dat=ascii.read('zetaTuc.txt')
x,y=np.unique(dat['Ref'],return_index=True) 
dat=dat[y]
x,y=np.unique(dat['Teff'],return_index=True) 
dat=dat[y]
um=np.where(dat['Ref'] == '2018A&A...614A..55A')[0]
print('zeta Tuc:')
print('# results',len(dat))
teffe=np.sqrt((systefferr*dat['Teff'][um[0]])**2 + np.std(dat['Teff'])**2)
#teffe=np.sqrt(50.**2 + np.std(dat['Teff'])**2)
teffe=np.sqrt(50.**2 + (systefferr*dat['Teff'][um[0]])**2)
fehe=np.sqrt((np.std(dat['Fe_H']))**2 + sysfeherr**2)
print('teff:',dat['Teff'][um[0]],np.std(dat['Teff']),teffe)
print('feh:',dat['Fe_H'][um[0]],np.std(dat['Fe_H']),fehe)
teff=dat['Teff'][um[0]]
numax=2717.
print('logg:',calclogg(numax,teff))
print('median teff:',np.median(dat['Teff']),np.std(dat['Teff']))
print('median feh:',np.median(dat['Fe_H']),np.std(dat['Fe_H']))

print(' ')
dat=ascii.read('pimen.txt')
x,y=np.unique(dat['Ref'],return_index=True) 
dat=dat[y]
x,y=np.unique(dat['Teff'],return_index=True) 
dat=dat[y]
um=np.where(dat['Ref'] == '2018A&A...614A..55A')[0]
print('pi Men:')
print('# results',len(dat))
teffe=np.sqrt((systefferr*dat['Teff'][um[0]])**2 + np.std(dat['Teff'])**2)
#teffe=np.sqrt(50.**2 + np.std(dat['Teff'])**2)
teffe=np.sqrt(50.**2 + (systefferr*dat['Teff'][um[0]])**2)
fehe=np.sqrt((np.std(dat['Fe_H']))**2 + sysfeherr**2)
print('teff:',dat['Teff'][um[0]],np.std(dat['Teff']),teffe)
print('feh:',dat['Fe_H'][um[0]],np.std(dat['Fe_H']),fehe)
teff=dat['Teff'][um[0]]
numax=2717.
print('logg:',calclogg(numax,teff))
print('median teff:',np.median(dat['Teff']),np.std(dat['Teff']))
print('median feh:',np.median(dat['Fe_H']),np.std(dat['Fe_H']))
print('  ')

def calcl(fbol,fbole,plx,plxe):
	dis=1./plx
	dise=dis*plxe/plx
	pctocm=3.08567758128e18
	lsun=3.839e33
	dis=dis*pctocm
	dise=dise*pctocm
	lum1=4.*np.pi*dis**2*fbol
	lum1e=lum1*np.sqrt( (2.*dise/dis)**2 + (fbole/fbol)**2)
	return lum1/lsun,lum1e/lsun


# fbols (isocl-vtmag,isocl-btmag,stassun,casagrande)
fbols_gammapav=[5.450754925810345e-07,5.344506406968792e-07,5.489e-7,5.6263e-07]
fbolse_gammapav=[1.5097059864067728e-08,1.5754859794290815e-08,0.064e-7]
plx_gammapav=0.10801
plxe_gammapav=0.000106
fbol=np.median(fbols_gammapav)
fbole=np.sqrt( np.median(fbolse_gammapav)**2 + np.std(fbols_gammapav)**2)
lum,lume=calcl(fbol,fbole,plx_gammapav,plxe_gammapav)
print('gamma pav (fbol + Lum):',fbol,fbole,lum,lume)

fbols_zetatuc=[5.317557552178402e-07,5.301920773061478e-07,5.356e-7,5.5084e-07]
fbolse_zetatuc=[1.4728140558242096e-08,1.5629323282521243e-08,0.062e-7]
plx_zetatuc=0.11618
plxe_zetatuc=0.000133
fbol=np.median(fbols_zetatuc)
fbole=np.sqrt( np.median(fbolse_zetatuc)**2 + np.std(fbols_zetatuc)**2)
lum,lume=calcl(fbol,fbole,plx_zetatuc,plxe_zetatuc)
print('zeta tuc (fbol + Lum):',fbol,fbole,lum,lume)

fbols_pimen=[1.3814969038627393e-07,1.3434862807975348e-07,1.437e-7,1.4544e-07]
fbolse_pimen=[3.8263583198136465e-09,3.960410256393189e-09,0.017e-7]
plx_pimen=0.054683
plxe_pimen=0.000035
fbol=np.median(fbols_pimen)
fbole=np.sqrt( np.median(fbolse_pimen)**2 + np.std(fbols_pimen)**2)
lum,lume=calcl(fbol,fbole,plx_pimen,plxe_pimen)
print('pi men (fbol + Lum):',fbol,fbole,lum,lume)


