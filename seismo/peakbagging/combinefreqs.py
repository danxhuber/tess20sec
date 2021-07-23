import numpy as np
import os, sys
from astropy.io import ascii, fits
import matplotlib.pyplot as plt
import glob, pdb

star='pimen'
#star='gammapav'
#star='zetatuc'

if (star == 'gammapav'):
	# gamma Pav = TIC265488188
	files1=glob.glob("*ed/*/*265488188*")
	files2=glob.glob("*ed/*265488188*")
	files3=glob.glob("*ed/*gamma*")

if (star == 'zetatuc'):
	# zeta Tuc = TIC425935521
	files1=glob.glob("*ed/*/*425935521*")
	files2=glob.glob("*ed/*425935521*")
	files3=glob.glob("*ed/*zeta*")

if (star == 'pimen'):
	# pi Men = TIC261136679
	files1=glob.glob("*ed/*/*261136679*")
	files2=glob.glob("*ed/*261136679*")
	files3=glob.glob("*ed/*pi*")

files=np.append(files1,files2)
files=np.append(files,files3)

fs=[]
fse=[]

fd=ascii.read(files[-1])

for i in range(0,len(files)):
	dat=ascii.read(files[i])
	fs=np.append(fs,dat['col1'])
	fse=np.append(fse,dat['col2'])
	
	
vals=np.zeros(len(fd))
err=np.zeros(len(fd))
ell=np.zeros(len(fd))

for i in range(0,len(fd)):
	um=np.where(np.abs(fd['col1'][i]-fs) < 3.)[0]
	
	#print(fd['col2'][i],np.std(fs[um]))
	#err[i]=np.sqrt(fd['col2'][i]**2 + np.std(fs[um]))
	
	err[i]=np.sqrt(np.median(fse[um])**2 + np.std(fs[um])**2)
	vals[i]=np.median(fs[um])	# use the median over all estimates
	vals[i]=fd['col1'][i]		# use Hans' values
	ell[i]=np.int(fd['col3'][i])
	
	'''
	plt.clf()
	#plt.plot(ps['col1'],ps['col2'],color='grey')
	plt.plot(fs,np.zeros(len(fs))+10,'o')
	plt.plot(fs[um],np.zeros(len(fs[um])),'o',color='red')
	plt.xlim([fd['col1'][i]-5,fd['col1'][i]+5])
	plt.show()
	plt.draw()
	'''

	print(fd['col1'][i])
	print(fs[um])
	print(vals[i],err[i])
	#input(':')
	#input(':')

um=np.where(ell == 1)[0]
f=np.polyfit(np.arange(len(um)),vals[um],1)
print(star)
print('dnu:',f[0])

ascii.write([vals,err,ell],star+'-freq.csv',names=['freq','err','l'],formats={'freq': '%12.4f','err': '%12.4f','l': '%1i'},delimiter=',')

#ascii.write([vals,err,ell],'freq-zetatuc.csv',names=['freq','err','l'],formats={'freq': '%12.4f','err': '%12.4f','l': '%1i'},delimiter=',')
#ascii.write([vals,err,ell],'freq-gammapav.csv',names=['freq','err','l'],formats={'freq': '%12.4f','err': '%12.4f','l': '%1i'},delimiter=',')