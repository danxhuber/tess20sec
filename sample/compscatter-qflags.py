import numpy as np
import os, sys
from astropy.io import ascii, fits
import matplotlib.pyplot as plt
import pandas as pd
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel
import glob, pdb
import bin
from scipy.signal import savgol_filter as savgol
from astropy.stats import mad_std



def getnoise(data1,data2):

	# data1=20sec
	# data2=2min

	time=data2['TIME']
	flux=data2['PDCSAP_FLUX']
	qflag=data2['QUALITY']
	s=np.argsort(time)
	time=time[s]
	flux=flux[s]
	qflag=qflag[s]
	


	'''
	qf=np.unique(qflag1)
	for r in range(0,len(qf)):
		tmp1=np.where(qflag1 == qf[r])[0]
		tmp2=np.where(qflag2 == qf[r])[0]
		print(qf[r],len(tmp1),len(tmp2))
	pdb.set_trace()
	'''
	
	# bitmasks as defined in lightkurve
	default=175 		# default
	default=4271		# default + ScatteredLight
	
	defaultpluscr=1263 	# default + CR flags only
	
	hard = 3311		# hard
	#hard = 5359		# hard but ScatteredLight instead of Straylight
	#bitmask = 4095		# hardest 
	
	quality_mask = (qflag & default) == 0
	um=np.where((quality_mask == True) & (np.isfinite(flux)))[0]
	time1=time[um]
	flux1=flux[um]/np.median(flux[um])
	print(np.unique(qflag[um]))
	
	quality_mask = (qflag & defaultpluscr) == 0
	um=np.where((quality_mask == True) & (np.isfinite(flux)))[0]
	time2=time[um]
	flux2=flux[um]/np.median(flux[um])
	print(np.unique(qflag[um]))

	quality_mask = (qflag & hard) == 0
	um=np.where((quality_mask == True) & (np.isfinite(flux)))[0]
	time3=time[um]
	flux3=flux[um]/np.median(flux[um])
	print(np.unique(qflag[um]))
	
	um=np.where((qflag == 0) & (np.isfinite(flux)))[0]
	time4=time[um]
	flux4=flux[um]/np.median(flux[um])
	#pdb.set_trace()

	print(len(time1),len(time2),len(time3),len(time4))
	
	test=np.where(qflag == 4096)[0]
	print('q for scattered light:',len(test))
	test=np.where((qflag == 4096) & (np.isnan(flux)))[0]
	print('q for scattered light & NaN:',len(test))

	'''
	plt.ion()
	plt.clf()
	plt.plot(time1,flux1,'.',label='default')
	plt.plot(time2,flux2,'.',label='defaultpluscr')
	plt.plot(time3,flux3,'.',label='hard')
	plt.plot(time4,flux4,'.',label='hardest')
	plt.legend()
	plt.title('Tmag:'+str(tmag)[0:4])
	plt.tight_layout()
	input(':')
	'''
		
	#time1,flux1,resz=bin.bin_time_digitized(time1,flux1,2./60./24.)
	#time2,flux2,resz=bin.bin_time_digitized(time2,flux2,2./60./24.)
	#time3,flux3,resz=bin.bin_time_digitized(time3,flux3,2./60./24.)
	#time4,flux4,resz=bin.bin_time_digitized(time4,flux4,2./60./24.)
	
	df1=np.median(time1[1::]-time1[0:-1])*24.*60.
	df2=np.median(time2[1::]-time2[0:-1])*24.*60.
	df3=np.median(time3[1::]-time3[0:-1])*24.*60.
	df4=np.median(time4[1::]-time4[0:-1])*24.*60.
	print(df1,df2,df3,df4)

	hp=0.5
	boxsize1=hp/(df1/60./24.)
	boxsize2=hp/(df2/60./24.)
	boxsize3=hp/(df3/60./24.)
	boxsize4=hp/(df4/60./24.)
	
	if (int(boxsize1) % 2 == 0):
		boxsize1=int(boxsize1)+1
	else:
		boxsize1=int(boxsize1)
		
	if (int(boxsize2) % 2 == 0):
		boxsize2=int(boxsize2)+1
	else:
		boxsize2=int(boxsize2)		

	if (int(boxsize3) % 2 == 0):
		boxsize3=int(boxsize3)+1
	else:
		boxsize3=int(boxsize3)
		
	if (int(boxsize4) % 2 == 0):
		boxsize4=int(boxsize4)+1
	else:
		boxsize4=int(boxsize4)	

	
	smoothed_flux1 = savgol(flux1,boxsize1,1,mode='mirror')
	smoothed_flux2 = savgol(flux2,boxsize2,1,mode='mirror')
	smoothed_flux3 = savgol(flux3,boxsize3,1,mode='mirror')
	smoothed_flux4 = savgol(flux4,boxsize4,1,mode='mirror')
	
	flux1s=flux1/smoothed_flux1
	flux2s=flux2/smoothed_flux2
	flux3s=flux3/smoothed_flux3
	flux4s=flux4/smoothed_flux4
	
	'''
	plt.ion()
	plt.clf()
	
	plt.subplot(2,2,1)
	plt.plot(time1,flux1s,'.',label='default')
	plt.legend()

	plt.subplot(2,2,2)
	plt.plot(time2,flux2s,'.',label='default+crs')
	plt.legend()

	plt.subplot(2,2,3)
	plt.plot(time3,flux3s,'.',label='hard')
	plt.legend()

	plt.subplot(2,2,4)
	plt.plot(time4,flux4s,'.',label='hardest')
	plt.legend()

	
	plt.tight_layout()
		
	plt.draw()
	plt.show()
	'''
	
	print('default:',np.std(flux1s)*1e6,len(flux1s))
	print('default+CR:',np.std(flux2s)*1e6,len(flux2s))
	print('hard:',np.std(flux3s)*1e6,len(flux3s))
	print('hardest:',np.std(flux4s)*1e6,len(flux4s))
	print('')
	#input(':')
	
	return np.std(flux1s)*1e6,np.std(flux2s)*1e6,np.std(flux3s)*1e6,np.std(flux4s)*1e6




dat=ascii.read('../MAST_Advanced_Search_1.csv') 
ids=dat['target_name'] 
s=np.argsort(ids)
uids=ids[s]
ticids=np.unique(uids)

tics=[]

tmags=[]
teffs=[]
rads=[]
secs=[]

default=[]
defaultcrs=[]
hard=[]
hardest=[]
		
ntot=0
for i in range(0,len(ticids)):

	print(ticids[i],i,len(ticids))

	#if (ticids[i] != 141186075):
	#	continue

	fast=glob.glob('../mastDownload/*/*/*s00*'+str(ticids[i])+'*fast*')
	slow=glob.glob('../mastDownload/*/*/*s00*'+str(ticids[i])+'*s_lc*')
	
	secs_fast=np.zeros(len(fast),dtype='int')
	secs_slow=np.zeros(len(slow),dtype='int')
	
	for r in range(0,len(secs_fast)):
		tmp=fast[r].split('-')[1]
		secs_fast[r]=np.float(tmp[1:5])
	for r in range(0,len(secs_slow)):
		tmp=slow[r].split('-')[1]
		secs_slow[r]=np.float(tmp[1:5])

	#if (len(fast) != len(slow)):
	#	pdb.set_trace()
	#	continue
		
	#if ((len(fast) > 1) | (len(slow) > 1)):
	#	fast=np.sort(fast)
	#	slow=np.sort(slow)
	
	#if ((len(fast) < 2) | (len(slow) < 2)):
	#	continue
	
	print(secs_fast)
	print(secs_slow)	
	
	#input(':')
	for j in range(0,len(fast)):
		try:
			print('sector',secs_fast[j])
			um=np.where(secs_fast == secs_fast[j])[0]
			x1=fits.open(fast[um[0]])
			um=np.where(secs_slow == secs_fast[j])[0]
			x2=fits.open(slow[um[0]])
		except:
			continue
		data1=x1[1].data
		data2=x2[1].data
	
		header=x1[0].header
		tmags=np.append(tmags,np.float(header['TESSMAG']))
		if ((x1[0].header['TEFF'] == None) | (x1[0].header['RADIUS'] == None)):
			teffs=np.append(teffs,0.)
			rads=np.append(rads,0.)
		else:
			teffs=np.append(teffs,np.float(x1[0].header['TEFF']))
			rads=np.append(rads,np.float(x1[0].header['RADIUS']))
		secs=np.append(secs,np.float(x1[0].header['SECTOR']))
		tics=np.append(tics,ticids[i])
	
		#pdb.set_trace()
		#print('Tmag/teff/rad/sec:',tmags[i],teffs[i],rads[i],secs[i])
		
		rms1,rms2,rms3,rms4=getnoise(data1,data2)
	
		default=np.append(default,rms1)
		defaultcrs=np.append(defaultcrs,rms2)
		hard=np.append(hard,rms3)
		hardest=np.append(hardest,rms4)

		#input(':')

	#continue
	
	
	'''
	# sigma-clip outliers from the light curve and overplot it
	res=sigclip(time,flux,300,4)
	good = np.where(res == 1)[0]
	time=time[good]
	flux=flux[good]
	ax.plot(time,flux)


	# next, run a filter through the data to remove long-periodic (low frequency) variations
	# let's pick a 5 day width for the filter
	width=1.0
	boxsize=width/(2./60./24.)
	box_kernel = Box1DKernel(boxsize)
	smoothed_flux = savgol(flux,int(boxsize)-1,1,mode='mirror')
	# overplot this smoothed version, and then divide the light curve through it
	ax.plot(time,smoothed_flux)
	ax.set_title(str1)
	flux=flux/(smoothed_flux)
	'''

ascii.write([np.asarray(tics,dtype='int'),tmags,teffs,rads,secs,default,defaultcrs,hard,hardest],'scatter-2min-qflags-fullsecs.csv',names=['ticids','tmags','teff','rad','sec','default','defaultcrs','hard','hardest'],delimiter=',',formats={'ticids':'%i', 'tmags':'%8.3f', 'teff':'%8.0f', 'rad':'%8.3f', 'sec':'%i', 'default':'%8.3f', 'defaultcrs':'%8.3f', 'hard':'%8.3f', 'hardest':'%8.3f'})



