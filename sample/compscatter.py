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

	time1=data1['TIME']
	flux1=data1['PDCSAP_FLUX']
	qflag1=data1['QUALITY']
	s=np.argsort(time1)
	time1=time1[s]
	flux1=flux1[s]
	qflag1=qflag1[s]
	
	time2=data2['TIME']
	flux2=data2['PDCSAP_FLUX']
	qflag2=data2['QUALITY']
	s=np.argsort(time2)
	time2=time2[s]
	flux2=flux2[s]
	qflag2=qflag2[s]
	

	'''
	qf=np.unique(qflag1)
	for r in range(0,len(qf)):
		tmp1=np.where(qflag1 == qf[r])[0]
		tmp2=np.where(qflag2 == qf[r])[0]
		print(qf[r],len(tmp1),len(tmp2))
	pdb.set_trace()
	'''
	
	# bitmasks as defined in lightkurve
	#bitmask=175 		# default
	#bitmask=1263 		# default + CR flags only
	#bitmask = 3311		# hard
	bitmask = 4095		# hardest 
	
	
	quality_mask = (qflag2 & bitmask) == 0
	um=np.where((quality_mask == True) & (np.isfinite(flux2)))[0]
	
	# hardest bitmask in lightkurve
	um=np.where((qflag2 == 0)  & (np.isfinite(flux2)))[0]
	
	# hardest plus cosmic ray flags
	#um=np.where(((qflag2 == 0) | (qflag2 == 64) | (qflag2 == 1024))  & (np.isfinite(flux2)))[0]

	# hard bitmask in lightkurve
	#um=np.where((qflag2 != 1) & (qflag2 != 2) & (qflag2 != 4) & (qflag2 != 8) & (qflag2 != 32) & (qflag2 != 128) & (qflag2 != 64) & (qflag2 != 1024) & (qflag2 != 2048)  & (np.isfinite(flux2)))[0]
	
	# default bitmask in lightkurve
	#um=np.where((qflag2 != 1) & (qflag2 != 2) & (qflag2 != 4) & (qflag2 != 8) & (qflag2 != 32) & (qflag2 != 128)  & (np.isfinite(flux2)))[0]


	time2=time2[um]
	flux2=flux2[um]/np.median(flux2[um])
	#print(np.unique(qflag2))
	#print(np.unique(qflag2[um]))
	#print(' ')

	quality_mask = (qflag1 & bitmask) == 0
	um=np.where((quality_mask == True) & (np.isfinite(flux1)))[0]

	# hardest bitmask in lightkurve	
	um=np.where((qflag1 == 0)  & (np.isfinite(flux1)))[0]
	
	# hardest plus cosmic ray flags	
	#um=np.where(((qflag1 == 0) | (qflag1 == 64) | (qflag1 == 1024))  & (np.isfinite(flux1)))[0]

	# hard bitmask in lightkurve
	#um=np.where((qflag1 != 1) & (qflag1 != 2) & (qflag1 != 4) & (qflag1 != 8) & (qflag1 != 32) & (qflag1 != 128) & (qflag1 != 64) & (qflag1 != 1024) & (qflag1 != 2048)  & (np.isfinite(flux1)))[0]

	# default bitmask in lightkurve
	#um=np.where((qflag1 != 1) & (qflag1 != 2) & (qflag1 != 4) & (qflag1 != 8) & (qflag1 != 32) & (qflag1 != 128)  & (np.isfinite(flux1)))[0]

	time1=time1[um]
	flux1=flux1[um]/np.median(flux1[um])
	#print(np.unique(qflag1))
	#print(np.unique(qflag1[um]))
	
	#input(':')
		
	#print(' ')
	#print('total number of 20-sec data:',len(flux1))
	#print('total number of 2-min data:',len(flux2))
	#print('ratio (20sec/2min):',np.float(len(flux1))/np.float(len(flux2)))	
		
	# 20-sec to 2-min
	time20sto2m,flux20sto2m,resz=bin.bin_time_digitized(time1,flux1,2./60./24.)
	
	# 20-sec to 1hr
	time20sto1h,flux20sto1h,resz2=bin.bin_time_digitized(time1,flux1,1./24.)
	
	# 2-min 
	time2m=time2
	flux2m=flux2
	
	# 2-min to 1hr	
	time2mto1h,flux2mto1h,resz3=bin.bin_time_digitized(time2,flux2,1./24.)

	df1=np.median(time20sto2m[1::]-time20sto2m[0:-1])*24.*60.
	df2=np.median(time20sto1h[1::]-time20sto1h[0:-1])*24.*60.
	df3=np.median(time2m[1::]-time2m[0:-1])*24.*60.
	df4=np.median(time2mto1h[1::]-time2mto1h[0:-1])*24.*60.
	
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
		
	smoothed_flux1 = savgol(flux20sto2m,boxsize1,1,mode='mirror')
	smoothed_flux2 = savgol(flux20sto1h,boxsize2,1,mode='mirror')
	smoothed_flux3 = savgol(flux2m,boxsize3,1,mode='mirror')
	smoothed_flux4 = savgol(flux2mto1h,boxsize4,1,mode='mirror')
	
	flux20sto2ms=flux20sto2m/smoothed_flux1
	flux20sto1hs=flux20sto1h/smoothed_flux2
	flux2ms=flux2m/smoothed_flux3
	flux2mto1hs=flux2mto1h/smoothed_flux4

	'''
	plt.ion()
	plt.clf()
	
	plt.subplot(5,2,1)
	plt.plot(time2m,flux2m,'.')
	plt.plot(time2m,smoothed_flux3,'.')
	plt.title('2min to 2min')
	plt.subplot(5,2,2)
	plt.plot(time2m,flux2ms,'.')
	plt.title('2min to 2min')
	
	plt.subplot(5,2,3)
	plt.plot(time2m,flux2m,'.')
	plt.plot(time2mto1h,flux2mto1h,'.')
	plt.plot(time2mto1h,smoothed_flux4,'.')
	plt.title('2min to 1hr')

	plt.subplot(5,2,4)
	plt.plot(time2mto1h,flux2mto1hs,'.')
	plt.title('2min to 1hr')

	plt.subplot(5,2,5)
	plt.plot(time1,flux1,'.')
	plt.plot(time20sto2m,flux20sto2m,'.')
	plt.plot(time20sto2m,smoothed_flux1,'.')
	plt.title('20sec to 2min')

	plt.subplot(5,2,6)
	plt.plot(time20sto2m,flux20sto2ms,'.')
	plt.title('20sec to 2min')

	plt.subplot(5,2,7)
	plt.plot(time1,flux1,'.')
	plt.plot(time20sto1h,flux20sto1h,'.')
	plt.plot(time20sto1h,smoothed_flux2,'.')
	plt.title('20sec to 1hr')

	plt.subplot(5,2,8)
	plt.plot(time20sto1h,flux20sto1hs,'.')
	plt.title('20sec to 1hr')

	plt.subplot(5,2,9)
	plt.plot(time2m,flux2ms,'.',label='2min')
	plt.plot(time20sto2m,flux20sto2ms,'.',label='20sec')
	plt.title('binned to 2min')
	plt.legend()

	plt.subplot(5,2,10)
	plt.plot(time2mto1h,flux2mto1hs,'.',label='2min')
	plt.plot(time20sto1h,flux20sto1hs,'.',label='20sec')
	plt.title('binned to 1hr')
	plt.legend()

	plt.tight_layout()
		
	plt.draw()
	plt.show()
	input(':')
	'''
	
	return np.std(flux20sto2ms)*1e6,np.std(flux20sto1hs)*1e6,np.std(flux2ms)*1e6,np.std(flux2mto1hs)*1e6
	#return mad_std(flux20sto2ms)*1e6,mad_std(flux20sto1hs)*1e6,mad_std(flux2ms)*1e6,mad_std(flux2mto1hs)*1e6




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

slow_2min=[]
slow_30min=[]
fast_2min=[]
fast_30min=[]
		
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
	
		slow_2min=np.append(slow_2min,rms3)
		slow_30min=np.append(slow_30min,rms4)

		fast_2min=np.append(fast_2min,rms1)
		fast_30min=np.append(fast_30min,rms2)
		
		print(' ')
		print('2min data scatter in 2min cadence:',rms3)
		print('20sec data scatter in 2min cadence:',rms1)
		print(' ')
		print('2min data scatter in 30min cadence:',rms4)
		print('20sec data scatter in 30min cadence:',rms2)
	
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

ascii.write([np.asarray(tics,dtype='int'),tmags,teffs,rads,secs,slow_2min,slow_30min,fast_2min,fast_30min],'scatter-all-fullsecs.csv',names=['ticids','tmags','teff','rad','sec','slow_2min','slow_1hr','fast_2min','fast_1hr'],delimiter=',',formats={'ticids':'%i', 'tmags':'%8.3f', 'teff':'%8.0f', 'rad':'%8.3f', 'sec':'%i', 'slow_2min':'%8.3f', 'slow_1hr':'%8.3f', 'fast_2min':'%8.3f', 'fast_1hr':'%8.3f'})

	
plt.ion()
plt.clf()

plt.subplot(2,1,1)
um=np.where(tmags > 0.)[0]
plt.plot(tmags[um],fast_2min[um],'.',label='20-sec cadence')
plt.semilogy(tmags[um],slow_2min[um],'.',label='2-min cadence')
plt.ylabel("Scatter obver 2-min timescale (ppm)")
plt.legend()
plt.ylim([0.3,1.2])

plt.subplot(2,1,2)

plt.clf()
plt.plot(tmags[um],fast_2min[um]/slow_2min[um],'.')
plt.ylabel("2-min/20-sec scatter")
plt.ylabel("TESS mag")
plt.ylim([0.3,1.2])


