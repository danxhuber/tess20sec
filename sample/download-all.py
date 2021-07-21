from astroquery.mast import Observations  
from astropy.io import ascii
import numpy as np
import pdb, glob
import fnmatch
import pandas as pd


# downloads all 20-sec data
dat=ascii.read('MAST_Advanced_Search_1.csv') 

tstrs=dat['obs_id'] 
sectors=np.zeros(len(dat))
for i in range(0,len(dat)):
	sectors[i]=np.float(tstrs[i].split('-')[1][3:5])      

#um=np.where((sectors > 30.) & (sectors < 34.))[0]
um=np.where(sectors > 28.)[0]
dat=dat[um]
ids=dat['target_name'] 
s=np.argsort(ids)
uids=ids[s]
uids=np.unique(uids)

files=glob.glob('mastDownload/**/',recursive=True)

for i in range(941,len(uids)):

	starid="TIC"+str(uids[i])
	print(i,len(uids),starid)
	
	obs_table = Observations.query_object(starid,radius=".001 deg")                                                                                                                                                          

	um=np.where((obs_table['obs_collection'] == 'TESS') &  (obs_table['t_exptime'] <= 120.)) 
	obs_table=obs_table[um] 
	#input(':')

	tstrs=obs_table['obs_id'] 
	sectors=np.zeros(len(tstrs))
	for j in range(0,len(tstrs)):
		sectors[j]=np.float(tstrs[j].split('-')[1][3:5])      

	#um=np.where((sectors > 30.) & (sectors < 34.))[0]
	um=np.where(sectors > 28.)[0]

	if (len(um) == 0):
		print('no data')
		#um=np.where((obs_table['obs_collection'] == 'TESS'))
		#print(obs_table['t_exptime'][um])
		#pdb.set_trace() 
		continue
	  
	print('downloading ',obs_table[um])
	data_products = Observations.get_product_list(obs_table[um])
	#pdb.set_trace() 
    
	um=np.where((data_products['productType'] == 'SCIENCE') & (data_products['description'] == 'Light curves'))[0]
	#um=np.where((data_products['productType'] == 'SCIENCE') & (data_products['description'] == 'Target pixel files'))[0]
	manifest = Observations.download_products(data_products[um])
	print(' ')
	#input(':')


