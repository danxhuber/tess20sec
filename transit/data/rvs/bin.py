import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.stats import mad_std
import pdb
from scipy import stats

def bin(xdata,ydata,sigmas,nbins):

    steps=(np.max(xdata)-np.min(xdata))/nbins
    start=np.min(xdata)
    stop=start+steps
    
    resx=np.zeros(nbins)
    resy=np.zeros(nbins)
    erry=np.zeros(nbins)
        
    
    for i in range(0,nbins):
        um=np.where( (xdata >= start) & (xdata < stop))[0]
        #print start,stop,len(um)
        if (len(um) < 5):
            start = start+steps
            stop = stop+steps
            continue
        resx[i]=start+steps/2. #np.median(xdata[um])
        resy[i]=np.median(ydata[um])
        
        #resy[i]=np.average(ydata[um],weights=sigmas[um])
        
        #print start,stop,resx[i],resy[i],len(um)
        #print xdata[um]
        #break
        #plt.clf()
        #plt.errorbar(xdata[um],ydata[um],1./sigmas[um])
        #raw_input(':')
        
        
        #pdb.set_trace()
        erry[i]=np.std(ydata[um])/np.sqrt(len(um))
        start = start+steps
        stop = stop+steps
        
    '''    
    plt.clf()
    plt.plot(xdata,ydata,'.')
    plt.errorbar(resx,resy,yerr=erry,fmt='o')
    pdb.set_trace()
    '''

    um=np.where(resy > 0.)[0]
    
    return resx[um],resy[um],erry[um]
  
def bin_time(xdata,ydata,steps):

    start=np.min(xdata)
    stop=start+steps
    
    nbins=int((np.max(xdata)-np.min(xdata))/steps)
    
    resx=np.zeros(nbins)
    resy=np.zeros(nbins)
    erry=np.zeros(nbins)
    
    for i in range(0,nbins):
        um=np.where( (xdata >= start) & (xdata < stop))[0]
        if (len(um) > 0):
                resx[i]=np.median(xdata[um])
                resy[i]=np.mean(ydata[um])
        #erry[i]=np.std(ydata[um])/np.sqrt(len(um))
        start = start+steps
        stop = stop+steps

    um=np.where((resy != 0.) & (np.isfinite(resy)))[0]
    return resx[um],resy[um],erry[um]

def bin_time_err(xdata,ydata,yerr,steps):

    start=np.min(xdata)
    stop=start+steps
    
    nbins=int((np.max(xdata)-np.min(xdata))/steps)+10
    
    resx=np.zeros(nbins)
    resy=np.zeros(nbins)
    erry=np.zeros(nbins)
    
    for i in range(0,nbins):
        um=np.where( (xdata >= start) & (xdata < stop))[0]
        if (len(um) > 0):
                resx[i]=np.median(xdata[um])
                resy[i]=np.mean(ydata[um])
                erry[i]=np.mean(yerr[um])
        start = start+steps
        stop = stop+steps

    um=np.where((resy != 0.) & (np.isfinite(resy)))[0]
    return resx[um],resy[um],erry[um]
  
    
def bin_time_digitized(xdata,ydata,steps):

    start=np.min(xdata)
    stop=start+steps
    
    nbins=int((np.max(xdata)-np.min(xdata))/steps)
    
    resx=np.zeros(nbins)
    resy=np.zeros(nbins)
    erry=np.zeros(nbins)
    
    
    xax=np.arange(np.min(xdata),np.max(xdata),steps)
    digitized=np.digitize(xdata,xax)
    yax=np.array([ydata[digitized == i].mean() for i in range(1,len(xax)+1)])
    um=np.isfinite(yax)
    xax=xax[um]
    yax=yax[um]
    return xax,yax,np.zeros(len(xax))
    
    
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
        #print bins[i],bins[i+1],resx[i],resy[i],len(um)
        #print xdata[um]
        #break
        
    '''    
    plt.clf()
    plt.plot(xdata,ydata,'.')
    plt.errorbar(resx,resy,yerr=erry,fmt='o')
    pdb.set_trace()
    '''
    
    return resx,resy,erry
    
        
def bin_set_mean(xdata,ydata,bins):

    nbins=len(bins)-1
    
    resx=np.zeros(nbins)
    resy=np.zeros(nbins)
    erry=np.zeros(nbins)
    
    for i in range(0,len(bins)-1):
        um=np.where( (xdata >= bins[i]) & (xdata < bins[i+1]))[0]
        print(bins[i],bins[i+1])
        resx[i]=(bins[i]+bins[i+1])/2.
        resy[i]=np.mean(ydata[um])
        #resy[i]=stats.mode(ydata[um])[0]
        erry[i]=np.std(ydata[um])/np.sqrt(len(um))
        
    '''    
    plt.clf()
    plt.plot(xdata,ydata,'.')
    plt.errorbar(resx,resy,yerr=erry,fmt='o')
    pdb.set_trace()
    '''
   
def bin_cadence(x,y,ncad):
    
	ncad=ncad
	resx=np.zeros(len(x))
	resy=np.zeros(len(x))
	erry=np.zeros(len(x))
	start=0
	end=ncad

	for i in range(0,len(x)-1):
		#print(start,end)
		arr=x[start:end]
		#print(start,end)
		#print(len(arr))
		resx[i]=np.mean(x[start:end])
		resy[i]=np.mean(y[start:end])
		erry[i]=np.std(y[start:end])
		start=start+ncad
		end=end+ncad
		#input(':')
		if (end > len(x)):
			break
			
	um=np.where(resx > 0.)[0]
			
	return resx[um],resy[um],erry[um]


