import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
import exoplanet as xo
import pymc3 as pm
import pymc3_ext as pmx
import theano.tensor as tt
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel
from scipy.signal import savgol_filter as savgol
import bin
import pdb
import corner
from celerite2.theano import terms, GaussianProcess
import h5py
import astropy.constants as c
from astropy import units
import fnmatch
import pandas as pd

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
plt.rcParams['xtick.major.pad']='3'
plt.rcParams['ytick.major.pad']='3'
plt.rcParams['ytick.minor.visible'] = 'True'
plt.rcParams['xtick.minor.visible'] = 'True'
plt.rcParams['xtick.direction'] = 'inout'
plt.rcParams['ytick.direction'] = 'inout'
plt.rcParams['ytick.right'] = 'True'
plt.rcParams['xtick.top'] = 'True'

# subroutine to perform rough sigma clipping
def sigclip(x,y,subs,sig):
    keep = np.zeros_like(x)
    start=0
    end=subs
    nsubs=int((len(x)/subs)+1)
    for i in range(0,nsubs):        
        me=np.mean(y[start:end])
        sd=np.std(y[start:end])
        good=np.where((y[start:end] > me-sig*sd) & (y[start:end] < me+sig*sd))[0]
        keep[start:end][good]=1
        start=start+subs
        end=end+subs
    return keep

periods=6.2679
t0s=2040.03

dat=ascii.read('../SYDSAP/piMen_20s_SYDSAP.csv',delimiter=',')
time=dat['time']
fluxs=dat['flux']
fluxe=dat['flux_err']

res=sigclip(time,fluxs,1000,5)
good = np.where(res == 1)[0]
time=time[good]
fluxs=fluxs[good]
fluxe=fluxe[good]

ph=time % 6.2679
out=np.where((ph < 2.9) | (ph > 3.04))

width=1.0
boxsize=width/(2./60./24.)
box_kernel = Box1DKernel(boxsize)
smoothed_flux = savgol(fluxs[out],int(boxsize)-1,1,mode='mirror')

smo_int=np.interp(time,time[out],smoothed_flux)
fluxs=fluxs#/(smo_int)
fluxs=fluxs-np.mean(fluxs)


plt.ion()
plt.clf()
plt.plot(time,fluxs,'.')

ph = time % periods
intr=np.where((ph > 2.8) & (ph < 3.1))
intr=np.where((ph > 2.5) & (ph < 3.5))
#intr=np.where((ph > 2.3) & (ph < 3.7))

plt.clf()
plt.plot(ph,fluxs,'.')
plt.plot(ph[intr],fluxs[intr],'.')

#intr=np.arange(0,len(time))
t=np.array(time[intr])
y=np.array(fluxs[intr])
yerr = np.zeros(len(t))+np.std(y)
yerr=np.array(fluxe[intr])


binx2min,biny2min,binz2min=bin.bin_time(t,y,2./60./24.)
binx30min,biny30min,binz30min=bin.bin_time(t,y,30./60./24.)

#t=binx30min
#y=biny30min
#yerr = np.zeros(len(t))+np.std(y)



##################################################################################
### original RV data
'''
dat=ascii.read('pimen_rvs_binned.csv')
t_rv=np.array(dat['t'])
rv=np.array(dat['rv'])
rv_err=np.array(dat['rv_err'])
inst_id=np.array(dat['inst_id'])
t_rv=t_rv-2457000.
inst_names = ["aat", "harps_pre", "harps_post", "coralie", "espresso"]
ids=np.unique(inst_id)
for i in range(0,len(ids)):
	um=np.where(inst_id == ids[i])[0]
	rv[um]=rv[um]-np.mean(rv[um])
	if (ids[i] == 4):
		rv[um]=rv[um]+200.
	if (ids[i] == 2):
		rv[um]=rv[um]+200.
'''
	
dat=ascii.read('data/rvs/pimen_rvs_binned.csv')
#um=np.where(dat['inst_id'] == 7)[0]
#dat['inst_id'][um]=6

inst_names = ["aat", "harps_pre", "harps_post", "coralie_98", "coralie_07", 
	"coralie_14", "espresso_1", "espresso_2"]
#inst_names = ["aat", "harps_pre", "harps_post", "coralie_98", "coralie_07", 
#	"coralie_14", "espresso"]
ids=np.unique(dat['inst_id'])
for i in range(0,len(ids)):
	um=np.where(dat['inst_id'] == ids[i])[0]
	dat['rv'][um]=dat['rv'][um]-np.mean(dat['rv'][um])
	if (ids[i] > 5):
		dat['rv'][um]=dat['rv'][um]+200.
	if (ids[i] == 1):
		dat['rv'][um]=dat['rv'][um]-50.
	if (ids[i] == 2):
		dat['rv'][um]=dat['rv'][um]+150.
	if (ids[i] == 3):
		dat['rv'][um]=dat['rv'][um]-100.
	if (ids[i] == 5):
		dat['rv'][um]=dat['rv'][um]+70.
	if (ids[i] >= 6):
		dat['rv'][um]=dat['rv'][um]		

#um=np.where((dat['inst_id'] >= 3) & (dat['inst_id'] <= 5))[0]
#dat['inst_id'][um]=3
#um=np.where((dat['inst_id'] == 6))[0]
#dat['inst_id'][um]=4
#inst_names = ["aat", "harps_pre", "harps_post", "coralie", "espresso"]	

#um=np.where((dat['inst_id'] >= 3) & (dat['inst_id'] <= 5))[0]
#dat['inst_id'][um]=3
#um=np.where((dat['inst_id'] == 6))[0]
#dat['inst_id'][um]=4
#um=np.where((dat['inst_id'] == 7))[0]
#dat['inst_id'][um]=5
#inst_names = ["aat", "harps_pre", "harps_post", "coralie", "espresso_1", "espresso_2"]	


t_rv=np.array(dat['t'])
rv=np.array(dat['rv'])
rv_err=np.array(dat['rv_err'])
inst_id=np.array(dat['inst_id'])
t_rv=t_rv-2457000.	


##################################################################################


##################################################################################
### RV data with LS fit to means and quadratic trend subtracted
#dat=ascii.read('pimen_rvs_corr.csv')
#inst_names = ["aat", "harps_pre", "harps_post", "coralie", "espresso"]

#dat=ascii.read('pimen_rvs_v2_corr.csv')
#inst_names = ["aat", "harps_pre", "harps_post", "coralie-98", "coralie-07", 
#	"coralie-14","espresso_1","espresso_2"]

#t_rv=np.array(dat['t'])
#rv=np.array(dat['rv'])
#rv_err=np.array(dat['rv_err'])
#inst_id=np.array(dat['inst_id'])
##################################################################################


num_inst = len(inst_names)

# period and T0 priors
periods = [2090.0, 6.2679]
period_errs = [10.0, 0.01]
t0s = [-466.78, 1519.8068]
t0_errs = [1.0, 0.1]

#Ks = [190., 2.]
#Ks_errs = [10., 5.]
mpls = [4500., 5.]
mpls_errs = [500., 5.]



# Compute a reference time that will be used to normalize the trends model
x_ref = 0.5 * (t_rv.min() + t_rv.max())
t_pred = np.linspace(t_rv.min() - 400, t_rv.max() + 400, 10000)

plt.clf()
plt.subplot(2,1,1)
plt.plot(t,y,'.')
plt.subplot(2,1,2)
for i in range(0,len(inst_names)):
	um=np.where(inst_id == i)[0]
	plt.plot(t_rv[um],rv[um],'.',label=inst_names[i])

plt.legend()
input(':')


with pm.Model() as model:

	# parameters shared between transit and RVs (t0, periods, eccentricities)
	t0 = pm.Normal("t0", mu=np.array(t0s), sd=np.array(t0_errs), shape=2)
	P = pm.Bound(pm.Normal, lower=0)(
		"P",
		mu=np.array(periods),
		sd=np.array(period_errs),
		shape=2,
		testval=np.array(periods),
	)

	# NB ecs = sqrt(e)sinw and sqrt(e)cosw	
	ecs = pmx.UnitDisk("ecs", shape=(2, 2), testval=0.01 * np.ones((2, 2)))
	ecc = pm.Deterministic("ecc", tt.sum(ecs ** 2, axis=0))
	omega = pm.Deterministic("omega", tt.arctan2(ecs[1], ecs[0]))	
	
	m_pl = pm.Normal("m_pl", mu=np.array(mpls), sd=np.array(mpls_errs), shape=2)

	# seismology, this work
	rho_star = pm.Normal("rho_star", mu=1.050, sd=0.013)
	r_star = pm.Normal("r_star", mu=1.136, sd=0.010, shape=1)
	#m_star = pm.Normal("m_star", mu=1.091, sd=0.030)
	
	# Damasso et al.
	#m_star = pm.Normal("m_star", mu=1.07, sd=0.04, shape=1)
	#r_star = pm.Normal("r_star", mu=1.17, sd=0.02, shape=1)


	################################################
	## TRANSIT MODEL STARTS HERE
	################################################

	# The baseline flux
	mean = pm.Normal("mean", mu=0.0, sd=1.0)

	# The Kipping (2013) parameterization for quadratic limb darkening paramters
	#u = xo.distributions.QuadLimbDark("u", testval=np.array([0.3, 0.2]))
	#u = [0.28,0.28]
	u = pm.Normal("u", mu=0.28, sd=0.2, shape=2)
	#u = xo.QuadLimbDark("u")

	r = pm.Uniform("r", lower=0.01, upper=1.0, shape=2, testval=0.016)
	
	#incl = pm.Uniform("incl", lower=-np.pi, upper=np.pi, shape=2, testval=np.pi/2.)
	#b = xo.distributions.ImpactParameter("b", ror=r, shape=2, testval=0.5)
	b = pm.Uniform("b", lower=0.0, upper=10.0, shape=2, testval=0.5)

	# extra noise term for error bars
	sigma = pm.InverseGamma("sigma", alpha=3.0, beta=2 * np.std(y))

	# Gaussian process noise model
	# GP amplitude
	sigma_gp = pm.Lognormal("sigma_gp", mu=0.0, sigma=10.0)
	# GP timescale
	rho_gp = pm.Lognormal("rho_gp", mu=np.log(4000.0), sigma=10.0)
	#Q_gp = pm.Uniform("Q_gp", lower=0.01, upper=20.)
	kernel = terms.SHOTerm(sigma=sigma_gp, rho=rho_gp, Q=1/np.sqrt(2))

	# Set up a Keplerian orbit for the planets
	orbit = xo.orbits.KeplerianOrbit(period=P, t0=t0,b=b, rho_star=rho_star, r_star=r_star, 
		ecc=ecc, omega=omega,m_planet=m_pl, m_planet_units=units.M_earth)
	#orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, b=b, rho_star=rho_star)

	#b = pm.Deterministic("b", orbit.b)
	incl = pm.Deterministic("incl", orbit.incl)
	aor = pm.Deterministic("aor", orbit.a/orbit.r_star)
	rplanet = pm.Deterministic("rplanet", r*c.R_sun.value/c.R_earth.value)
	ror = pm.Deterministic("ror", r/orbit.r_star)
	rstar = pm.Deterministic("rstar", orbit.r_star)
	#mplanet = pm.Deterministic("mplanet", orbit.m_planet)

	# Compute the model light curve using starry
	light_curves = xo.LimbDarkLightCurve(u).get_light_curve(orbit=orbit, r=r, t=t)
	light_curve = pm.math.sum(light_curves, axis=-1) + mean

	# Here we track the value of the model light curve for plotting
	# purposes
	pm.Deterministic("light_curves", light_curves)

	# Finally the GP observation model
	gp = GaussianProcess(
		kernel, t=t, diag=yerr ** 2 + sigma ** 2, mean=light_curve
	)
	gp.marginal("transit_obs", observed=y)
	pm.Deterministic("gpmod", gp.predict(y, include_mean=False))
	#pm.Deterministic("mod", gp.predict(y)) 


	################################################
	## RV MODEL STARTS HERE
	################################################

	# Wide log-normal prior for semi-amplitude
	#K = pm.Bound(pm.Normal, lower=0)("K", mu=np.array(Ks), sd=np.array(Ks_errs), shape=2, testval=np.array(Ks))

	means = pm.Uniform("means", lower=-100., upper=0., shape=num_inst, testval=-40.)
	logs = pm.Uniform("logs", lower=-1., upper=5., shape=num_inst, testval=0.)
	#logs = pm.Normal("logs", mu=np.log(np.median(rv_err)), sd=5.0, shape=num_inst)    
	trend = pm.Normal("trend", mu=0, sd=10.0 ** -np.arange(2)[::-1], shape=2)

	# Compute the RV offset and jitter for each data point depending on its instrument
	rvmean = tt.zeros(len(t_rv))
	for i in range(len(inst_names)):
		rvmean += means[i] * (inst_id == i)
	pm.Deterministic("rvmean", rvmean)
	resid = rv - rvmean

	def get_rv_model(x, name=""):
		# First the RVs induced by the planets
		vrad = orbit.get_radial_velocity(x)
		pm.Deterministic("vrad"+ name, vrad)

		# Define the background model
		A = np.vander(x - x_ref, 2)
		bkg = pm.Deterministic("bkg"+ name, tt.dot(A, trend))
		#bkg = pm.Deterministic("bkg"+ name, tt.zeros(len(x)))
		
		# Sum over planets and add the background to get the full model
		#return pm.Deterministic("rv_model" + name, tt.sum(vrad, axis=-1))
		return pm.Deterministic("rv_model" + name, tt.sum(vrad, axis=-1)+bkg)

	rv_model = get_rv_model(t_rv)
	rv_model_pred = get_rv_model(t_pred, name="_pred")

	jitter = tt.zeros(len(t_rv))
	for i in range(len(inst_names)):
		jitter += logs[i] * (inst_id == i)
	pm.Deterministic("jitter", jitter)
	pm.Deterministic("rvscatter", tt.exp(logs))

	err = tt.sqrt(rv_err ** 2 + tt.exp(2 * jitter))
	pm.Normal("obs", mu=rv_model, sd=err, observed=resid)

	map_soln = model.test_point
	
	map_soln = pmx.optimize(map_soln, [means])
	map_soln = pmx.optimize(map_soln, [means, logs])
	map_soln = pmx.optimize(map_soln, [means, logs, trend])
	map_soln = pmx.optimize(map_soln, [means, logs, trend, rho_gp, sigma_gp])
	map_soln = pmx.optimize(map_soln, [means, logs, trend, m_pl, rho_gp, sigma_gp])
	map_soln = pmx.optimize(map_soln, [means, logs, trend, t0, m_pl, P, rho_gp, sigma_gp])
	map_soln = pmx.optimize(map_soln, [means, logs, trend, t0, m_pl, P, ecc, omega, b, r, rho_gp, sigma_gp])
	map_soln = pmx.optimize(map_soln)


print(map_soln['ror'][1],map_soln['rplanet'][1],map_soln['b'][1],map_soln['ecc'][1])
print(map_soln['sigma_gp'],map_soln['rho_gp']) 


########################################################################################
### start MAP model plotting
########################################################################################
plt.ion()
plt.clf()
plt.subplot(2,2,1)
plt.plot(t, y, ".k", ms=4, label="data",color='grey')
plt.plot(t, map_soln['light_curves'][:,1], label="transit",color='red')
plt.plot(t, map_soln['gpmod'], label="GP",color='green')
plt.plot(t, map_soln['light_curves'][:,1]+map_soln['gpmod'], label="GP+transit",color='blue')
plt.legend()

plt.subplot(2,2,3)
plt.plot(t % map_soln['P'][1], y, ".k", color='grey', ms=10, label="data")
plt.plot(t % map_soln['P'][1], y-map_soln['gpmod'], ".k", ms=10, label="data-GP")

xdetrend=t % map_soln['P'][1]
ydetrend=y-map_soln['gpmod']
binx,biny,binz=bin.bin_time(xdetrend,ydetrend,0.01)
plt.plot(binx, biny, "o")

plt.plot(t % map_soln['P'][1], map_soln['light_curves'][:,1], ".k", ms=4, label="transit",color='red')

rve=np.sqrt( rv_err**2 + np.exp(map_soln['jitter'])**2 )

detrended = rv- map_soln["rvmean"] - map_soln["vrad"][:,1] - map_soln["bkg"]
mod =  map_soln["vrad_pred"][:,0]
period = map_soln["P"][0]
ph=t_pred % period
s=np.argsort(ph)

plt.subplot(2,2,2)
plt.errorbar(t_rv % period, detrended, yerr=rve, fmt=",k")
plt.plot(t_rv % period, detrended,'o')
plt.plot(ph[s], mod[s], alpha=0.5)

detrended = rv- map_soln["rvmean"] - map_soln["vrad"][:,0] - map_soln["bkg"]
mod =  map_soln["vrad_pred"][:,1]
period = map_soln["P"][1]
ph=t_pred % period
s=np.argsort(ph)

plt.subplot(2,2,4)
um=np.where((inst_id >= 6))[0]
#um=np.where((inst_id > 4))[0]
plt.errorbar(t_rv[um] % period, detrended[um], yerr=rve[um], fmt=".",color='grey',zorder=-32)
plt.scatter(t_rv[um] % period, detrended[um])
plt.plot(ph[s], mod[s], alpha=0.5)

plt.tight_layout()

plt.show()

input(':')

# the background model
plt.clf()
detrended = rv- map_soln["rvmean"]
for i in range(0,len(inst_names)):
	um=np.where(inst_id == i)[0]
	plt.plot(t_rv[um],detrended[um],'o',label=inst_names[i])
plt.plot(t_rv,map_soln["bkg"],color='grey',lw=2,alpha=0.5,ls='dashed')
plt.legend()

input(':')
########################################################################################
### end MAP model plotting
########################################################################################


# sample or load trace from previous run
np.random.seed(888)
with model:
	trace = pm.load_trace('trace_transitplusrv_210507')
    #trace = pmx.sample(tune=2500,draws=1500,start=map_soln,chains=4,initial_accept=0.9,target_accept=0.99)

#pm.save_trace(trace,directory='trace_transitplusrv_210507',overwrite=True)
   
input(':')


# transit+GP model 
pred = trace["gpmod"]+trace["light_curves"][:,:,1]
pred = np.percentile(pred, [16, 50, 84], axis=0)

# GP only
gpmod = trace["gpmod"]
gpmod = np.percentile(gpmod, [16, 50, 84], axis=0)

# transit model only
trmod = trace["light_curves"][:,:,1]
trmod = np.percentile(trmod, [16, 50, 84], axis=0)

# RV background model 
bkg = trace["bkg"]
bkg = np.percentile(bkg, [16, 50, 84], axis=0)

	
# output the transit model to a text file
#ascii.write([t,y,pred[1],pred[0],pred[2],gpmod[1],gpmod[0],gpmod[2],trmod[1],trmod[0],trmod[2]],'transitfit_model.csv',names=['time','flux','model','modelerrm','modelerrp','gpmod','gpmoderrm','gpmoderrp','trmod','trmoderrm','trmoderrp'],delimiter=',')

#variables=["b", "ecc","omega","ror"]
#allsamples=pm.trace_to_dataframe(trace, varnames=variables)
#allsamples.to_csv('samples_transitplusrv_v3.csv')


# pymc3 summary (check convergence)
variables=["P", "t0", "r", "b", "rho_star", "ecs", "u", "sigma", "sigma_gp", 
	"rho_gp", "ecc", "omega", "aor", "incl", "rplanet","ror" ,"m_pl","means","rvscatter","trend"]
pm.summary(trace, varnames=variables)
	
'''
func_dict = {"mean": np.mean, "std": np.std, "median": lambda x: np.percentile(x, 50), "16%": lambda x: 
	np.percentile(x, 50)-np.percentile(x, 16), "84%": lambda x: np.percentile(x, 84)-np.percentile(x, 50)}
pm.summary(trace, varnames=variables,round_to="none", stat_funcs=func_dict, extend=False)

func_dict = {"mean": np.mean, "std": np.std, "median": lambda x: np.percentile(x, 50), "16%": lambda x: 
	np.percentile(x, 50)-np.percentile(x, 16), "84%": lambda x: np.percentile(x, 84)-np.percentile(x, 50)}
df=pm.summary(trace, varnames=variables,round_to="none", stat_funcs=func_dict, extend=False)
'''



########################################################################################
### begin latex table output
########################################################################################

var=["t0","P","b","rho_star","u","ror","sigma_gp","rho_gp","sigma","rvscatter","ecc","omega","aor","incl","m_pl","rplanet"]
rf=[2,4,2,5,1,3,3,2,2,2,5,1,2,1,2,2,2,2,2,2,2,2,4,3,2,1,1,3,2,2,1,2,1,3,2,2]

rnd=5
ct=0

for j in range(0,len(var)):
	xt=trace[var[j]]
	if (fnmatch.fnmatch(var[j],'*sigma*')):
		xt=xt*1e6
	if (fnmatch.fnmatch(var[j],'*incl*')):
		xt=xt*180./np.pi
	if (fnmatch.fnmatch(var[j],'*omega*')):
		xt=xt*180./np.pi
	try:
		for i in range(0,xt.shape[1]):
			x1=np.median(xt[:,i])
			x2=np.percentile(xt[:,i], 84)-x1
			x3=x1-np.percentile(xt[:,i], 16)
			print(var[j]+'_'+str(i),'$'+str(round(x1,rf[ct]))+'^{+'+str(round(x2,rf[ct]))+'}'+'_{-'+str(round(x3,rf[ct]))+'}'+'$')
			ct=ct+1	
	except:
			x1=np.median(xt)
			x2=np.percentile(xt, 84)-x1
			x3=x1-np.percentile(xt, 16)
			print(var[j],'$'+str(round(x1,rf[ct]))+'^{+'+str(round(x2,rf[ct]))+'}'+'_{-'+str(round(x3,rf[ct]))+'}'+'$')	
			ct=ct+1	

print(' ')
print('K')				
# calculate semi-amplitude K (since we sampled over mass)
rhosun=c.M_sun.cgs.value/(4.*np.pi*c.R_sun.cgs.value**3/3)
mstar=(trace['rho_star']/rhosun)*trace['r_star'][:,0]**3
k=np.sqrt(c.G.cgs.value/(1.-trace['ecc']**2))*trace['m_pl']*c.M_earth.cgs.value* \
	np.sin(trace['incl'])*(1.091*c.M_sun.cgs.value+trace['m_pl']* \
	c.M_earth.cgs.value)**(-0.5)*(trace['aor']*trace['r_star']*c.R_sun.cgs.value)**(-0.5)/1e2
xt=k
for i in range(0,xt.shape[1]):
	x1=np.median(xt[:,i])
	x2=np.percentile(xt[:,i], 84)-x1
	x3=x1-np.percentile(xt[:,i], 16)
	print('K_'+str(i),'$'+str(round(x1,rf[ct]))+'^{+'+str(round(x2,rf[ct]))+'}'+'_{-'+str(round(x3,rf[ct]))+'}'+'$')
	ct=ct+1	

print(' ')
print('au')
rf=5
xt=trace['aor']*trace['rstar']*c.R_sun.cgs.value/c.au.cgs.value
for i in range(0,xt.shape[1]):
	x1=np.median(xt[:,i])
	x2=np.percentile(xt[:,i], 84)-x1
	x3=x1-np.percentile(xt[:,i], 16)
	print('au_'+str(i),'$'+str(round(x1,rf))+'^{+'+str(round(x2,rf))+'}'+'_{-'+str(round(x3,rf))+'}'+'$')

	
print(' ')
print('esinw and ecosw')	
rf=4
xt=trace['ecs'][:,:,0]
for i in range(0,xt.shape[1]):
	x1=np.median(xt[:,i])
	x2=np.percentile(xt[:,i], 84)-x1
	x3=x1-np.percentile(xt[:,i], 16)
	print('ecs_b'+'_'+str(i),'$'+str(round(x1,rf))+'^{+'+str(round(x2,rf))+'}'+'_{-'+str(round(x3,rf))+'}'+'$')

rf=3
xt=trace['ecs'][:,:,1]
for i in range(0,xt.shape[1]):
	x1=np.median(xt[:,i])
	x2=np.percentile(xt[:,i], 84)-x1
	x3=x1-np.percentile(xt[:,i], 16)
	print('ecs_c'+'_'+str(i),'$'+str(round(x1,rf))+'^{+'+str(round(x2,rf))+'}'+'_{-'+str(round(x3,rf))+'}'+'$')

print(' ')
print('mass of planet b')
rf=2
xt=trace['m_pl'][:,0]*c.M_earth.value/c.M_jup.value
x1=np.median(xt)
x2=np.percentile(xt, 84)-x1
x3=x1-np.percentile(xt, 16)
print('M_b sini'+'_'+str(i),'$'+str(round(x1,rf))+'^{+'+str(round(x2,rf))+'}'+'_{-'+str(round(x3,rf))+'}'+'$')

incdist=(49.9+np.random.randn(len(xt))*5.0)*np.pi/180.
xt=trace['m_pl'][:,0]*c.M_earth.value/c.M_jup.value/np.sin(incdist)
x1=np.median(xt)
x2=np.percentile(xt, 84)-x1
x3=x1-np.percentile(xt, 16)
print('M_b'+'_'+str(i),'$'+str(round(x1,rf))+'^{+'+str(round(x2,rf))+'}'+'_{-'+str(round(x3,rf))+'}'+'$')
	
########################################################################################
### end latex table output
########################################################################################

	
########################################################################################
########################################################################################
### Posterior Plotting starts here
########################################################################################
########################################################################################

## various corner plots
'''
plt.figure()
samples = pm.trace_to_dataframe(trace, varnames=["b", "ecc", "omega","ror"])
_ = corner.corner(samples)

samples = pm.trace_to_dataframe(trace, varnames=["b", "incl"])
_ = corner.corner(samples)

samples = pm.trace_to_dataframe(trace, varnames=["sigma", "mean", "rho_gp", "sigma_gp"])
_ = corner.corner(samples)

samples = pm.trace_to_dataframe(trace, varnames=["means"])
_ = corner.corner(samples)

samples = pm.trace_to_dataframe(trace, varnames=["rvscatter"])
_ = corner.corner(samples)

samples = pm.trace_to_dataframe(trace, varnames=["logs"])
_ = corner.corner(samples)

samples = pm.trace_to_dataframe(trace, varnames=["trend"])
_ = corner.corner(samples)
'''

## background model
detrended = rv - np.median(trace["rvmean"],axis=0)
plt.plot(t_rv,detrended,'o')
plt.plot(t_rv, bkg[1], label="background model", color='red')
art = plt.fill_between(t_rv, bkg[0], bkg[2], alpha=0.5, zorder=1000,color='red')



## transit + RV data and model
plt.ion()
plt.clf()

plt.rcParams['font.size']=15

plt.subplot(2,2,1)
plt.plot(t, y, "o", color='grey', label='20-sec', ms=3, alpha=0.7)
plt.plot(binx2min, biny2min, "D", ms=4, color='blue', label='2-min', alpha=0.7)
plt.plot(binx30min, biny30min, "s", ms=7, color='green',label='30-min')

#plt.plot(x, gp_mod, color="C1", label="model")
pred = np.percentile(pred, [16, 50, 84], axis=0)
plt.plot(t, pred[1], label="GP+transit model", color='red')
art = plt.fill_between(t, pred[0], pred[2], alpha=0.5, zorder=1000,color='red')
plt.xlim(t.min(), t.max())
#plt.legend(fontsize=10)
plt.xlabel("Time (days)")
plt.ylabel("Flux")

plt.subplot(2,2,3)
# Get the posterior median orbital parameters
p = np.median(trace["P"][:,1])
t0 = np.median(trace["t0"][:,1])

# Plot the folded data
x_fold = (t - t0 + 0.5 * p) % p - 0.5 * p
#plt.plot(x_fold, y, ".k", label="20-sec", zorder=-1000, color='grey', label='data - GP')
plt.plot(x_fold, y-np.median(gpmod,axis=0), "o", color='grey', label='20-sec', ms=3, alpha=0.7)

photdetr=y-np.median(gpmod,axis=0)
detrbinx2min,detrbiny2min,detrbinz2min=bin.bin_time(t,photdetr,2./60./24.)
detrbinx30min,detrbiny30min,detrbinz30min=bin.bin_time(t,photdetr,30./60./24.)

x_fold_2min = (detrbinx2min - t0 + 0.5 * p) % p - 0.5 * p
x_fold_30min = (detrbinx30min - t0 + 0.5 * p) % p - 0.5 * p

plt.plot(x_fold_2min, detrbiny2min, "D", ms=4, color='blue', label='2-min', alpha=0.7)
plt.plot(x_fold_30min, detrbiny30min, "s", ms=7, color='green',label='30-min')

# Plot the folded model
inds = np.argsort(x_fold)
inds = inds[np.abs(x_fold)[inds] < 0.3]
pred = trace["light_curves"][:, inds, 1]
pred = np.percentile(pred, [16, 50, 84], axis=0)
plt.plot(x_fold[inds], pred[1], label="",lw=3,color='red')
art = plt.fill_between(
    x_fold[inds], pred[0], pred[2], alpha=0.5, zorder=1000,color='red')
art.set_edgecolor("none")
#plt.legend(fontsize=10, loc=4)
plt.xlim(-0.5 * p, 0.5 * p)
plt.xlim(-0.1, 0.1);
#plt.title('pi Mensae c');

#plt.legend(fontsize=10, loc=4)
plt.xlim(-0.5 * p, 0.5 * p)
plt.xlabel("Time since transit (days)")
plt.ylabel("De-trended flux")
plt.xlim(-0.1, 0.1);

plt.legend(loc='upper right')


plt.subplot(2,2,2)

detrended = rv - np.median(trace["vrad"][:,:,1],axis=0) - np.median(trace["rvmean"],axis=0) - np.median(trace["bkg"],axis=0)
rvmod =  trace["vrad_pred"][:,:,0]
mod = np.percentile(rvmod, [16, 50, 84], axis=0)
period = map_soln["P"][0]
ph=t_pred % period
s=np.argsort(ph)
plt.errorbar(t_rv % period, detrended, yerr=rve, fmt=".",color='grey',zorder=-32)
plt.plot(t_rv % period, detrended,'o')
plt.plot(ph[s], mod[1][s], alpha=0.5, color="red")
art = plt.fill_between(ph[s], mod[0][s], mod[2][s], color="red", alpha=0.5, zorder=1000)
plt.xlabel("Phase (days)")
plt.ylabel("RV (m/s)")
plt.xlim(0,period)

plt.subplot(2,2,4)
detrended = rv - np.median(trace["vrad"][:,:,0],axis=0) - np.median(trace["rvmean"],axis=0) - np.median(trace["bkg"],axis=0)
rvmod =  trace["vrad_pred"][:,:,1]
mod = np.percentile(rvmod, [16, 50, 84], axis=0)
period = map_soln["P"][1]
ph=t_pred % period
s=np.argsort(ph)
#keep=np.where((inst_id >= 1.) & (inst_id <= 2.) )[0]
#plt.errorbar(t_rv[keep] % period, detrended[keep], yerr=rve[keep], fmt=".",color='grey',zorder=-32,alpha=0.2)
#plt.plot(t_rv[keep] % period, detrended[keep],'o',color='green',alpha=0.2)
keep=np.where(inst_id > 6.)[0]
plt.errorbar(t_rv[keep] % period, detrended[keep], yerr=rve[keep], fmt=".",color='grey',zorder=-32)
plt.plot(t_rv[keep] % period, detrended[keep],'o')
plt.plot(ph[s], mod[1][s], alpha=0.5, color="red")
art = plt.fill_between(ph[s], mod[0][s], mod[2][s], color="red", alpha=0.5, zorder=1000)
plt.xlabel("Phase (days)")
plt.ylabel("RV (m/s)")
plt.xlim(0,period)
plt.ylim(-6,6)

plt.tight_layout()

#plt.savefig('transitplusrv.png',dpi=200)





# use a color-blind friendly palette
# orange, red, light blue, dark blue
colors=['#FF9408','#DC4D01','#00A9E0','#016795']


## transit data and model only
plt.clf()
plt.rcParams['font.size']=18

fig = plt.figure(figsize=(15, 5))

# Get the posterior median orbital parameters
p = np.median(trace["P"][:,1])
t0 = np.median(trace["t0"][:,1])

# Plot the folded data
x_fold = (t - t0 + 0.5 * p) % p - 0.5 * p
#plt.plot(x_fold, y, ".k", label="20-sec", zorder=-1000, color='grey', label='data - GP')
plt.plot(x_fold*24, (y-np.median(gpmod,axis=0))*1e6, "o", color='grey', label='20-sec', ms=3, alpha=0.7)

photdetr=y-np.median(gpmod,axis=0)
detrbinx2min,detrbiny2min,detrbinz2min=bin.bin_time(t,photdetr,2./60./24.)
detrbinx30min,detrbiny30min,detrbinz30min=bin.bin_time(t,photdetr,30./60./24.)

x_fold_2min = (detrbinx2min - t0 + 0.5 * p) % p - 0.5 * p
x_fold_30min = (detrbinx30min - t0 + 0.5 * p) % p - 0.5 * p

plt.plot(x_fold_2min*24, detrbiny2min*1e6, "D", ms=4, color=colors[3], label='2-min', alpha=0.7)
plt.plot(x_fold_30min*24, detrbiny30min*1e6, "s", ms=7, color=colors[0],label='30-min')

# Plot the folded model
inds = np.argsort(x_fold)
inds = inds[np.abs(x_fold)[inds] < 0.3]
pred = trace["light_curves"][:, inds, 1]
pred = np.percentile(pred, [16, 50, 84], axis=0)
plt.plot(x_fold[inds]*24, pred[1]*1e6, label="",lw=3,color=colors[1])
art = plt.fill_between(
    x_fold[inds]*24, pred[0]*1e6, pred[2]*1e6, alpha=0.5, zorder=1000,color=colors[1])
art.set_edgecolor("none")

#plt.legend(fontsize=10, loc=4)
plt.xlim(-3,3)
plt.ylim([-1000,1000])
plt.xlabel("Time from Mid-Transit (hours)")
plt.ylabel("Relative Flux (ppm)")
plt.legend(loc='upper right')

plt.tight_layout(pad=0.4)

plt.savefig('fig10.png',dpi=200)

#plt.savefig('plots/fig-foldedtransit-v2.png',dpi=200)



## RV data and model only
plt.rcParams['font.size']=16

fig = plt.figure(figsize=(7, 8))

plt.clf()
plt.subplot(2,1,1)

rve=np.sqrt((rv_err**2) + np.median(np.exp(trace['jitter']),axis=0)**2)

detrended = rv - np.median(trace["vrad"][:,:,1],axis=0) - np.median(trace["rvmean"],axis=0) - np.median(trace["bkg"],axis=0)
rvmod =  trace["vrad_pred"][:,:,0]
mod = np.percentile(rvmod, [16, 50, 84], axis=0)
period = np.median(trace["P"][:,0])
ph=t_pred % period
s=np.argsort(ph)
plt.errorbar(t_rv % period, detrended, yerr=rve, fmt=".",color='grey',zorder=-32)

keep=np.where(inst_id == 0.)[0]
plt.plot(t_rv[keep] % period, detrended[keep],'D',label='AAT',color='black')
keep=np.where((inst_id == 1.) | (inst_id == 2.))[0]
plt.plot(t_rv[keep] % period, detrended[keep],'s',label='HARPS',color=colors[0])
keep=np.where((inst_id == 3.) | (inst_id == 4.) | (inst_id == 5.))[0]
plt.plot(t_rv[keep] % period, detrended[keep],'^',label='CORALIE',color=colors[2])
keep=np.where(inst_id > 5.)[0]
plt.plot(t_rv[keep] % period, detrended[keep],'o',label='ESPRESSO',color=colors[3])

plt.plot(ph[s], mod[1][s], alpha=0.5, color="red")
art = plt.fill_between(ph[s], mod[0][s], mod[2][s], color=colors[1], alpha=0.5, zorder=1000)
plt.xlabel("Phase (days)")
plt.ylabel("Radial Velocity (m/s)")
plt.xlim(0,period)
plt.legend()

plt.subplot(2,1,2)
detrended = rv - np.median(trace["vrad"][:,:,0],axis=0)- np.median(trace["rvmean"],axis=0) - np.median(trace["bkg"],axis=0)
rvmod =  trace["vrad_pred"][:,:,1]
mod = np.percentile(rvmod, [16, 50, 84], axis=0)
period = np.median(trace["P"][:,1])
ph=t_pred % period
s=np.argsort(ph)
#keep=np.where(rve < 2.)[0]

keep=np.where(rve <= 5.)[0]
plt.errorbar(t_rv[keep] % period, detrended[keep], yerr=rve[keep], fmt=".",color='grey',zorder=-32,alpha=0.3)

keep=np.where(inst_id > 5.)[0]
plt.errorbar(t_rv[keep] % period, detrended[keep], yerr=rve[keep], fmt=".",color='grey',zorder=-32)
plt.plot(t_rv[keep] % period, detrended[keep],'o',color=colors[3])
plt.plot(ph[s], mod[1][s], alpha=0.5, color="red")
art = plt.fill_between(ph[s], mod[0][s], mod[2][s], color=colors[1], alpha=0.5, zorder=1000)
plt.xlabel("Phase (days)")
plt.ylabel("Radial Velocity (m/s)")
plt.xlim(0,period)
plt.ylim(-6,7)

plt.subplots_adjust(wspace=0.20,hspace=0.24,left=0.15,right=0.98,bottom=0.07,top=0.99)

plt.savefig('fig11.png',dpi=200)



## eccentricity posteriors
fig = plt.figure(figsize=(7, 5))

df=pd.read_csv('data/transit/samples_transit_v2.csv')
df2 = df[["b__0","ecc","omega","ror__0"]]
ecc=trace['ecc']

plt.rcParams['font.size']=18
plt.ion()
plt.clf()

bs=np.arange(0.,1.,0.02)

plt.hist(df2['ecc'],bins=bs,alpha=0.5,density=True,label='Asteroseismology+Transit',color=colors[1])
plt.hist(df2['ecc'],bins=bs,alpha=0.5,density=True,label='',color=colors[1],
	histtype='step',ls='dashed',lw=3)

plt.hist(ecc[:,1],bins=bs,alpha=0.5,density=True,label='Asteroseismology+Transit+RV',color=colors[3])
plt.hist(ecc[:,1],bins=bs,alpha=0.5,density=True,label='',color=colors[3],
	histtype='step',ls='dashed',lw=3)

plt.xlabel("Eccentricity of $\pi$ Men c")
plt.ylabel("Probability Density")

plt.legend()
plt.xlim([0.0,0.9])
plt.tight_layout(pad=0.4)
plt.savefig('fig12.png',dpi=200)


