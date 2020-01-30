#!/usr/bin/env python

"""
SYN_EMCEEFIT.PY
Created by A. Shulevski, 10 Oct 2014

Fit Synchrotron models to data using the emcee module

The help (Syn_emceefit.py -h) provides information about inputs.

Versions
--------
v0.1  AS Initial version 20141010
v0.2  AS Updated code for readability 20151029
"""

'''
class Pass:
	
	def __init__(self, nu_arr, s_arr, s_err, fixed):
		self.nu_arr = nu_arr
		self.s_arr = s_arr
		self.s_err = s_err
		self.fixed = fixed
	
	def __call__(self):
		return self.nu_arr, self.s_arr, self.s_err, self.fixed
'''

version_string = 'v0.2 20151029 AS'

"""
Logging setup
"""
import logging
logfm = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(logfm)

logger = logging.getLogger('genlog')
logger.addHandler(handler)
logger.setLevel(logging.INFO)

logger.info('Syn_emceefit.py ' + version_string)

logger.info('Loading modules...')

import argparse
import sys
sys.path.append('/Users/shulevski/Documents/Kapteyn/ExtEnD/')
from Synfit import Synfit
import emcee as mc
import scipy.optimize as op
import numpy as np
import math as mt
import matplotlib.pyplot as plt
from pylab import *
import corner

def model_values(z, t_a, t_i, B, nu_arr, a_in, sf, model, loss):

	global synfit

	if synfit is not None:
		synfit.set_ta(t_a)
		synfit.set_ti(t_i)
		synfit.set_sf(sf)
		s_arr_model = synfit()
		return s_arr_model
	else:
		synfit = Synfit(z, t_a, t_i, B, nu_arr, a_in, sf, model, loss)
		s_arr_model = synfit()
		return s_arr_model

def lnlike(state, z, B, model, loss, nu_arr, s_arr, s_err):
	t_a, t_i, sf, a_in = state
	s_arr_model = model_values(z, t_a, t_i, B, nu_arr, a_in, sf, model, loss)
	inv_sigma2 = 1.0 / (s_err**2)
	lnlik = -0.5 * (np.sum((s_arr - s_arr_model)**2 * inv_sigma2 - np.log(2. * np.pi * inv_sigma2)))
	return lnlik

def lnprior(state):
	global bounds
	t_a, t_i, sf, a_in = state
	if bounds[0][0] < t_a < bounds[0][1] and bounds[2][0] < sf < bounds[2][1] and bounds[1][0] < t_i < bounds[1][1] and bounds[3][0] < a_in < bounds[3][1]:
		return 0.0
	return -np.inf

def lnprob(state, z, B, model, loss, nu_arr, s_arr, s_err):
	lp = lnprior(state)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(state, z, B, model, loss, nu_arr, s_arr, s_err)

def main(args):

	"""
	The main function.
	"""

	par_init = np.array([70., 10., 1.e-22, -0.7]) # starting point [the delta JP model has this scaling factor]
	global bounds
	bounds = ((20., 200.), (1., 20.), (1.e-27, 1.e-19), (-1., -0.6)) # bounds for the optimization
	sigma = [5., 5., 1.e-28, 0.05] # radius of the sphere in phase space from whithin which the walkers are released
	
	###### Region 5 flux density #####

	#nu_arr =np.array([135.e6, 145.e6, 325.e6, 610.e6, 1425.e6])
	#s_arr = np.array([3.40e-2, 2.48e-2, 3.12e-3, 1.53e-3, 2.73e-4])
	#s_err = np.array([0.0070304481547869852, 0.0052041315377105149, 0.00063277514019785433, 0.00012159455671392348, 5.250804236594206e-05])
	###########################################

	#'''
	# B2 0924+30 - integrated
	#frequencies_int = np.array([63.2, 112.6, 124.3, 132.1, 136.0, 159.5, 163.4, 167.3, 151., 325., 609., 1400., 4750., 10550.]) * 1.e6
	nu_arr = np.array([112.6, 132.1, 136.0, 159.5, 163.4, 167.3, 151., 325., 609., 1400., 4750., 10550.]) * 1.e6
	#flux_int = np.array([6501.5, 8384.4, 8937.3, 6774.5, 6738.5, 5214.0, 4751.2, 4701.7, 4600., 2425., 1094., 420., 60., 10.]) * 1.e-3
	s_err = np.array([7.6, 4.5, 4.4, 3.5, 2.2, 1.9, 360., 124., 56., 43., 7., 4.]) * 1.e-3

	s_arr = np.array([8384.4 / 1.108621, 6774.5 / 1.08504573, 6738.5 / 1.07699903, 5214.0 / 0.95513001, 4751.2 / 0.92609498, 4701.7 / 0.89703054, 4600., 2425., 1094., 420., 60., 10.]) * 1.e-3 # beam normalization correction

	#'''

	flux_pix_num = np.array([1065., 1619., 1619., 1619., 1619., 1619., 1619.])
	beam_pix_num = np.array([10.4, 16.4, 16.6, 16.6, 16.6, 16.6, 16.6])
	rms_pix_num = np.array([916., 1398., 1398., 1398., 1398., 1398., 1398.])
	
	
	for i in range(len(flux_pix_num)):
		s_err[i] = np.sqrt(np.power(np.divide(np.multiply(flux_pix_num[i]/beam_pix_num[i], s_err[i]), np.sqrt(rms_pix_num[i]/beam_pix_num[i])),2.0) + np.power(np.divide(s_err[i], np.sqrt(flux_pix_num[i]/beam_pix_num[i])),2.0) + np.power(s_arr[i] * 0.2, 2.))


	z = 0.026141
	B = 1.35e-6
	model = 'CI_off'
	loss = 'JP'

	global lnprob
	global lnprior
	global lnlike
	global synfit

	synfit = None
	
	ntemps, ndim, nwalkers = 10, 4, 200 # ntemps are number of T values for the PT sampler

	thd, burnit, runit, thi = 10, 100, 2000, 2

	stype = 0
	
	nll = lambda *parms: -lnlike(*parms)
	result = op.minimize(nll, par_init, args=(z, B, model, loss, nu_arr, s_arr, s_err), bounds=bounds, options={'maxiter': 2, 'disp': True})

	logger.info("Initial guess optimization result %s", str(result["x"]))
	logger.info("Initial guess optimization status: %s", str(result["status"]))
	logger.info("Initial guess optimization message: %s", str(result["message"]))
	
	brc, mac = 0, 0

	if stype == 0:
		# Ensemble Sampler
		logger.info("Using the Ensemble Sampler")
		init_pos = mc.utils.sample_ball(result["x"], sigma, size=nwalkers)
		sampler = mc.EnsembleSampler(nwalkers, ndim, lnprob, threads=thd, args=(z, B, model, loss, nu_arr, s_arr, s_err))
		for p, lnp, rstate in sampler.sample(init_pos, iterations=burnit):
			brc += 1
			logger.info("[Burn-in] Iteration %s", str(brc))
		sampler.reset()

		for p, lnp, rstate in sampler.sample(p, lnprob0=lnp , iterations=runit, thin=thi):
			mac += 1
			logger.info("[Main] Iteration %s", str(mac))
		samples = sampler.chain[:, burnit:, :].reshape((-1, ndim))
	else:
		# PT Sampler
		logger.info("Using the PT Sampler")
		init_pos = [mc.utils.sample_ball(result["x"], sigma, size=nwalkers) for i in range(ntemps)]
		##print 'Init pos: ', init_pos
		sampler = mc.PTSampler(ntemps, nwalkers, ndim, lnlike, lnprior, threads=thd, loglargs=(z, B, model, loss, nu_arr, s_arr, s_err))
		for p, lnp, lnl in sampler.sample(init_pos, iterations=burnit):
			brc+=1
			logger.info("[Burn-in] Iteration: %s", str(brc))
		sampler.reset()

		for p, lnp, lnl in sampler.sample(p, lnprob0=lnp, lnlike0=lnl, iterations=runit, thin=thi):
			mac+=1
			logger.info("[Main] Iteration: %s", str(mac))
		samples = sampler.chain[:, :, burnit:, :].reshape((-1, ndim))

	logger.info("Autocorrelation time: %s", mc.autocorr.integrated_time(samples))
	logger.info("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

	fig = corner.corner(samples, labels=["$T \, off [Myr]$", "$T \, on [Myr]$", "$sf$", "$a_{inj}$"], truths=[result["x"][0], result["x"][1], result["x"][2], result["x"][3]])
	plt.savefig('/Users/shulevski/Desktop/emcee_landscape.png', bbox_inches='tight')
	#plt.savefig("corner.png")
	plt.show()
	'''
	for i in range(ndim):
	    plt.figure()
	    plt.hist(10**sampler.flatchain[:,i], 100, histtype="step")
	    plt.title("Dimension {0:d}".format(i))
	plt.show()
	print 'Samples reshaped: ', samples
	'''
	t_a_mcmc, t_i_mcmc, sf_mcmc, a_inj_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))

	logger.info("Estimated parameters: %s %s %s %s", str(t_a_mcmc), str(t_i_mcmc), str(sf_mcmc), str(a_inj_mcmc))
	logger.info("Initial guess: %s %s %s %s", str(result["x"][0]), str(result["x"][1]), str(result["x"][2]), str(result["x"][3]))
	logger.info('End')

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--path', help='Location of data file [default='']', default='', type=str)

if __name__ == '__main__':
  args = ap.parse_args()
  main(args)
