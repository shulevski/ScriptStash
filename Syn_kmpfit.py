#!/usr/bin/env python

# Kmpfit fitting of the JP CI synchrotrom model

import sys
#sys.path.append('/Users/shulevski/Documents/Kapteyn/1431+1331_spix/')
#from Synfit import Synfit
import Synfit_Leith as sl
#from Synfit_Eint import Synfit
import numpy as np
import math as mt
import matplotlib.pyplot as plt
from pylab import *
from scipy.integrate import quad, quadrature, fixed_quad, simps
import scipy.special as ss
#from kapteyn import kmpfit
from scipy.special import gammainc, chdtrc
from scipy.optimize import fminbound
from scipy.optimize import curve_fit
import scipy.ndimage as spn

global call_counter
call_counter = 0

def synfit_opt_func(freqs, t_a, t_i, N, a_inj):

	z = 0.026141
	B = 1.35e-6

	synfit = Synfit(z, 0., 0., B, 10.**freqs, 0., 0., 'CI_off', 'JP')
	'''
	print "Freqs: ", 10.**freqs
	print t_a
	print t_i
	print N
	print a_inj
	'''
	synfit.set_ti(10.**t_i)
	synfit.set_ta(10.**t_a)
	synfit.set_ntot(10.**N)
	synfit.set_inj((10.**a_inj) * -1.)

	model_flux = synfit()
	#print "S: ", model_flux

	#print np.log10(1.),  np.log10(80.)
	#print np.log10(1.e-8), np.log10(1.e-15)
	#print np.log10(0.5), np.log10(1.)

	if (np.log10(1.) < t_a < np.log10(80.)) and (np.log10(1.) < t_i < np.log10(80.)) and (np.log10(1.e-15) < N < np.log10(1.e-8)) and (np.log10(0.5) < a_inj < np.log10(1.)):
		print "Return Something"
		return np.log10(np.array(model_flux))
	else:
		print "Return Inf"
		return np.log10(np.array(model_flux)) * np.Inf

def residuals(p, data):
	
	#t_a, t_i, angle, scale= p
	###t_a, t_i, N = 10.**p
	t_a, t_i, N, a_inj = 10.**p
	#scale= p
	
	#meas_frequencies, meas_flux, meas_flux_rms, redshift, a_in, B_field, model, variant, t_a, t_i = data
	###meas_frequencies, meas_flux, meas_flux_rms, redshift, B_field, a_inj, model, loss = data
	meas_frequencies, meas_flux, meas_flux_rms, redshift, B_field, model, loss = data
	####meas_frequencies = np.array(meas_frequencies)
	meas_flux = np.array(meas_flux)
	
	###s = Synfit(redshift, t_a, t_i, B_field, meas_frequencies, a_inj * -1., N, model, loss)
	###s = Synfit(redshift, t_a, t_i, B_field, meas_frequencies, a_inj, N, model, loss)

	global synfit

	global call_counter
	call_counter += 1
	sys.stdout.write('Synfit call: ' + str(call_counter) + '\r')
	sys.stdout.flush()

	synfit.set_ti(t_i)
	synfit.set_ta(t_a)
	synfit.set_ntot(N)
	synfit.set_inj(a_inj * -1.)

	model_flux = synfit()
	
	#print model_flux
	#print meas_flux
	#print N
	return (meas_flux - model_flux) / meas_flux_rms ## Return logarithm? 2015-11-04

def residuals_leith(p, data):
	
	#t_a, t_i, q, a_inj = 10.**p
	###t_a, q, a_inj = 10.**p # JP, fit for a_inj
	###t_a, q = 10.**p # JP
	t_a, t_i, q = 10.**p # CI_off
	
	#meas_frequencies, meas_flux, meas_flux_rms, redshift, B_field, vol, delta, model = data
	meas_frequencies, meas_flux, meas_flux_rms, redshift, B_field, vol, delta, model, a_inj = data
	meas_frequencies = np.array(meas_frequencies)
	meas_flux = np.array(meas_flux)

	gamma = 1.0 - 2.0 * (a_inj * -1.)
	#gamma = 1.0 - 2.0 * (a_inj) # only for JP Leith age maps fit, or when a_inj is not fitted for

	#print 'T_off: ', t_a, 'T_on: ', t_i, 'Q: ', q

	model_flux = sl.get_fluxes(meas_frequencies, t_a * 1.e6, (t_a + t_i) / t_a, q, gamma, B_field, vol, redshift, delta, model)
	###model_flux = sl.get_fluxes(meas_frequencies, t_a * 1.e6, 1., q, gamma, B_field, vol, redshift, delta, model) # JP model, a_inj fit

	return (meas_flux - model_flux) / meas_flux_rms ## Return logarithm? 2015-11-04

'''
# Input Data

meas_flux = [64.5e-3, 63.3e-3, 63.6e-3, 59.9e-3, 47.9e-3, 52.8e-3, 51.6e-3, 25.2e-3, 9.1e-3, 0.8e-3]
meas_flux_rms = [0.013, 0.013, 0.013, 0.012, 0.01, 0.01, 0.01, 0.001, 6.867e-05, 4.120e-05]
meas_frequencies = [116.9e6, 124.7e6, 132.5e6, 140.3e6, 148.1e6, 155.9e6, 163.8e6, 325.e6, 610.e6, 1425.e6]
'''

def fit(meas_frequencies, meas_flux, meas_flux_rms, model, loss, constant_params, fitted_params):
	# Fitting setup
	# Fit parameters are: t_a, t_i, B, alpha_in and scale factor. We may model B (equipartition) or fix alpha_in. 
	# ts = t_i + t_a
	
	#p0 = [51., 0.05, np.pi / 2.5, 1.e-24] # Initial fitting parameters (variable model parameters)
	#p0 = [100., 100., 1.e-24] # Initial fitting parameters (variable model parameters)
	#p0 = [40., 40., 1.e-24]
	#p0 = [40.0, 1.e-26]
	# Fixed model parameters: a_in, z, B_field
	
	z = constant_params[0]
	B_field = constant_params[1]
	###a_inj = constant_params[2]

	meas_frequencies = np.array(meas_frequencies)
	meas_flux = np.array(meas_flux)
	global synfit 
	synfit = Synfit(z, 0., 0., B_field, meas_frequencies, 0., 0., model, loss)
	
	#fitter = kmpfit.Fitter(residuals=residuals, data=(meas_frequencies, meas_flux, meas_flux_rms, z, a_inj, B_field, model, variant, t_a, t_i))
	###fitter = kmpfit.Fitter(residuals=residuals, data=(meas_frequencies, meas_flux, meas_flux_rms, z, B_field, a_inj, model, loss))
	fitter = kmpfit.Fitter(residuals=residuals, data=(meas_frequencies, meas_flux, meas_flux_rms, z, B_field, model, loss))
	#fitter.parinfo = [{'limits': (50., 120.)}, {'limits': (0.01, 1.)}, {'limits': (0., np.pi / 2.)}, {}]
	#fitter.parinfo = [{'limits': (90., 110.)}, {'limits': (90., 110.)}, {}]
	#fitter.parinfo = [{'limits': (20., 110.)}, {'limits': (1., 10.)}, {}, {'limits': (0.5, 1.5)}]
	##fitter.parinfo = [{'limits': np.log10((10., 150.))}, {'limits': np.log10((0.09, 40.))}, {}]
	#fitter.parinfo = [{'limits': np.log10((5., 80.))}, {'limits': np.log10((5., 80.))}, {'limits: ': np.log10((1.e-8, 1.e-15))}, {'limits: ': np.log10((0.5, 1.))}]
	fitter.parinfo = [{'limits': np.log10((1., 150.))}, {}, {'limits: ': np.log10((1.e-7, 1.e-15))}] # JP model fit
	#fitter.parinfo = [{'limits': np.log10((10., 150.))}, {'fixed': True}, {}]
	#fitter.parinfo = [{}]
	#fitter.parinfo = [{'limits': (5., 500.)}, {'limits': (1., 50.)}, {'limits': (-0.9, 0.)}, {}, {'limits': (1.e-6, 15.e-6)}]
	#fitter.parinfo = [{'limits': np.log10((10., 170.))}, {}, {}, {'limits': np.log10((0.6, 1.5))}]
	fitter.fit(params0=np.log10(fitted_params))
	
	if (fitter.status <= 0): 
	   print "Status:  ", fitter.status
	   print 'error message = ', fitter.errmsg
	   raise SystemExit 
	
	# Rescale the errors to force a reasonable result:
	#err[:] *= np.sqrt(0.9123*fitter.rchi2_min)
	fitter.fit()
	
	print "======== Fit results =========="
	
	prms0 = np.array(fitter.params0)
	prms = np.array(fitter.params)
	#print "Initial params:", 'Source off time: ', prms0[0], ' [Myr], Source on time: ', prms0[1], ' [Myr], Pitch angle: ', prms0[2], ', Scale factor: ', prms0[3]
	#print "Fitted params:", 'Source off time: ', prms[0], ' [Myr], Source on time: ', prms[1], ', [Myr], Pitch angle: ', prms[2], ', Scale factor: ', prms[3]
	#print "Initial params:", 'Source off time: ', prms0[0], ' [Myr], Source on time: ', prms0[1], ' [Myr], Scale factor: ', prms0[2]
	#print "Fitted params:", 'Source off time: ', prms[0], ' [Myr], Source on time: ', prms[1], ' [Myr], Scale factor: ', prms[2]
	
	#print "Initial params:", 'Source off time: ', prms0[0], ' [Myr], Scale factor: ', prms0[1], 'Injection index: ', prms0[2]
	#print "Fitted params:", 'Source off time: ', prms[0], ' [Myr], Scale factor: ', prms[1], 'Injection index: ', prms[2]
	
	print "Initial params:", 'Source off time: ', 10.**prms0[0], ' [Myr], Source on time: ', 10.**prms0[1], ' [Myr]  N_tot: ', 10.**prms0[2], 'Injection index: ', 10.**prms0[3] * -1
	print "Fitted params:", 'Source off time: ', 10.**prms[0], ' [Myr], Source on time: ', 10.**prms[1], ' [Myr] N_tot: ', 10.**prms[2], 'Injection index: ', 10.**prms[3] * -1
	
	###print "Initial params:", 'Source off time: ', 10.**prms0[0], ' [Myr], Source on time: ', 10.**prms0[1], ' [Myr]  N_tot: ', 10.**prms0[2]
	###print "Fitted params:", 'Source off time: ', 10.**prms[0], ' [Myr], Source on time: ', 10.**prms[1], ' [Myr] N_tot: ', 10.**prms[2]
	
	##print "Initial params:", 'Source off time: ', 10.**prms0[0], ' [Myr], Source on time: ', 10.**prms0[1], ' [Myr]  Scale factor: ', 10.**prms0[2],  ' Injection index: ', (10.**prms0[3]) * -1.
	##print "Fitted params:", 'Source off time: ', 10.**prms[0], ' [Myr], Source on time: ', 10.**prms[1], ' [Myr] Scale factor: ', 10.**prms[2], ' injection index: ', (10.**prms[3]) * -1.
	
	#print "Initial params:", 'Source off time: ', prms0[0], ' [Myr], Scale factor: ', prms0[1]
	#print "Fitted params:", 'Source off time: ', prms[0], ' [Myr], Scale factor: ', prms[1]
	
	#print "Initial params:", ' Scale factor: ', prms0[0]
	#print "Fitted params:", ' Scale factor: ', prms[0]
	
	print "Iterations:    ", fitter.niter
	print "Function ev:   ", fitter.nfev 
	print "Uncertainties: ", np.divide(fitter.xerror, np.multiply(np.array([10.**i for i in prms]), np.log(10.)))
	print "Uncertainties_default: ", fitter.xerror
	print "Uncertainties_trans: ", np.log(10.) * fitter.xerror
	print "Uncertainties_pow: ", 10.**(np.log(10.) * fitter.xerror)
	print "Uncertainties_pow_mod: ", (10.**prms) * (np.log(10.) * fitter.xerror)
	print "Uncertainties_onebyone: ", 10.**(np.log(10.) * fitter.xerror[0]), 10.**(np.log(10.) * fitter.xerror[1]), 10.**(np.log(10.) * fitter.xerror[2]), 10.**(np.log(10.) * (fitter.xerror[3] * -1.))
	print "Uncertainties_ord: ", 10.**fitter.xerror
	print "dof:           ", fitter.dof
	print "chi^2, rchi2:  ", fitter.chi2_min, fitter.rchi2_min
	print "stderr:        ", fitter.stderr
	print "Covariance:    ", fitter.covar   
	print "Status:        ", fitter.status
	print "Message        ", fitter.message
	
	print "\n======== Statistics ========"
	
	from scipy.stats import chi2
	rv = chi2(fitter.dof)
	print "Three methods to calculate the right tail cumulative probability:"
	print "1. with gammainc(dof/2,chi2/2):  ", 1-gammainc(0.5*fitter.dof, 0.5*fitter.chi2_min)
	print "2. with scipy's chdtrc(dof,chi2):", chdtrc(fitter.dof,fitter.chi2_min)
	print "3. with scipy's chi2.cdf(chi2):  ", 1-rv.cdf(fitter.chi2_min)
	print ""
	
	
	xc = fitter.chi2_min
	print "Threshold chi-squared at alpha=0.05: ", rv.ppf(1-0.05)
	print "Threshold chi-squared at alpha=0.01: ", rv.ppf(1-0.01)
	
	f = lambda x: -rv.pdf(x)
	x_max = fminbound(f,1,200)
	print """For %d degrees of freedom, the maximum probability in the distribution is
	at chi-squared=%g """%(fitter.dof, x_max)
	
	alpha = 0.05           # Select a p-value
	chi2max = max(3*x_max, fitter.chi2_min)
	chi2_threshold = rv.ppf(1-alpha)
	
	print "For a p-value alpha=%g, we found a threshold chi-squared of %g"%(alpha, chi2_threshold)
	print "The chi-squared of the fit was %g. Therefore: "%fitter.chi2_min 
	if fitter.chi2_min <= chi2_threshold:
	   print "we do NOT reject the hypothesis that the data is consistent with the model"
	else:
	   print "we REJECT the hypothesis that the data is consistent with the model"
	'''
	# Plot data and best fit model
	
	model_frequencies = np.linspace(30.e6, 1.5e9, 500)
	#t_a, t_i, a_in, scale, B_field = prms
	t_a, t_i, a_in, scale = prms
	
	#t_a, t_i, a_in, scale, B_field = 180.25, 28.3, -0.396, 9.8316924e-26, 2.567e-6
	s = Synfit(redshift, t_a, t_i, B_field, model_frequencies, a_in, scale, 0., model, variant)
	model_flux = s()
	
	
	fig = plt.figure()
	axis = fig.add_subplot(111)
	axis.grid()
	
	axis.set_aspect('equal')
	axis.tick_params(axis='x', labelsize=17, pad=15)
	axis.tick_params(axis='y', labelsize=17)
	axis.tick_params(length=10)
	
	data = axis.errorbar(meas_frequencies, meas_flux, meas_flux_rms, markerfacecolor='g', ecolor='g', marker='h', markersize=6, alpha=0.75, linestyle='none')
	data = axis.loglog(model_frequencies, model_flux, '-r', linewidth=2)
	
	xlabel('Frequency [Hz]', fontsize=18, fontweight='bold', color='#000000', labelpad=5)
	ylabel('Flux [mJy]', fontsize=18, fontweight='bold', color='#000000')
	
	#plt.xlim(1.e8, 2.e9)
	#plt.ylim(7.e-4, 7.e-2)
	
	plt.show()
	'''
	return fitter

def fit_leith(meas_frequencies, meas_flux, meas_flux_rms, constant_params, fitted_params):
	
	z = constant_params[0]
	B_field = constant_params[1]
	vol = constant_params[2]
	delta = constant_params[3]
	model = constant_params[4]
	a_inj = constant_params[5]

	meas_frequencies = np.array(meas_frequencies)
	meas_flux = np.array(meas_flux)
	
	###fitter = kmpfit.Fitter(residuals=residuals, data=(meas_frequencies, meas_flux, meas_flux_rms, z, B_field, a_inj, vol, delta))
	###fitter = kmpfit.Fitter(residuals=residuals_leith, data=(meas_frequencies, meas_flux, meas_flux_rms, z, B_field, vol, delta, model))
	fitter = kmpfit.Fitter(residuals=residuals_leith, data=(meas_frequencies, meas_flux, meas_flux_rms, z, B_field, vol, delta, model, a_inj))
	###fitter.parinfo = [{'limits': np.log10((1., 80.))}, {'limits': np.log10((1., 80.))}, {'limits: ': np.log10((2.e-3, 4.e-3))}, {'limits: ': np.log10((0.5, 1.))}]
	
	###fitter.parinfo = [{'limits': np.log10((0.01, 150.))}, {'limits': np.log10((0.01, 70.))}, {}, {'limits: ': np.log10((0.5, 1.))}] # CI_off model fit

	###fitter.parinfo = [{'limits': np.log10((1., 80.))}, {}, {'limits: ': np.log10((0.5, 1.))}] # JP model fit, a_inj too
	fitter.parinfo = [{'limits': np.log10((0.01, 150.))}, {'limits: ': np.log10((0.01, 70.))}, {}] # CI_off model fit
	###fitter.parinfo = [{'limits': np.log10((1., 180.))}, {}] # JP model fit
	fitter.fit(params0=np.log10(fitted_params))
	
	if (fitter.status <= 0): 
	   print "Status:  ", fitter.status
	   print 'error message = ', fitter.errmsg
	   raise SystemExit 
	
	fitter.fit()
	
	print "======== Fit results =========="
	
	prms0 = np.array(fitter.params0)
	prms = np.array(fitter.params)
	
	#print "Initial params:", 'Source off time: ', 10.**prms0[0], ' [Myr], Source on time: ', 10.**prms0[1], ' [Myr]  Q: ', 10.**prms0[2], 'Injection index: ', 10.**prms0[3] * -1
	#print "Fitted params:", 'Source off time: ', 10.**prms[0], ' [Myr], Source on time: ', 10.**prms[1], '[Myr] Q: ', 10.**prms[2], 'Injection index: ', 10.**prms[3] * -1

	###print "Initial params:", 'Source off time: ', 10.**prms0[0], ' [Myr]  Q: ', 10.**prms0[1], 'Injection index: ', 10.**prms0[2] * -1
	###print "Fitted params:", 'Source off time: ', 10.**prms[0], '[Myr] Q: ', 10.**prms[1], 'Injection index: ', 10.**prms[2] * -1

	print "Initial params:", 'Source off time: ', 10.**prms0[0], ' Source on time: ', 10.**prms0[1], ' [Myr]  Q: ', 10.**prms0[2]
	print "Fitted params:", 'Source off time: ', 10.**prms[0], 'Source on time: ', 10.**prms[1], ' [Myr]  Q: ', 10.**prms[2]

	###print "Initial params:", 'Source off time: ', 10.**prms0[0], ' [Myr]  Q: ', 10.**prms0[1]
	###print "Fitted params:", 'Source off time: ', 10.**prms[0], '[Myr] Q: ', 10.**prms[1]
	
	print "Iterations:    ", fitter.niter
	print "Function ev:   ", fitter.nfev 
	print "Uncertainties: ", np.divide(fitter.xerror, np.multiply(np.array([10.**i for i in prms]), np.log(10.)))
	print "Uncertainties_default: ", fitter.xerror
	print "Uncertainties_trans: ", np.log(10.) * fitter.xerror
	print "Uncertainties_pow: ", 10.**(np.log(10.) * fitter.xerror)
	print "Uncertainties_pow_mod: ", (10.**prms) * (np.log(10.) * fitter.xerror)
	###print "Uncertainties_onebyone: ", 10.**(np.log(10.) * fitter.xerror[0]), 10.**(np.log(10.) * fitter.xerror[1]), 10.**(np.log(10.) * fitter.xerror[2]), 10.**(np.log(10.) * (fitter.xerror[3] * -1.))
	###print "Uncertainties_onebyone: ", 10.**(np.log(10.) * fitter.xerror[0]), 10.**(np.log(10.) * fitter.xerror[1]), 10.**(np.log(10.) * fitter.xerror[2])
	'''
	print "Uncertainties_onebyone: ", 10.**(np.log(10.) * fitter.xerror[0]), 10.**(np.log(10.) * fitter.xerror[1]), 10.**(np.log(10.) * fitter.xerror[2])
	print "Uncertainties_ord: ", 10.**fitter.xerror
	print "dof:           ", fitter.dof
	print "chi^2, rchi2:  ", fitter.chi2_min, fitter.rchi2_min
	print "1-sigma error in parameter estimates:        ", (10.**prms[0]) * (np.log(10.) * fitter.stderr[0]), (10.**prms[1]) * (np.log(10.) * fitter.stderr[1]), (10.**prms[2]) * (np.log(10.) * fitter.stderr[2])

	print "1-sigma error in parameter estimates (xerr):        ", (10.**prms[0]) * (np.log(10.) * fitter.xerror[0]), (10.**prms[1]) * (np.log(10.) * fitter.xerror[1]), (10.**prms[2]) * (np.log(10.) * fitter.xerror[2])

	print "1-sigma error in parameter estimates (uncorrected):        ", fitter.stderr[0], fitter.stderr[1], fitter.stderr[2]
	print "1-sigma error in parameter estimates (pow):        ", 10.**fitter.stderr[0], 10.**fitter.stderr[1], 10.**fitter.stderr[2]

	print "1-sigma error in parameter estimates (inverse):        ", (10.**prms[0]) * np.log(10.) * fitter.stderr[0]

	print "Uncertainities stderr:        ", 10.**fitter.xerror[0], 10.**fitter.xerror[1], 10.**fitter.xerror[2]
	'''
	print "Covariance:    ", fitter.covar   
	print "Status:        ", fitter.status
	print "Message        ", fitter.message
	
	print "\n======== Statistics ========"
	
	from scipy.stats import chi2
	rv = chi2(fitter.dof)
	print "Three methods to calculate the right tail cumulative probability:"
	print "1. with gammainc(dof/2,chi2/2):  ", 1-gammainc(0.5*fitter.dof, 0.5*fitter.chi2_min)
	print "2. with scipy's chdtrc(dof,chi2):", chdtrc(fitter.dof,fitter.chi2_min)
	print "3. with scipy's chi2.cdf(chi2):  ", 1-rv.cdf(fitter.chi2_min)
	print ""
	
	
	xc = fitter.chi2_min
	print "Threshold chi-squared at alpha=0.05: ", rv.ppf(1-0.05)
	print "Threshold chi-squared at alpha=0.01: ", rv.ppf(1-0.01)
	
	f = lambda x: -rv.pdf(x)
	x_max = fminbound(f,1,200)
	print """For %d degrees of freedom, the maximum probability in the distribution is
	at chi-squared=%g """%(fitter.dof, x_max)
	
	alpha = 0.001          # Select a p-value
	chi2max = max(3*x_max, fitter.chi2_min)
	chi2_threshold = rv.ppf(1-alpha)
	
	print "For a p-value alpha=%g, we found a threshold chi-squared of %g"%(alpha, chi2_threshold)
	print "The chi-squared of the fit was %g. Therefore: "%fitter.chi2_min 
	if fitter.chi2_min <= chi2_threshold:
	   print "we do NOT reject the hypothesis that the data is consistent with the model"
	else:
	   print "we REJECT the hypothesis that the data is consistent with the model"

	return fitter

def fit_regions_age(region_data_path, filename, N_pix_beam, N_pix_sigma, noise_arr, freq_arr, z, model, variant):
	
	'''
	with open(path + filename, 'r') as f:
		content = f.readlines()
		for line in content:
			line.strip()
			if not line.startswith('#'):
				np_arr =  np.(line[1 :])
				print len(np_arr)
	'''
	
	fit_res = fit([74.e6, 135.e6, 325.e6, 610.e6, 1425.e6], [3.65, 1953.5e-3, 364.8e-3, 109.8e-3, 14.7e-3], [3.6602447133273364, 0.39078236526415172, 0.072961526321674436, 0.021964107215696166, 0.0029494543219208613], 0.1559, -0.75, 1.75, 'CI_off', True) # 1 HBA and LBA point, integrated spectrum
	
	'''
	regions = np.genfromtxt(path + filename, comments='#')
	noise_area = N_pix_sigma / N_pix_beam
	fit_regions_res = []
	
	#region = regions[6]
	for region in regions:
		region_area = region[1] / N_pix_beam
		#print 'Input noise: ', noise_arr
		for idx in range(len(noise_arr)):
			if idx in range(0, 7):
				flux_rms = region[idx + 2] * 0.2
			elif idx in range(7, 8):
				flux_rms = region[idx + 2] * 0.05
			else:
				flux_rms = 0.
			noise_arr[idx] = np.sqrt((((noise_arr[idx]**2.) * region_area**2.) / noise_area) + ((noise_arr[idx]**2.) / region_area) + flux_rms**2.)
		#print 'Corrected noise:', noise_arr
		print 'Fitting model for region ', region[0], '...'
		
		# B field calculated per region, with the path through the source taken to be the average of the overall source dimensions - 239.85 kpc
		#print 'The magnetic field for region ', region[0], ' is: ', B_field_estimator(1., 1., 0.1599, 10., 10., 239.85, np.pi/2., region[8], 0.325, 0.01, 100., region[11]), ' G'
		
		a_in = -0.7 # Injection spectral index
		
		fit_res = fit(freq_arr, region[2:11], noise_arr, z, a_in, region[12], model, variant)
		fit_regions_res.append(region[0])
		fit_regions_res.append(fit_res.params[1])
		fit_regions_res.append(fit_res.params[0])
		fit_regions_res.append(fit_res.rchi2_min)
		'''
	return fit_res
	

def fit_pixels(path, images, N_pix_beam, N_pix_sigma, noise_arr, freq_arr):
	from astropy.io import fits
	import pyfits
	import aplpy
	import matplotlib.pyplot as plt
	import aplpy
	
	flux_arr = []
	im_dict = dict()
	header = []
	
	for im_idx in range(len(images)):	
		hdulist = fits.open(path + images[im_idx], do_not_scale_image_data=True)
		im_dict[im_idx] = hdulist[0]
		if im_idx == 0:
			header =  hdulist[0].header
		
		
	#hdulist.info()
	#header =  hdulist[0].header
	#print hdulist[0]
	#scidata = hdulist[0].data
	#print scidata.shape
	#print scidata[0][0][50:100,50:100].shape
	#scidata = scidata[0][0][50:350,50:350]
	#print scidata.shape
	#scidata.shape()
	#print scidata
	#new_hdu = fits.PrimaryHDU(scidata[0][0][50:100,50:100])
		
	keys = im_dict.keys()
	#im_noise_arr = [0.135, 0.0014, 0.0014, 0.0005] # noise measured in an empty region in each image
	# Calculate the scaled measurement error
	noise_area = N_pix_sigma / N_pix_beam
	target_area = 1. / N_pix_beam
	noise_arr_bckp = []
	for idx in range(len(noise_arr)):
		noise_arr_bckp.append(noise_arr[idx]) 
		noise_arr[idx] = np.sqrt((((noise_arr[idx]**2.) * target_area**2.) / noise_area) + ((noise_arr[idx]**2.) / target_area))
	#print 'Scaled noise: ', noise_arr
	#freq_arr = [127.e6, 325.e6, 610.e6, 1425.e6]
	#freq_arr = [127.e6, 325.e6]
	#freq_arr = [325.e6, 610.e6]
	
	specdata0 = im_dict[keys[0]].data[0][0]
	specdata1 = im_dict[keys[1]].data[0][0]
	specdata2 = im_dict[keys[2]].data[0][0]
	#specdata3 = im_dict[keys[3]].data[0]
	
	islands = where(specdata0 > 0.015, specdata0, 0.)
	islands1 = where(islands > 0.11, islands, 0.)
	
	#gauss = spn.filters.gaussian_filter(islands, sigma=3)
	imshow(islands1, vmin=0.001, vmax=0.1)
	plt.show()
	
	'''
	islands = where(specdata0 > 0.015, specdata0, 0.)
	mask1 = where(specdata0 > 0.15, 2, 0)
	mask2 = where((specdata0 < 0.15) and (specdata0 > 0.11), 1, 0)
	mask = mask1 + mask2
	mask[0, 0] = 0
	imshow(islands, vmin=0.001, vmax=0.01)
	plt.show()
	imshow(mask, vmin=0.001, vmax=0.01)
	plt.show()
	
	watershed = spm.watershed_ift(uint8(islands), mask)
	
	imshow(watershed, vmin=0.001, vmax=0.01)
	plt.show()
	'''
	
	#specdata = specdata0
	'''
	inj_age = specdata0
	rel_age = specdata1
	chisq_red = specdata2
	'''
	
	'''
	color1,color2 = [],[]
	
	factor = 0.05
	num_pix = 0.
	curr_pix = 0.
	
	# Calculate the number of pixels in the run
	for col in range(len(specdata1[0][:])):
		for row in range(len(specdata1[:][0])):			
			for idx in range(len(noise_arr)):
				if idx < 6:
					noise_arr[idx] = np.sqrt(noise_arr[idx]**2. + (0.2 * (im_dict[keys[idx]].data[0][0])[col, row])**2.) # scale the LOFAR flux error by 20% in quadrature
					#if idx==0: print 'Scaled noise after flux scaling: ', noise_arr[0], 'Flux: ', (im_dict[keys[idx]].data[0][0])[col, row]
				else:
					if idx == 7:
						noise_arr[idx] = np.sqrt(noise_arr[idx]**2. + (0.05 * (im_dict[keys[idx]].data[0][0])[col, row])**2.) # scale the GMRT flux error by 5% in quadrature
					else:
						noise_arr[idx] = np.sqrt(noise_arr[idx]**2. + (0.05 * (im_dict[keys[idx]].data[0])[col, row])**2.) # scale the GMRT flux error by 5% in quadrature
			
			#print 'Flux: ', (im_dict[keys[0]].data[0][0])[col, row]
			#print 'Noise: ', noise_arr[0]
			
			if (im_dict[keys[0]].data[0][0])[col, row] > factor * noise_arr[0] and (im_dict[keys[1]].data[0][0])[col, row] > factor * noise_arr[1] and (im_dict[keys[2]].data[0][0])[col, row] > factor * noise_arr[2] and (im_dict[keys[3]].data[0][0])[col, row] > factor * noise_arr[3] and (im_dict[keys[4]].data[0][0])[col, row] > factor * noise_arr[4] and (im_dict[keys[5]].data[0][0])[col, row] > factor * noise_arr[5] and (im_dict[keys[6]].data[0])[col, row] > factor * noise_arr[6] and (im_dict[keys[7]].data[0][0])[col, row] > factor * noise_arr[7] and (im_dict[keys[8]].data[0])[col, row] > factor * noise_arr[8]:
				num_pix += 1.
	
	noise_arr = noise_arr_bckp
	
	for idx in range(len(noise_arr)):
		noise_arr[idx] = np.sqrt((((noise_arr[idx]**2.) * target_area**2.) / noise_area) + ((noise_arr[idx]**2.) / target_area))
	
	for col in range(len(specdata1[0][:])):
		for row in range(len(specdata1[:][0])):			
			for idx in range(len(noise_arr)):
				if idx < 6:
					noise_arr[idx] = np.sqrt(noise_arr[idx]**2. + (0.2 * (im_dict[keys[idx]].data[0][0])[col, row])**2.) # scale the LOFAR flux error by 20% in quadrature
					#if idx==0: print 'Scaled noise after flux scaling: ', noise_arr[0], 'Flux: ', (im_dict[keys[idx]].data[0][0])[col, row]
				else:
					if idx == 7:
						noise_arr[idx] = np.sqrt(noise_arr[idx]**2. + (0.05 * (im_dict[keys[idx]].data[0][0])[col, row])**2.) # scale the GMRT flux error by 5% in quadrature
					else:
						noise_arr[idx] = np.sqrt(noise_arr[idx]**2. + (0.05 * (im_dict[keys[idx]].data[0])[col, row])**2.) # scale the GMRT flux error by 5% in quadrature
			
			#print 'Flux: ', (im_dict[keys[0]].data[0][0])[col, row]
			#print 'Noise: ', noise_arr[0]
			
			if (im_dict[keys[0]].data[0][0])[col, row] > factor * noise_arr[0] and (im_dict[keys[1]].data[0][0])[col, row] > factor * noise_arr[1] and (im_dict[keys[2]].data[0][0])[col, row] > factor * noise_arr[2] and (im_dict[keys[3]].data[0][0])[col, row] > factor * noise_arr[3] and (im_dict[keys[4]].data[0][0])[col, row] > factor * noise_arr[4] and (im_dict[keys[5]].data[0][0])[col, row] > factor * noise_arr[5] and (im_dict[keys[6]].data[0])[col, row] > factor * noise_arr[6] and (im_dict[keys[7]].data[0][0])[col, row] > factor * noise_arr[7] and (im_dict[keys[8]].data[0])[col, row] > factor * noise_arr[8]:
				
			#if (specdata1[col,row] > im_noise_arr[1]) and (specdata2[col,row] > im_noise_arr[2]):
			#if (specdata0[col,row] > im_noise_arr[0]) and (specdata1[col,row] > im_noise_arr[1]):
				flux_arr = []
				# Enable for age maps
				for idx in range(len(noise_arr)):
					if idx < 6:
						flux_arr.append((im_dict[keys[idx]].data[0][0])[col, row])
					else:
						if idx == 7:
							flux_arr.append((im_dict[keys[idx]].data[0][0])[col, row])
						else:
							flux_arr.append((im_dict[keys[idx]].data[0])[col, row])
				
				#inj_age[col, row] = flux_arr[0]
				fit_res = fit(freq_arr, flux_arr, noise_arr, 'JP', True)
				inj_age[col, row] = fit_res.params[1]
				rel_age[col, row] = fit_res.params[0]
				chisq_red[col, row] = fit_res.rchi2_min
				#print fit_res[1], fit_res[0]
				
				# Enable for spcetral index maps
				#flux_arr = [specdata0[col,row], specdata1[col,row], specdata2[col,row], specdata3[col,row]]
				#flux_arr = [specdata1[col,row], specdata2[col,row]]
				#flux_arr = [specdata0[col,row], specdata1[col,row]]
				#specdata[col,row] = np.polyfit(np.log10(freq_arr), np.log10(flux_arr), 1)[0]
				
				# Enable for color-color plot
				#color1.append(np.polyfit(np.log10(freq_arr[:2]), np.log10(flux_arr[:2]), 1)[0])
				#color2.append(np.polyfit(np.log10(freq_arr[1:]), np.log10(flux_arr[1:]), 1)[0])
				
				curr_pix += 1.
				pct_done = int(curr_pix * 10. / num_pix)
				sys.stdout.write('\r[{0}] {1}%'.format('#' * (pct_done * 2), pct_done * 10))
				sys.stdout.flush()
				
			else:
				#specdata[col,row] = NaN
				inj_age[col,row] = NaN
				rel_age[col,row] = NaN
				chisq_red[col, row] = NaN
	'''
	# Enable for spectral maps	
	'''
	new_hdu = fits.PrimaryHDU(specdata)
	new_hdu.header = header
	new_hdu.writeto(path + "temp.fits", clobber=True)
	im = aplpy.FITSFigure(path + "temp.fits") # Will not read <astropy.io.fits.hdu.image.PrimaryHDU object>, so we have to use a tmp fits file
	#im.show_grayscale()
	im.show_colorscale(cmap='jet', stretch='linear', vmin=-3.5, vmax=0.5)
	im.add_colorbar()
	im.add_beam()
	'''
	'''
	# Enable for age maps
	
	inj_hdu = fits.PrimaryHDU(inj_age)
	inj_hdu.header = header
	inj_hdu.writeto(path + "temp_inj.fits", clobber=True)
	inj_map = aplpy.FITSFigure(path + "temp_inj.fits") # Will not read <astropy.io.fits.hdu.image.PrimaryHDU object>, so we have to use a tmp fits file
	#inj_map.show_grayscale()
	#inj_map.show_colorscale(cmap='jet', stretch='linear')
	#inj_map.add_colorbar()
	#inj_map.add_beam()
	
	rel_hdu = fits.PrimaryHDU(rel_age)
	rel_hdu.header = header
	rel_hdu.writeto(path + "temp_rel.fits", clobber=True)
	#rel_map = aplpy.FITSFigure(path + "temp_rel.fits") # Will not read <astropy.io.fits.hdu.image.PrimaryHDU object>, so we have to use a tmp fits file
	#rel_map.show_grayscale()
	#rel_map.show_colorscale(cmap='jet', stretch='linear')
	#rel_map.add_colorbar()
	#rel_map.add_beam()
	
	chr_hdu = fits.PrimaryHDU(chisq_red)
	chr_hdu.header = header
	chr_hdu.writeto(path + "temp_chr.fits", clobber=True)
	
	#data = plt.plot(color1, color2, 'or')
	#plt.show()
	'''

	hdulist.close()

def show_images(path):
	import aplpy
	rel_map = aplpy.FITSFigure(path + "temp_rel.fits") # Will not read <astropy.io.fits.hdu.image.PrimaryHDU object>, so we have to use a tmp fits file
	#rel_map.show_grayscale()
	rel_map.show_colorscale(cmap='jet', stretch='linear')
	rel_map.add_colorbar()
	rel_map.add_beam()

	inj_map = aplpy.FITSFigure(path + "temp_inj.fits") # Will not read <astropy.io.fits.hdu.image.PrimaryHDU object>, so we have to use a tmp fits file
	#inj_map.show_grayscale()
	inj_map.show_colorscale(cmap='jet', stretch='linear')
	inj_map.add_colorbar()
	inj_map.add_beam()

	plt.show()	

#	Calculate the magnetic field in a plasma from equipartition assumptions (Miley, 1980)
#	k		- ratio of the energy contained in heavy particles vs. that in electrons
#	eta		- filling factor of the emitting regions
#	z		- redshift
#	th_x	- equivalent beam width or source component size in arcsec in direction x
#	th_y	- equivalent beam width or source component size in arcsec in direction y
#	s		- path length through the source (kpc) in the line of sight
#	phi		- angle between the uniform magnetic field and the line of sight
#	F_0		- flux density (Jy) or brightness (Jy / beam) of the source region at frequency nu_0
#	nu_0	- frequency of measurement in GHz
#	nu_1	- lower cutoff frequency in GHz
#	nu_2	- upper cutoff frequency in GHz
#	sp_in	- spectral index (F(nu) ~ nu^sp_in, nu_1 < nu < nu_2)
#	gamma_min	- Lorentz factor of lowest energy electrons
#
# returns the strength of the equipartition magnetic field in Gauss 

def B_field_estimator(k, eta, z, th_x, th_y, s, phi, F_0, nu_0, nu_1, nu_2, sp_in, gamma_min):
	
	B_eq = 5.69e-5 * (((1. + k) / eta) * ((1. + z)**(3. - sp_in)) * (1. / (th_x * th_y * s * np.sin(phi)**(3. / 2.))) * (F_0 / (nu_0**sp_in)) * (((nu_2**(sp_in + 1. / 2.)) - (nu_1**(sp_in + 1. / 2.))) / (sp_in + (1. / 2.))))**(2. / 7.)
	
	print 'Equipartition magnetic fied: B_eq = ', B_eq, ' G'
	
	sp_in = -sp_in
	B_eq_corr = 1.1 * (gamma_min**((1. - 2. * sp_in) / (3. + sp_in))) * (B_eq**(7. / (6. + 2. * sp_in)))	
	
	print 'Corrected equipartition magnetic fied: B_eq_corr = ', B_eq_corr, ' G'
	
	B_CMB = 3.25 * (1. + z)**2. * 1.e-6
	B_min = B_CMB / np.sqrt(3.)
	
	print 'Minimum magnetic field (for maximum particle age): B_max = ', B_min, ' G'
	
	B_IC = np.sqrt(2. / 3.) * B_CMB
	
	print 'IC equivalent magnetic fied: B_IC = ', B_IC, ' G'


	print 'CMB equivalent magnetuc filed B_CMB = ', B_CMB, ' G'


#	Calculate the magnetic field in a plasma from equipartition assumptions (Govoni, 2004)
#	k			- ratio of the energy contained in heavy particles vs. that in electrons
#	z			- redshift
#	s			- path length through the source (kpc) in the line of sight
#	F_0			- Surface brightness (mJy / arcsec^2) of the source region at frequency nu_0
#	nu_0		- frequency of measurement in MHz
#	nu_1		- lower cutoff frequency in Hz
#	nu_2		- upper cutoff frequency in Hz	
#	sp_in		- spectral index (F(nu) ~ nu^sp_in, nu_1 < nu < nu_2)
#	gamma_min	- Lorentz factor of lowest energy electrons
#
# returns the strength of the equipartition magnetic field in Gauss 

def B_field_estimator_1(k, z, s, F_0, nu_0, nu_1, nu_2, sp_in, gamma_min):
	sp_in = np.abs(sp_in)
	print 'spix = ', sp_in
	ksi = ((2. * sp_in - 2.) / (2. * sp_in - 1.)) * ((nu_1**((1. - 2. * sp_in) / 2.) - nu_2**((1. - 2. * sp_in) / 2.)) / (nu_1**(1. - sp_in) - nu_2**(1. - sp_in)))
	ksi = 1.72e-12
	u_min = ksi * ((1. + k)**(4. / 7.)) * nu_0**((4. * sp_in) / 7.) * (1. + z)**((12. + (4. * sp_in)) / 7.) * F_0**(4. / 7.) * s**(-4. / 7.)
	
	print 'ksi = ', ksi
	B_eq = np.sqrt(24. * np.pi * u_min / 7.)
	
	print 'Equipartition magnetic fied: B_eq = ', B_eq, ' G'
	
	sp_in = -sp_in
	B_eq_corr = 1.1 * (gamma_min**((1. - 2. * sp_in) / (3. + sp_in))) * (B_eq**(7. / (6. + 2. * sp_in)))
	
	print 'Corrected equipartition magnetic fied: B_eq_corr = ', B_eq_corr, ' G'

def plotModelFit(redshift, meas_frequencies, a_in, model, variant):
	
	# JP model, CI + aging
	#t_a = [109.27, 111.85, 79.04, 109.34, 109.87, 76.60, 116.91, 160.35, 52.70, 85.74, 108.18, 109.00, 107.42, 74.58, 82.12, 67.03, 72.61]
	#t_i = [0.1, 0.1, 0.53, 0.1, 0.1, 0.58, 0.29, 0.21, 0.44, 0.86, 0.1, 0.1, 0.1, 0.85, 0.65, 0.42, 0.10]
	#t_a = [109.27, 111.85, 79.04, 109.34, 109.87, 76.60, 500.0, 160.35, 52.70, 85.74, 108.18, 109.00, 107.42, 74.58, 82.12, 67.03, 72.61, 96.94]
	#t_i = [0.1, 0.1, 0.53, 0.1, 0.1, 0.58, 422.70234163, 0.21, 0.44, 0.86, 0.1, 0.1, 0.1, 0.85, 0.65, 0.42, 0.10, 0.028]
	#scale = [-1.98994191798e-25, -2.09157957122e-25, 8.93294642352e-26, -1.89206245098e-25, -2.05614089832e-25, 8.36171642773e-26, 4.99058280421e-28, 3.70010017002e-25, 8.2680854393e-26, 6.2498342171e-26, -2.26962884483e-25, -2.45551230387e-25, -2.60744556032e-25, 9.99486468129e-27, 1.21748029245e-26, 3.8872384655e-26, 1.10807781949e-25, 1.94971399534e-24]
	t_a = 101.91
	t_i = 17.87
	scale = 0.05e-23
	model_frequencies = linspace(1.e8, 1.5e9, 10)
	angle = 1.25663706144
	
	path = '/Users/users/shulevski/brooks/Research_Vault/1431+1331_spix/Images_Feb_2014/'
	filename_1 = 'region_data.txt'
	filename_2 = 'region_rms.txt'
	regions = np.genfromtxt(path + filename_1, comments='#')
	rms = np.genfromtxt(path + filename_2, comments='#')
	
	#for idx in range(len(regions)):
		#if scale[idx] > 0.:
	fig = plt.figure()
	axis = fig.add_subplot(111)
	axis.grid()
	axis.tick_params(axis='x', labelsize=17)
	axis.tick_params(axis='y', labelsize=17)
	axis.tick_params(length=10)
	axis.set_aspect('equal')
	
	#s = Synfit(redshift, t_a[idx], t_i[idx], regions[idx][12], model_frequencies, a_in, scale[idx], angle, model, variant)
	s = Synfit(redshift, t_a, t_i, 4.e-6, model_frequencies, a_in, scale, angle, model, variant)
	model_flux = s()
	#meas_flux = regions[idx][2:11]
	meas_flux = [6.90e-2, 1.61e-2, 6.86e-3, 1.38e-3]
	rms = [0.0139078986086372, 0.0032216059330584515, 0.00035500461717153059, 5.0854665266358703e-05]
	
	#data = axis.errorbar(meas_frequencies, meas_flux, rms[idx][2:], markerfacecolor='g', ecolor='g', marker='h', markersize=6, alpha=0.75, linestyle='none')
	data = axis.errorbar(meas_frequencies, meas_flux, rms, markerfacecolor='g', ecolor='g', marker='h', markersize=6, alpha=0.75, linestyle='none')
	data = axis.loglog(model_frequencies, model_flux, '-r')
	
	xlabel('Frequency [Hz]', fontsize=18, fontweight='bold', color='#000000')
	ylabel('Flux [Jy]', fontsize=18, fontweight='bold', color='#000000')

	title('', fontsize=20, fontweight='bold')
	plt.xlim(4.e7, 3.e9)
	plt.ylim(1.e-4, 4.0)
	plt.show()
	
def explore_fit_space(mfr, mfl, mfle):
	
	t_a = np.logspace(1., 8.2, 200)
	t_i = np.logspace(1., 8.2, 200)
	
	x, y = np.meshgrid(t_a, t_i)
	
	chisq = np.zeros([len(t_a), len(t_a)])
	for i in range(len(t_a)):
		for j in range(len(t_i)): 
			print 'Fitting ', i, 'x', j
			fitter = fit(mfr, mfl, mfle, 0.1599, -0.7, 4.25e-6, 'CI_off', True, t_a[i] / 1.e6, t_i[j] / 1.e6)
			print 'T_a: ', i, 'T_i: ', j
			chisq[i, j] = fitter.chi2_min
			#chisq[i, j] = np.random.randn(1)

	#print chisq.index(np.min(chisq))
	#print chisq.index(np.max(chisq))
	#imshow(chisq)
	plt.figure()
	minim = np.amin(chisq)
	maxim = np.amax(chisq)
	cs = plt.contour(x, y, chisq)
	plt.clabel(cs, inline=1, fontsize=10)
	cb = plt.colorbar(cs, shrink=0.8, extend='both')
	plt.show()
	
def generate_model_fluxes(redshift, t_a, t_i, B_field, model_frequencies, a_in, scale, angle, model, variant):
	
	s = Synfit(redshift, t_a, t_i, B_field, model_frequencies, a_in, scale, angle, model, variant)
	print s()
	'''
	model_fluxes = s()
	fig = plt.figure()
	axis = fig.add_subplot(111)
	axis.grid()
	axis.tick_params(axis='x', labelsize=17)
	axis.tick_params(axis='y', labelsize=17)
	axis.tick_params(length=10)
	axis.set_aspect('equal')

	data = axis.loglog(model_frequencies, model_fluxes, '-r')
	
	xlabel('Frequency [Hz]', fontsize=18, fontweight='bold', color='#000000')
	ylabel('Flux [Jy]', fontsize=18, fontweight='bold', color='#000000')

	title('Model', fontsize=20, fontweight='bold')
	plt.xlim(4.e7, 3.e9)
	plt.ylim(1.e-4, 4.0)
	plt.show()
	'''

# B [\muG], nu_b [GHz]
def age_estimate(B, z, nu_b):
	
	B_IC = 3.25 * (1. + z)**2.
	
	return 1590. * ((B**0.5) / ((B**2. + B_IC**2.) * ((1. + z) * nu_b)**0.5))
	
# B [\muG], nu_b [GHz], t_s = t_a + t_i [Myr]
def break_estimate(B, z, t_s):
	
	B_IC = 3.25 * (1. + z)**2.
	
	return B / ((1. + z) * ((t_s / 1590.) * (B**2. + B_IC**2.))**2.)
	
def convert_data_synage():
	freq_arr = [120.e6, 127.e6, 135.e6, 145.e6, 154.e6, 164.e6, 325.e6, 610.e6, 1425.e6]
	path = '/Users/users/shulevski/brooks/Research_Vault/1431+1331_spix/Images_Feb_2014/'
	fluxfile = 'region_data.txt'
	rmsfile = 'region_rms.txt'
	flux_regions = np.genfromtxt(path + fluxfile, comments='#')
	rms_regions = np.genfromtxt(path + rmsfile, comments='#')
	
	f = open('/Users/users/shulevski/Desktop/Synage_input_regions_allLP.tab', 'w')
	f.write('freq_units: MHz\nflux_units: Jy\n\n')
	for flr, rmr in zip(flux_regions, rms_regions):
		f.write('point ' + str(int(flr[0]) - 1) + '\nlabel: Region' + str(int(flr[0])) + '\n\n')
		f.write(str(freq_arr[0] / 1.e6) + ' ' + str(flr[2]) + ' ' + str(rmr[2]) + '\n')
		f.write(str(freq_arr[1] / 1.e6) + ' ' + str(flr[3]) + ' ' + str(rmr[3]) + '\n')
		f.write(str(freq_arr[2] / 1.e6) + ' ' + str(flr[4]) + ' ' + str(rmr[4]) + '\n')
		f.write(str(freq_arr[3] / 1.e6) + ' ' + str(flr[5]) + ' ' + str(rmr[5]) + '\n')
		f.write(str(freq_arr[4] / 1.e6) + ' ' + str(flr[6]) + ' ' + str(rmr[6]) + '\n')
		f.write(str(freq_arr[5] / 1.e6) + ' ' + str(flr[7]) + ' ' + str(rmr[7]) + '\n')
		f.write(str(freq_arr[6] / 1.e6) + ' ' + str(flr[8]) + ' ' + str(rmr[8]) + '\n')
		f.write(str(freq_arr[7] / 1.e6) + ' ' + str(flr[9]) + ' ' + str(rmr[9]) + '\n')
		f.write(str(freq_arr[8] / 1.e6) + ' ' + str(flr[10]) + ' ' + str(rmr[10]) + '\n\n')
		
def read_synage_model_compute_ages():
	
	# CIoff_fixed_07_allLP
	#path = '/Users/users/shulevski/Desktop/all_fits_synage/CIoff-fixed_07_allLP/'
	#modelfile = 'regions_allLP_CIOFF_0.7fixed.txt'
	
	# CIoff_fixed_07_oneLP
	#path = '/Users/users/shulevski/Desktop/all_fits_synage/CIoff-fixed_07_oneLP/'
	#modelfile = 'regions_oneLP_CIOFF_0.7fixed.txt'
	
	# CIoff_free_allLP
	#path = '/Users/users/shulevski/Desktop/all_fits_synage/CIoff-free_allLP/'
	#modelfile = 'regions_allLP_CIOFF.txt'
	
	# JP_free_allLP
	path = '/Users/users/shulevski/Desktop/all_fits_synage/JP-free_allLP/'
	modelfile = 'regions_allLP_JP.txt'
	
	# JP_fixed_allLP
	#path = '/Users/users/shulevski/Desktop/all_fits_synage/JP-fixed_07_allLP/'
	#modelfile = 'regions_allLP_JP_0.7fixed.txt'
	
	# JP_fixed_oneLP
	#path = '/Users/users/shulevski/Desktop/all_fits_synage/JP-fixed_07_oneLP/'
	#modelfile = 'regions_oneLP_JP_0.7fixed.txt'
	
	model_regions = np.genfromtxt(path + modelfile, comments='#')
	
	# NE: 70" x 50", alpha = -2.04
	B_NE = B_field_estimator(1., 1., 0.1599, 50., 70., 5.6e3, np.pi / 2., 0.29, 0.325, 0.01, 100., -2.04)
	
	# SW: 23" x 25", alpha = -1.66
	B_SW = B_field_estimator(1., 1., 0.1599, 25., 23., 76.54, np.pi / 2., 3.01e-2, 0.325, 0.01, 100., -1.66)
	print 'B_SW: ', B_SW, 'B_NE: ', B_NE, 'B_4C: ', B_field_estimator(1., 1., 0.1599, 50., 70., 5.6e3, np.pi / 2., 364.8e-3, 0.325, 0.01, 100., -1.76), 'Age 4C: ', age_estimate(1.46, 0.1559, 74 * 1.e-3)
	t_arr = []
	t_arr_e_lo = []
	t_arr_e_hi = []
	toff_arr = []
	toff_arr_e_lo = []
	toff_arr_e_hi = []
	
	for region in model_regions:
		if int(region[0]) + 1 <= 11:
			B = B_NE
		else:
			B = B_SW
		t = age_estimate(float(B) * 1.e6, 0.1559, float(region[5]) * 1.e-3)
		t_e_lo = age_estimate(float(B) * 1.e6, 0.1559, (float(region[5]) + float(region[6])) * 1.e-3)
		t_e_hi = age_estimate(float(B) * 1.e6, 0.1559, (float(region[5]) + float(region[7])) * 1.e-3)
		t_arr.append(t)
		t_arr_e_lo.append(t_e_lo)
		t_arr_e_hi.append(t_e_hi)
		print 'Region ', int(region[0]) + 1, ' is ', t, ' Myr old, + ', t_e_lo, ' - ', t_e_hi, ' Myr.' 
		if 'CI' in modelfile:
			t_off = float(region[14]) * t
			t_off_e_lo = abs((float(region[15]))) * t
			t_off_e_hi = float(region[16]) * t
			toff_arr.append(t_off)
			toff_arr_e_lo.append(t_off_e_lo)
			toff_arr_e_hi.append(t_off_e_hi)
			print 'T_OFF: ', t_off, ' Myr - ', t_off_e_lo, ' Myr + ', t_off_e_hi, 'Myr'
	print t_arr
	print t_arr_e_lo
	print t_arr_e_hi
	if 'CI' in modelfile:
		print toff_arr
		print toff_arr_e_lo
		print toff_arr_e_hi

def color_shift():
	
	'''
	#J1431.8+1331
	freq_arr = np.array([120.e6, 127.e6, 135.e6, 145.e6, 154.e6, 164.e6, 325.e6, 610.e6, 1425.e6])
	#path = '/Users/users/shulevski/brooks/Research_Vault/1431+1331_spix/Images_Feb_2014/'
	path = '/Users/shulevski/Documents/Kapteyn/1431+1331_spix/Images_Feb_2014/'
	fluxfile = 'region_data.txt'
	rmsfile = 'region_rms.txt'
	flux_regions = np.genfromtxt(path + fluxfile, comments='#')
	rms_regions = np.genfromtxt(path + rmsfile, comments='#')
	
	dn_select = [2, 3, 6]
	up_select = [6, 7, 8]
	'''
	
	#'''
	#B2 0924+30
	#path = '/Users/users/shulevski/brooks/Research_Vault/B20924+30/August_2014/HBA_Low_Res/Spix/'
	#fluxfile = 'region_data.txt'
	
	#freq_arr = np.array([132., 136., 160., 163., 167., 609., 1400.]) * 1.e6
	#flux_regions = np.genfromtxt(path + fluxfile, comments='#')
	#'''

	path = '/Users/shulevski/Documents/Kapteyn/B20924+30_spix/'
	fluxfile = 'region_data_new.txt'
	'''
	freq = np.array([113., 132., 136., 159., 163., 167., 609., 1400.]) * 1.e6
	flux_regions = np.genfromtxt(path + fluxfile, comments='#')
	
	mask = np.array([1., 1., 1., 1., 1., 1., 0., 0.]) * 1.e-3
	beam_pix_num = np.array([23., 23., 23., 23., 23., 23., 27., 28.])
	rms_pix_num = np.array([1392., 1392., 1392., 1392., 1392.,1392., 1757., 1757.])
	'''
	freq_arr = np.array([113., 132., 136., 159., 163., 167., 609., 1400.]) * 1.e6
	flux_regions = np.genfromtxt(path + fluxfile, comments='#')
	
	mask = np.array([1., 1., 1., 1., 1., 1., 0., 0.]) * 1.e-3
	beam_pix_num = np.array([28., 28., 28., 28., 28., 28., 27., 28.])
	rms_pix_num = np.array([1008., 1008., 1008., 1008., 1008., 1008., 1008., 1008.])

	#dn_select = [0, 1, 2, 3, 4, 5]
	dn_select = [2, 6]
	up_select = [6, 7]
	
	freqs_dn = freq_arr[dn_select]
	freqs_up = freq_arr[up_select]
	
	spec_dn = []
	spec_up = []
	spec_dn_err = []
	spec_up_err = []
	spec_tot = []
	flux_tot = []
	freq_tot = []
	rms_tot = []
	labels = []
	mod_spec_dn_KGJP = []
	mod_spec_up_KGJP = []
	mod_spec_dn_CIJP = []
	mod_spec_up_CIJP = []
	mod_spec_dn_JP = []
	mod_spec_up_JP = []
	mod_spec_dn_KGJP_1 = []
	mod_spec_up_KGJP_1 = []
	B = []
	
	'''
	# B2 0924+30
	mask = np.array([1., 1., 1., 1., 1., 0., 0.]) * 1.e-3
	beam_pix_num = np.array([23., 23., 23., 23., 23., 27., 28.])
	rms_pix_num = np.array([1392., 1392., 1392., 1392.,1392., 1757., 1757.])
	'''

	#'''
	for region in flux_regions: # B2 0924+30
	#for region, rms in zip(flux_regions, rms_regions):
		rms = np.array([7.6, 5.9, 7.8, 2.9, 3.1, 2.6, 1.38, 0.72]) * 1.e-3 # B2 0924+30
		#if region[0] in [1, 2, 3, 4, 6, 7, 8, 9, 10, 14, 16, 17]:
		#if region[0] in [2, 4, 7, 8]:
		#if region[0] in [3, 4, 6, 9, 10]:
		#if region[0] in [10]:
		#if region[0] in [14, 15, 16, 17]:
		# B20924+30
		#if (region[0] not in [16, 23, 30, 36, 37, 41, 42, 44, 45, 50, 56, 58, 59]) and (region[0]  not in [31, 32, 33, 34, 35, 38, 39, 40, 43, 46, 47, 48, 49]):
		if (region[0] not in [3, 35, 38]):
			for i in range(len(rms)):
				rms[i] = np.sqrt(np.power(np.divide(np.multiply(region[1]/beam_pix_num[i], rms[i]), np.sqrt(rms_pix_num[i]/beam_pix_num[i])),2.0) + np.power(np.divide(rms[i], np.sqrt(region[1]/beam_pix_num[i])),2.0) + np.power(region[i+2] * mask[i] * 0.2, 2.))
			
			flux_dn = np.array(region[[x+2 for x in dn_select]]) * 1.e-3 # B2 0924+30
			flux_up = np.array(region[[x+2 for x in up_select]]) * 1.e-3 # B2 0924+30
			rms_dn = np.array(rms[dn_select])
			rms_up = np.array(rms[up_select])
			
			##flux_dn = np.array(region[[x+2 for x in dn_select]]) # 1431+1331
			##flux_up = np.array(region[[x+2 for x in up_select]]) # 1431+1331
			##rms_dn = np.array(rms[[x+2 for x in dn_select]])
			##rms_up = np.array(rms[[x+2 for x in up_select]])
			
			#flux_tot.append(np.array(region[2:] * 1.e-3)) # B2 0924+30
			flux_tot.append(np.array(region[[4, 8, 8, 9]] * 1.e-3)) # B2 0924+30

			#flux_tot.append(np.array(region[2:11]))
			#freq_tot.append(np.array(freq_arr))
			#rms_tot.append(np.array(rms))
			B.append(1.35e-6) # B2 0924+30
			#B.append(3.92e-6) # J1431.8+1331
			#B.append(4.e-6) # J1431.8+1331, after correction
			
			# B2 0924+30
			freq_tot.append(np.concatenate((freqs_dn, freqs_up)))
			#flux_tot.append(np.concatenate((flux_dn, flux_up[2:])))
			rms_tot.append(np.concatenate((rms_dn, rms_up)))
			
			##flux_tot.append(np.concatenate((flux_dn, flux_up[1:])))
			##freq_tot.append(np.concatenate((freqs_dn, freqs_up[1:])))
			##rms_tot.append(np.concatenate((rms_dn, rms_up[1:])))
			
			##B.append(region[13])
			#B.append(4.4e-6) # New derivation
			
			print 'Fr dn: ', freqs_dn, 'flux_dn: ', flux_dn
			print 'Fr up: ', freqs_up, 'flux_up: ', flux_up
			#fit_dn = np.polyfit(np.log10(freqs_dn), np.log10(flux_dn), 1)
			#fit_up = np.polyfit(np.log10(freqs_up), np.log10(flux_up), 1)
			#fit_dn, cov_dn = np.polyfit(np.log10(freqs_dn), np.log10(flux_dn), 1, w=np.log10(rms_dn), cov=True)
			#fit_up, cov_up = np.polyfit(np.log10(freqs_up), np.log10(flux_up), 1, w=np.log10(rms_up), cov=True)
			#fit_dn, cov_dn = np.polyfit(np.log10(freqs_dn), np.log10(flux_dn), 1, cov=True)
			#fit_up, cov_up = np.polyfit(np.log10(freqs_up), np.log10(flux_up), 1, cov=True)

			# When fitting for two points only, then we adopt MC approach:

			samples_dn_lo =  np.random.uniform((flux_dn[0] - rms_dn[0]), (flux_dn[0] + rms_dn[0]), size = 100)
			samples_dn_hi =  np.random.uniform((flux_dn[1] - rms_dn[1]), (flux_dn[1] + rms_dn[1]), size = 100)
			polys_dn = []

			samples_up_lo =  np.random.uniform((flux_up[0] - rms_up[0]), (flux_up[0] + rms_up[0]), size = 100)
			samples_up_hi =  np.random.uniform((flux_up[1] - rms_up[1]), (flux_up[1] + rms_up[1]), size = 100)
			polys_up = []

			for k in range(100):
				fit_dn = np.polyfit(np.log10(freqs_dn), np.log10([samples_dn_lo[k], samples_dn_hi[k]]), 1)
				polys_dn.append(fit_dn[0])

				fit_up = np.polyfit(np.log10(freqs_up), np.log10([samples_up_lo[k], samples_up_hi[k]]), 1)
				polys_up.append(fit_up[0])
			
			spec_up.append(np.mean(polys_up))
			spec_up_err.append(np.std(polys_up)) # errors

			spec_dn.append(np.mean(polys_dn))
			spec_dn_err.append(np.std(polys_dn)) # errors

			#print 'Fit_dn: ', fit_dn
			#print 'Fit_up: ', fit_up
			#print cov_dn[0][0], cov_up[0][0]
			#print -1. * cov_dn[0][0], -1 * cov_up[0][0]
			#print "Region: ", region
			
			#spec_dn_err.append(np.sqrt(cov_dn[0][0]))
			#spec_up_err.append(np.sqrt(-1. * cov_up[0][0]))
			
			#spec_dn.append(fit_dn[0])
			#spec_up.append(fit_up[0])
			
			labels.append(region[0])
			#print "Freq. total: ", freq_tot
			#print "Flux total: ", flux_tot
	#'''
	
	fig = plt.figure()
	axis = fig.add_subplot(111)
	#axis.grid()
	axis.set_aspect('equal')
	axis.tick_params(axis='x', labelsize=17)
	axis.tick_params(axis='y', labelsize=17)
	axis.tick_params(length=10)
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')

	'''
	# Color-color plot
	t_aging = np.linspace(180., 35., 100)
	t_active = np.linspace(30., 0.1, 100)
	z = 0.026141
	##z = 0.1599
	a_inj = -0.5
	#a_inj_1 = -1.
	#a_inj = -1.5 # 1441.8+1331

	delta = 1.
	B = np.array(B) * 1.e-4 # [T]
	vol = 2.4e65 # m^3
	gamma = 1.0 - 2.0 * a_inj
	
	for idx in range(len(t_aging)):	

		model_flux_KGJP = sl.get_fluxes(freq_tot[0], t_aging[idx] * 1.e6, (t_aging[idx] + t_active[idx]) / t_aging[idx], 1.e-17, gamma, B[0], vol, z, delta, 'CI_off')
		
		model_flux_JP = sl.get_fluxes(freq_tot[0], t_aging[idx] * 1.e6, 1., 1.e-17, gamma, B[0], vol, z, delta, 'JP')
		
		#t_active_ci = np.linspace(200., 30., 100.)
		#model_flux_CIJP = sl.get_fluxes(freq_tot[0], t_aging[idx] * 1.e6, (t_aging[idx] + t_active_ci[idx]) / t_aging[idx], 1.e-17, gamma, B[0], vol, z, delta, 'CI')
		
		#s_KGJP_1 = Synfit(z, t_aging[idx], t_active[idx], B[0], freq_tot[0], a_inj_1, 1.e-17, 'CI_off', 'JP')
		#model_flux_KGJP_1 = np.array(s_KGJP_1())
		
		print "FREQ TOT: ", freq_tot[0] 
		print "KGJP MODEL: ", model_flux_KGJP
		#lower = dn_select
		#upper = up_select

		lower = [0, 1]
		upper = [2, 3]

		#upper = np.add(range(len(dn_select)), len(up_select) - 2)

		print "LOWER IDXs: ", lower
		print "UPPER IDXs: ", upper 
		
		mod_fit_dn_KGJP = np.polyfit(np.log10(freq_tot[0][lower]), np.log10(model_flux_KGJP[lower]), 1)
		mod_fit_up_KGJP = np.polyfit(np.log10(freq_tot[0][upper]), np.log10(model_flux_KGJP[upper]), 1)
		
		mod_fit_dn_JP = np.polyfit(np.log10(freq_tot[0][lower]), np.log10(model_flux_JP[lower]), 1)
		mod_fit_up_JP = np.polyfit(np.log10(freq_tot[0][upper]), np.log10(model_flux_JP[upper]), 1)
		
		#mod_fit_dn_CIJP = np.polyfit(np.log10(freq_tot[0][lower]), np.log10(model_flux_CIJP[lower]), 1)
		#mod_fit_up_CIJP = np.polyfit(np.log10(freq_tot[0][upper]), np.log10(model_flux_CIJP[upper]), 1)
		
		#mod_fit_dn_KGJP_1 = np.polyfit(np.log10(freq_tot[0][lower]), np.log10(model_flux_KGJP_1[lower]), 1)
		#mod_fit_up_KGJP_1 = np.polyfit(np.log10(freq_tot[0][upper]), np.log10(model_flux_KGJP_1[upper]), 1)
		
		mod_spec_dn_KGJP.append(mod_fit_dn_KGJP[0])
		mod_spec_up_KGJP.append(mod_fit_up_KGJP[0])
		
		mod_spec_dn_JP.append(mod_fit_dn_JP[0])
		mod_spec_up_JP.append(mod_fit_up_JP[0])
		
		#mod_spec_dn_CIJP.append(mod_fit_dn_CIJP[0])
		#mod_spec_up_CIJP.append(mod_fit_up_CIJP[0])
		
		#mod_spec_dn_KGJP_1.append(mod_fit_dn_KGJP_1[0])
		#mod_spec_up_KGJP_1.append(mod_fit_up_KGJP_1[0])
	
	#print 'Mod specs: ', mod_spec_dn, mod_spec_up
	#print "spec_dn: ", spec_dn
	#print "spec_up: ", spec_up
	#print "spec_up_err: ", spec_up_err
	#print "spec_dn_err: ", spec_dn_err
	dia = np.linspace(-2., -0.5, 10)
	axis.set_aspect('equal')
	data = axis.plot(dia, dia, '-.r', label='power law', linewidth=3)
	
	data = axis.errorbar(spec_dn, spec_up, spec_up_err, spec_dn_err, marker=None, fmt=None, ecolor='gray', barsabove=False, zorder=-100)
	data = axis.plot(mod_spec_dn_KGJP, mod_spec_up_KGJP, '--g', label='KGJP', linewidth=3)
	data = axis.plot(mod_spec_dn_JP, mod_spec_up_JP, '--k', label='JP', linewidth=3)
	#data = axis.plot(mod_spec_dn_CIJP, mod_spec_up_CIJP, '-.b', label='CIJP')
	#data = axis.plot(mod_spec_dn_KGJP_1, mod_spec_up_KGJP_1, '-.k', label='KGJP1')
	data = axis.scatter(spec_dn, spec_up, marker='o', c=np.abs(spec_up), s = np.abs(spec_dn) * 100, zorder=100)
	for label, x, y in zip(labels, spec_dn, spec_up):
		#print 'Spix dn: ', x, ' Spix up: ', y
		plt.annotate(label, xy=(x, y), xytext=(-5, 5), textcoords='offset points', ha='right', va='bottom')
		
	legend = axis.legend(loc='upper left', shadow=True, labelspacing=0.01, ncol=1)
	
	xlabel(r'$\alpha_{\mathrm{low}}$', fontsize=25, fontweight='bold', color='#000000')
	ylabel(r'$\alpha_{\mathrm{high}}$', fontsize=25, fontweight='bold', color='#000000')
	#plt.xlim(-3., -0.5)
	#plt.ylim(-3., -0.5)
	cb = plt.colorbar(data)
	cb.set_label(r'-$\alpha_{\mathrm{high}}$', fontweight='bold', fontsize=25)
	plt.show()
	'''
	
	'''
	t_aging = np.linspace(1., 4., 10)
	t_active = np.linspace(1., 50., 10)
	mod_freq = np.linspace(1.e8, 1.5e9, 10)
	#for idx in [1, 5, 9]:	
	#	s = Synfit(0.1599, t_aging[idx], t_active[idx], 8.e-9, mod_freq, -0.5, 1.e-16, 'delta', 'JP')
	#	model_flux = s()
	
	br_arr = []
	br_flux_arr = []
	#t_mod_arr = []
	#sf_mod_arr = []

	inj_idx = -0.85

	delta = 1.
	B = np.array(B) * 1.e-4 # [T]
	vol = 2.4e65 # m^3
	gamma = 1.0 - 2.0 * inj_idx
	z = 0.026141
	
	for idx in range(len(flux_tot)):
		
		print "All frequencies: ", freq_tot
		print "All fluxes: ", flux_tot
		print "Frequency from all frequencies for [", idx, "]: ", freq_tot[idx]
		print "Flux from all fluxes for [", idx, "]: ", flux_tot[idx]
		print "All RMSs: :", rms_tot[idx]
		print "All Bs: ", B
		print 'Spix: ', np.polyfit(np.log10(freq_tot[idx]), np.log10(flux_tot[idx]), 1)[0]
		polynom = np.poly1d(np.polyfit(np.log10(freq_tot[idx]), np.log10(flux_tot[idx]), 1))
		#axis.loglog(np.linspace(1.e8, 1.5e9, 100), 10**polynom(np.log10(np.linspace(1.e8, 1.5e9, 100))))
		data = axis.plot(freq_tot[idx], flux_tot[idx], marker='o', linestyle='none')
		axis.errorbar(freq_tot[idx], flux_tot[idx], rms_tot[idx], marker='o', markersize=4, fmt='-^', linestyle='none')
		
		#fitter = fit(freq_tot[idx], flux_tot[idx], rms_tot[idx], 'delta', 'JP', constant_params=[z, B[idx], inj_idx], fitted_params=[20., 1., 1.e-15])
		#print "Fitter params: ", 10.**fitter.params[0], 10.**fitter.params[1], 10.**fitter.params[2]
		
		#s = Synfit(z, 10.**fitter.params[0], 10.**fitter.params[1], B[idx], mod_freq, inj_idx, 10.**fitter.params[2], 'delta', 'JP')
		#mod_flux = s()

		# Leith fit
		fitter_JP = fit_leith(freq_tot[idx], flux_tot[idx], rms_tot[idx], constant_params = [z, B[idx], vol, delta, 'JP', inj_idx], fitted_params = [60., 1.e6])

		mod_flux = sl.get_fluxes(mod_freq, (10.**fitter_JP.params[0]) * 1.e6, 1., 10.**fitter_JP.params[1], gamma, B[0], vol, z, delta, 'JP')

		#br = break_estimate(B[idx] * 1.e6 * 1.e4, z, 10.**fitter_JP.params[0]) # this holds for the KGJP LF break
		
		# Kardashev 1962, for delta JP breaks
		br = (3.4e8 * (B[idx] * 1.e4)**-3. * ((10.**fitter_JP.params[0]) * 1.e6)**-2.) / 1.e9
		
		br_arr.append(br)
		#t_mod_arr.append(fitter.params[0])
		#sf_mod_arr.append(fitter.params[1])
		print 'nu_b: ', br, ' GHz'
		
		#s = Synfit(z, 10.**fitter.params[0], 10.**fitter.params[1], B[idx], [br * 1.e9], inj_idx, 10.**fitter.params[2], 'delta', 'JP')
		#mod_br_flux = s()

		model_flux_JP = sl.get_fluxes([br * 1.e9], 10.**fitter_JP.params[0] * 1.e6, 1., fitter_JP.params[1], gamma, B[0], vol, z, delta, 'JP')

		print "Model flux JP: ", model_flux_JP

		br_flux_arr.append(model_flux_JP[0])
		
		print "Mod freq: ", mod_freq
		print "Mod flux: ", mod_flux

		axis.loglog(mod_freq, mod_flux, linewidth=1)
		
		#for idx in range(len(flux_tot)):
				
		#plt.axvline(60.e6, color='y', linewidth=3, linestyle='-')
		#plt.axvline(150.e6, color='y', linewidth=3, linestyle='-')
		#plt.axvline(1.4e9, color='k', linewidth=3, linestyle='-')
		#plt.axvline(4.5e9, color='k', linewidth=3, linestyle='-')
		#plt.xlim(1.e7, 8.5e9)
		#xlabel(r'Frequency [arbitrary units]', fontsize=18, fontweight='bold', color='#000000')
		#ylabel(r'Flux density [arbitrary units]', fontsize=18, fontweight='bold', color='#000000')
	plt.show()
	print br_arr
	print br_flux_arr
	#plt.savefig(path+'spectral_fits.eps', bbox_inches='tight')
	'''
	
	#'''
	# Shifts
	# from the fitting above
	
	# for regions 2,4,7,8 for J1431.8+1331
	
	#br_arr = np.log10(np.array([0.47749889192881378, 1.4494051831119332, 0.92435578915154659, 0.35128149269692044]))
	#br_flux_arr = np.log10(np.array([0.0029655677023559166, 0.0013251908740005949, 0.0018054211913134141, 0.0071107471571854501]))
	#ref_idx = 1 # the best fitted region
	
	# for regions 3, 6, 9, 10 for J1431.8+1331
	#br_arr = np.log10(np.array([0.90318766053535315, 0.53103672798329626, 0.42589046048571927, 0.69411398356633203]))
	#br_flux_arr = np.log10(np.array([0.0038681828287542921, 0.013079676482439016, 0.015717054334086388, 0.0059977318581678173]))
	#ref_idx = 2 # the best fitted region
	
	#br_arr = np.log10(np.array([0.91804720945964358, 1.0014921067757998, 0.54219257375498275, 0.72358779404233098]))
	#br_flux_arr = np.log10(np.array([0.0037300962648417137, 0.0035057498594740855, 0.0093932258849480435, 0.0054686890371507179]))
	#ref_idx = 2 # the best fitted region
	
	# for regions 3, 6, 9, 10 for J1431.8+1331
	#br_arr = np.log10([0.91804723856730086, 1.0025922706563306, 0.54219257642326524, 0.72358780265855027])
	#br_flux_arr = np.log10([0.0037300771044417882, 0.0034965604820532048, 0.0093932260339157653, 0.0054686482938245198])
	ref_idx = 36 # the best fitted region
	# for regions 14, 15, 16, 17 for J1431.8+1331
	#br_arr = np.log10(np.array([0.48037642879535081, 0.40748014826445694, 0.42174532222805894, 0.45770770787529741]))
	#br_flux_arr = np.log10(np.array([0.0027123715127149398, 0.0028667549832069161, 0.006865639432732407, 0.0039242038394277454]))
	#ref_idx = 1 # the best fitted region
	
	#br_arr = np.log10(np.array([0.50184952047242426, 0.40732166713575968, 0.4251551541056951, 0.47888842715176788]))
	#br_flux_arr = np.log10(np.array([0.0025007382540974702, 0.0028818184857804831, 0.0068111234668502682, 0.0036134374556140427]))
	#ref_idx = 1 # the best fitted region
	
	# For B2 0924+30, all regions except the NVSS negative flux value ones
	# Kardashev breeaks
	#br_arr = np.log10([280.29057537059111, 1381.9031651679115, 700.46609615879424, 51.084002590170478, 588.98077022446489, 1381.9031651679115, 1381.9031651679115, 133.02540229209492, 80.320763735358582, 142.86038388355493, 297.73162846835515, 479.56538321670467, 142.91190184566918, 87.461969167623337, 128.18883269146477, 57.898446302603574, 283.77692605996867, 156.42591869721784, 137.22208314551952, 56.171093040118706, 58.894988730125043, 43.38668805020307, 57.898448480006948, 61.592778978017684, 127.94357183583851, 154.66619004194675, 53.71367104353736, 24.259065755420693, 31.861442860419061, 20.600553746274883, 22.286722336778613, 23.297190485439121, 14.147633120306946, 19.163902784460284, 10.671682783236783, 17.529218835504192, 26.80916289917274, 19.798148289142347, 30.837066380444824, 28.644085367737485, 41.097014134843079, 38.755105282102342, 32.719581715913918, 60.34026650612163, 46.014029633758021, 105.72952571077251, 78.864758148184222, 111.91098246375338, 67.746586934060943, 382.83798692058184, 1381.9031651679115, 1054.3420651209813, 78.410177446372344, 58.894982722215644, 280.29054282717641, 1381.9031651679115, 1381.9031651679115, 132.60924875798077, 119.17212442787252, 789.13951744906535, 605.59038685092321, 154.24818710493628, 33.809575427160311])
	
	#br_flux_arr = np.log10([1.0486478508837741e-15, 1.171257645346567e-62, 2.8566557665950693e-33, 2.1130830807198984e-18, 8.4421726929269535e-29, 1.8057590553768544e-62, 2.2003956327067934e-62, 1.4444455507615897e-09, 6.6075337021281759e-27, 1.2590353860909378e-10, 7.5100352453310846e-16, 2.2442041555215511e-23, 7.7522278034920103e-10, 1.6624902827267185e-08, 8.361775140645159e-10, 4.413126198872614e-20, 4.5202427068403476e-15, 3.210318164485912e-10, 1.0716289487537085e-09, 1.7291299968197848e-19, 1.5438086606435192e-20, 4.5742167165196007e-16, 6.4762692997569867e-20, 6.8663596315970063e-21, 1.8744956555983298e-09, 1.6762502038445431e-10, 6.8290427331555461e-19, 5.7334628196623995e-12, 2.5815484781169859e-11, 1.0143048843605147e-17, 5.9008313808790673e-19, 8.0296224493671212e-20, 4.9938242048837586e-12, 6.0602730326364689e-17, 7.843946204294837e-12, 2.1095459848726137e-11, 1.1598737957398362e-11, 2.9864952008046305e-17, 1.9875396881216075e-11, 1.0384744735172107e-11, 9.6291608599867748e-19, 1.0215800278569759e-17, 1.5427546904851493e-15, 7.9217553503757784e-21, 1.0578394335601557e-16, 8.3078653074288392e-17, 2.8946512279694966e-26, 3.9230880767753191e-18, 5.1309539659798796e-23, 2.0305471350486999e-19, 1.9683928740657209e-62, 2.1657658512722575e-48, 2.1621371434630037e-26, 9.063862382240224e-21, 3.6222758143338702e-15, 3.2405476457424709e-62, 1.7628005752861549e-62, 7.3073445745478266e-10, 1.8772226662171475e-09, 5.8913037029605443e-37, 6.2253832440729137e-29, 1.282698575656685e-10, 2.8901636445289703e-16])
	
	# Murgia breaks
	#br_arr = np.log10([12.462049636953768, 55.454425616318083, 29.64103176009074, 2.3877417082011916, 25.200761545862701, 55.454425616318083, 55.454425616318083, 6.0762992506069322, 3.7186013816902639, 6.51130289701024, 13.20260622308308, 20.76701769696885, 6.5135778224131515, 4.0409350555164112, 5.8618468082219595, 2.6996740170812634, 12.6103172671157, 7.1090274684148556, 6.2620939347062361, 2.6207113626912273, 2.7451967485820123, 2.0339707169184349, 2.699674116572202, 2.8683169882697892, 5.8509624935913722, 7.0316355799338908, 2.50825060369618, 1.1473218869053408, 1.5011346497377445, 0.97629705630035268, 1.0551860468104455, 1.1024076918432175, 0.67326086425977161, 0.90898989881237269, 0.50918696639815564, 0.83229743512230914, 1.2662300166279812, 0.93871499632285516, 1.4535763372755568, 1.3516455778257357, 1.9284261232296698, 1.8203188493449425, 1.5409486926145872, 2.8111765556937574, 2.1549015177565494, 4.8611717271868615, 3.6527585969398784, 5.1374065466519179, 3.1485416311299357, 16.77660092429177, 55.454425616318083, 43.300064708582937, 3.632193069501588, 2.745196474207297, 12.462048252391126, 55.454425616318083, 55.454425616318083, 6.0578610242444135, 5.4610832778734713, 33.122414964381711, 25.86702565873037, 7.0132457670681729, 1.5914856569406082])
	
	#br_flux_arr = np.log10([0.00010890944118511169, 7.7458674957371111e-06, 6.8031306845287094e-05, 0.00055220780632282555, 4.6336770233179196e-05, 1.1942009879506757e-05, 1.455185635445927e-05, 0.00072277960681501349, 0.00056496035117027763, 0.00012914986552998772, 0.0003976802610891609, 0.00035568647398086109, 0.00079825245771986955, 0.00082128730068266504, 0.00029876820159010836, 0.0010531980066683219, 0.00064981423895758264, 0.00093807113211812759, 0.00072481815876506914, 0.0013088065081745916, 0.00071552425190091293, 0.00079526832179599937, 0.0015455718906324247, 0.001927421210959271, 0.00065863583325440902, 0.00042626108158763768, 0.0010143949784875634, 0.0014598390382572059, 0.0024068081159775889, 0.0045834848027860413, 0.0040075432023387184, 0.0027879692276960837, 0.0036677402982795517, 0.0027636921144748709, 0.0069848295877531887, 0.010517768732537909, 0.0029508689216336047, 0.0037411117147840124, 0.0017383246078642115, 0.0011412935496666161, 0.001395919283864375, 0.0020407741367832978, 0.0024581855966607402, 0.00096291808208869912, 0.0010061825045448718, 0.00074635519783521943, 0.00092157389297209359, 0.00020251173183064567, 0.00089388315004843542, 0.00032654166116543367, 1.3017554628260088e-05, 2.9436727700127777e-05, 0.00050593245399556213, 0.00042009010270382323, 0.00037619763177609578, 2.143068416865433e-05, 1.1657912955211027e-05, 0.00035503133871353195, 0.00037122464710762012, 6.9281049275958859e-05, 0.00016747339496679632, 0.00031560526507299843, 0.00092166959690425645])
	
	#br_arr = np.log10([67.099854995744877, 67.099854995744877, 67.099854995744877, 4.584327792612191, 67.099854995744877, 67.099854995744877, 67.099854995744877, 30.308571862281628, 10.000515000679236, 43.181659829789375, 67.099854995744877, 67.099854995744877, 37.016773272015236, 11.650042270552438, 29.169647432302007, 5.3190487565473559, 67.099854995744877, 47.882647205987176, 34.483490552308702, 5.319048744217385, 6.242806751887005, 3.7582176179915816, 5.8316319300038773, 6.2428068662083502, 24.452306409645644, 58.143044457715298, 4.8138436212576012, 1.7597423884472552, 2.4556813090045044, 1.3591776131768318, 1.5666206071828779, 1.6183083524215254, 0.89847521391969432, 1.1779224106682056, 0.64018667847888333, 1.1330007718393638, 1.910506192459752, 1.31633106473467, 2.2404078260933469, 1.967982800599881, 3.2933561522572732, 3.1005295794484966, 2.5249757103474728, 6.0176135779689925, 3.8553503924321855, 18.151667868106649, 9.6484163222889574, 24.51908470496295, 6.9811186993999987, 67.099854995744877, 67.099854995744877, 67.099854995744877, 8.9849663444803838, 5.6508416004099491, 67.099854995744877, 67.099854995744877, 67.099854995744877, 34.38730882223026, 23.129948757258866, 67.099854995744877, 67.099854995744877, 54.800302560754261, 2.6919295186251908])
	
	#br_flux_arr = np.log10([6.9519056568384994e-07, 1.7618921533531046e-06, 1.6798846167504516e-06, 0.00018124570159087423, 8.0244675906586627e-07, 2.7211320061415442e-06, 3.3170313661628402e-06, 6.0658995609406598e-05, 0.00014074132171129912, 3.9656555502329052e-06, 2.6995577785802577e-06, 4.3678141685392666e-06, 4.1219817836250571e-05, 0.0001931774878879901, 2.6593324513943865e-05, 0.00033480593383360856, 4.1993886027436982e-06, 2.0595713562081346e-05, 4.4628032780590342e-05, 0.00040387664436036287, 0.00019610528225485444, 0.00027208741496183547, 0.00044819469627996805, 0.00055670241149000316, 8.394907074795901e-05, 3.7429724416598385e-06, 0.00033180570910123201, 0.00063823793578230553, 0.00095933231264509376, 0.0022857124122917551, 0.0018343255675818048, 0.0012900984073693229, 0.0019822079249292761, 0.0015246772994699906, 0.00417729105767135, 0.005461896300496618, 0.0013094794373006771, 0.0018534669191726165, 0.00074476290503372857, 0.00052628564845801952, 0.00052305013267791287, 0.00076882737298185984, 0.00097667902813900766, 0.00028172485191246778, 0.00035646428440200929, 0.00012820769158337, 0.00023266974089423381, 2.2894356728034791e-05, 0.00025532534398307676, 2.9493198804374883e-06, 2.9653419709832746e-06, 2.2902823643269754e-06, 0.00013484850861934935, 0.00012779527387431433, 2.4019771849484646e-06, 4.8855585856149403e-06, 2.6597365411126922e-06, 2.1429477749545654e-05, 4.9173473846974589e-05, 2.2727073428719815e-06, 3.0586478164071163e-06, 3.7279216920422604e-06, 0.00035200373889186678])
	
	# Core regions
	#br_arr = np.log10([1.1473218869053408, 1.5011346497377445, 0.97629705630035268, 1.0551860468104455, 1.1024076918432175, 0.67326086425977161, 0.90898989881237269, 0.50918696639815564, 0.83229743512230914, 1.2662300166279812, 0.93871499632285516, 1.4535763372755568, 1.3516455778257357])
	#br_flux_arr = np.log10([0.0014598390382572059, 0.0024068081159775889, 0.0045834848027860413, 0.0040075432023387184, 0.0027879692276960837, 0.0036677402982795517, 0.0027636921144748709, 0.0069848295877531887, 0.010517768732537909, 0.0029508689216336047, 0.0037411117147840124, 0.0017383246078642115, 0.0011412935496666161])
	
	# Rest
	
	#br_arr = np.log10([12.462049636953768, 55.454425616318083, 29.64103176009074, 2.3877417082011916, 25.200761545862701, 55.454425616318083, 55.454425616318083, 6.0762992506069322, 3.7186013816902639, 6.51130289701024, 13.20260622308308, 20.76701769696885, 6.5135778224131515, 4.0409350555164112, 5.8618468082219595, 2.6996740170812634, 12.6103172671157, 7.1090274684148556, 6.2620939347062361, 2.6207113626912273, 2.7451967485820123, 2.0339707169184349, 2.699674116572202, 2.8683169882697892, 5.8509624935913722, 7.0316355799338908, 2.50825060369618, 1.9284261232296698, 1.8203188493449425, 1.5409486926145872, 2.8111765556937574, 2.1549015177565494, 4.8611717271868615, 3.6527585969398784, 5.1374065466519179, 3.1485416311299357, 16.77660092429177, 55.454425616318083, 43.300064708582937, 3.632193069501588, 2.745196474207297, 12.462048252391126, 55.454425616318083, 55.454425616318083, 6.0578610242444135, 5.4610832778734713, 33.122414964381711, 25.86702565873037, 7.0132457670681729, 1.5914856569406082])
	#br_flux_arr = np.log10([0.00010890944118511169, 7.7458674957371111e-06, 6.8031306845287094e-05, 0.00055220780632282555, 4.6336770233179196e-05, 1.1942009879506757e-05, 1.455185635445927e-05, 0.00072277960681501349, 0.00056496035117027763, 0.00012914986552998772, 0.0003976802610891609, 0.00035568647398086109, 0.00079825245771986955, 0.00082128730068266504, 0.00029876820159010836, 0.0010531980066683219, 0.00064981423895758264, 0.00093807113211812759, 0.00072481815876506914, 0.0013088065081745916, 0.00071552425190091293, 0.00079526832179599937, 0.0015455718906324247, 0.001927421210959271, 0.00065863583325440902, 0.00042626108158763768, 0.0010143949784875634, 0.001395919283864375, 0.0020407741367832978, 0.0024581855966607402, 0.00096291808208869912, 0.0010061825045448718, 0.00074635519783521943, 0.00092157389297209359, 0.00020251173183064567, 0.00089388315004843542, 0.00032654166116543367, 1.3017554628260088e-05, 2.9436727700127777e-05, 0.00050593245399556213, 0.00042009010270382323, 0.00037619763177609578, 2.143068416865433e-05, 1.1657912955211027e-05, 0.00035503133871353195, 0.00037122464710762012, 6.9281049275958859e-05, 0.00016747339496679632, 0.00031560526507299843, 0.00092166959690425645])

	# Leith Code, new regions

	#br_arr = np.log10(np.array([51.794634584133782, 35.644685829925336, 36.055238475476479, 30.774752026374582, 28.269864002291396, 29.568999028528093, 26.116466618836789, 23.710009731074226, 23.178314533746818, 24.547048909848151, 26.492243521080734, 24.904895863108546, 20.919038847348634, 15.290806555111102, 17.938116025790126, 21.562263366017291, 18.039982654399466, 14.707448997890864, 13.418044641500703, 10.423838340657516, 11.551482007535837, 17.108733856367479, 22.039511466334812, 14.811931292838459, 29.768160459701644, 31.123568744817142, 24.230509527865841, 25.170990974263411, 30.178339427515603, 32.026007266763777, 26.566675722957243, 21.708815130737896, 30.260588626802832, 33.569980690217641, 32.377985720857772, 21.618672190016053, 22.217897018622768, 33.822156471576903, 22.717373556615012]) * 1.e9)

	#br_flux_arr = np.log10(np.array([7.5631104295336782e-08, 9.4639683816332682e-08, 9.3991627688036971e-08, 1.0336009591545986e-07, 1.0876147642668296e-07, 1.0586868544782902e-07, 1.1405658603483254e-07, 1.2086751407541002e-07, 1.2252350826777081e-07, 1.1837750950909376e-07, 1.1308313313660933e-07, 1.1735402568891064e-07, 1.3029959282990912e-07, 1.5725641228560487e-07, 1.4288982134854015e-07, 1.2795333770434925e-07, 1.4240516719110451e-07, 1.6096964515488347e-07, 1.7007965433366977e-07, 1.9790092449273249e-07, 1.8607259944518204e-07, 1.4700646699635152e-07, 1.2628365502526338e-07, 1.6028741357941069e-07, 1.0544313758692722e-07, 1.0266349581065454e-07, 1.1930295219397807e-07, 1.166080907613415e-07, 1.0458089789638406e-07, 1.0091787696715027e-07, 1.1289293343779741e-07, 1.2743437228514284e-07, 1.0441025502875519e-07, 9.8106841740331689e-08, 1.0025820545084794e-07, 1.2775291794684348e-07, 1.2567433220376167e-07, 9.7667303923947031e-08, 1.2400910659106938e-07]))

	br_arr = np.log10(np.array([60.049274191399597, 46.500921263844589, 29.582203002533582, 26.747511891776558, 21.930551578701934, 20.171844337249844, 16.757148613273092, 16.811064154079755, 18.6521109409356, 17.835501512324292, 19.010399721747245, 18.607883908249708, 15.028937824574744, 11.98045811009835, 12.447076801762586, 19.046169814243139, 10.97546014790807, 9.5655227200477011, 9.0569434803844544, 7.805291740805357, 7.771378841156527, 11.259890282176119, 17.375511555595644, 10.120910770311806, 25.670308489344229, 24.991173521782038, 17.871554370409648, 19.632932819703143, 21.594354769332778, 23.141877096963544, 19.482691003651038, 15.611163509426836, 23.417673434320402, 32.930285890572833, 28.106818106848948, 21.491268522996997, 33.900883120850175, 20.753111811648438]))
	
	br_flux_arr = np.log10(np.array([1.1623862095515098e-13, 1.4392305061399786e-13, 2.2843891960033088e-13, 2.3933692539391262e-13, 2.7430706064060627e-13, 3.2424475648403396e-13, 3.7927240455981917e-13, 3.6083619769809179e-13, 3.1872815527670149e-13, 3.4505307945395647e-13, 3.3824918333459827e-13, 3.3375703539234616e-13, 3.937944150224838e-13, 4.6829869574670241e-13, 4.7412548123111584e-13, 3.2206005886102552e-13, 5.2417852758772029e-13, 5.5394070782598411e-13, 6.0220777864767368e-13, 6.9743835925238983e-13, 6.9703896743784006e-13, 5.128878215461631e-13, 3.4900028346161932e-13, 5.5325990969870142e-13, 2.5545820689611866e-13, 2.6156915476333933e-13, 3.4190846416253319e-13, 3.0825994932723964e-13, 3.0334037800183962e-13, 2.803594864644773e-13, 3.1856065482737747e-13, 3.7589878686903492e-13, 2.7515006759291819e-13, 1.962586747172867e-13, 2.2227249644746574e-13, 2.6571804025574442e-13, 1.8988372142507461e-13, 2.8096539479937174e-13]))
	
	#ref_idx = 30
	
	fr_shift = []
	flux_shift = []
	
	for idx in range(len(br_arr)):
		fr_shift.append(br_arr[idx] - br_arr[ref_idx])
		flux_shift.append(br_flux_arr[idx] - br_flux_arr[ref_idx])
		
	fr_shift = np.array(fr_shift)
	flux_shift = np.array(flux_shift)
	
	print "Freq. shift", fr_shift
	print "Flux shift", flux_shift
	
	axis.plot(fr_shift, flux_shift, marker='o', linestyle='none')
	
	polfit, covar = np.polyfit(fr_shift, flux_shift, 1, cov=True)
	print "Fit slope: ", polfit[0]
	print "Fit error: ", np.sqrt(1. * covar[0][0])
	polynom = np.poly1d(polfit)
	
	
	axis.plot(linspace(-1., 1.5, 10), polynom(linspace(-1., 1.5, 10)), '--r')
	#plt.xlim(-0.7, 0.3)
	#plt.ylim(-0.2, 0.5)
	xlabel(r'$\Delta$ log($\nu$)', fontsize=18, fontweight='bold', color='#000000')
	ylabel(r'$\Delta$ log(S)', fontsize=18, fontweight='bold', color='#000000')
	

	#print polynom
	#print 10**fr_shift
	
	##title('', fontsize=20, fontweight='bold')
	##plt.savefig('/Users/users/shulevski/Desktop/shift_fit.eps', bbox_inches='tight')
	plt.show()
	#'''
	
	'''
	# Model fitting to the shifted points
	tot_shifted_flux = []
	tot_shifted_freq = []
	tot_shifted_rms = []
	inj_idx = -0.85

	delta = 1.
	B = np.array(B) * 1.e-4 # [T]
	vol = 2.4e65 # m^3
	gamma = 1.0 - 2.0 * inj_idx
	
	for idx in range(len(flux_tot)):
		fl = flux_tot[idx]
		fr = freq_tot[idx]
		rms = rms_tot[idx]
		#for x in range(len(fl)):
			#print fl - flux_shift[idx]
			#print np.log10(fl[x]) - flux_shift[idx]
		tot_shifted_flux.append(np.log10(fl) - flux_shift[idx])
		tot_shifted_freq.append(np.log10(fr) - fr_shift[idx])
		tot_shifted_rms.append(rms)
	#print 10**np.array(tot_shifted_flux)
	#print 10**np.array(tot_shifted_freq)
	#print np.array(tot_shifted_rms)
	
	print "Flux tot arr: ", flux_tot
	print "Freq. tot arr: ", freq_tot

	print "Size of shift arrays", len(tot_shifted_freq), len(tot_shifted_flux)

	print "Tot. shuifted freq.", tot_shifted_freq
	print "Tot. shifted flux", tot_shifted_flux
	print "Tot. shifted rms", tot_shifted_rms

	dim = len(tot_shifted_flux)
	
	for idx in range(dim):
		color = np.random.rand(dim,1)
		axis.scatter(10.**np.array(tot_shifted_freq[idx]), 10.**np.array(tot_shifted_flux[idx]), s=50, c=color, zorder=2)
		axis.errorbar(10.**np.array(tot_shifted_freq[idx]), 10.**np.array(tot_shifted_flux[idx]), tot_shifted_rms[idx], marker=None, fmt=None, ecolor='gray', zorder=1)
		axis.loglog(10.**np.array(tot_shifted_freq[idx]), 10.**np.array(tot_shifted_flux[idx]), marker=None, linestyle='None')

	
	#plt.xlim(1.e7, 1.e10)
	#plt.ylim(1.e-6, 4.)
	
	#fitter_JP = fit(10**np.array(tot_shifted_freq), 10**np.array(tot_shifted_flux), np.array(tot_shifted_rms), 'delta', 'JP', constant_params = [z, B[0], inj_idx], fitted_params = [20., 1., 1.e-15])
	
	#fitter_CIJP = fit(10**np.array(tot_shifted_freq), 10**np.array(tot_shifted_flux), np.array(tot_shifted_rms), 'CI', 'JP', constant_params = [0.1599, 4.e-6, -1.], fitted_params = [40., 10., 1.e-22])
	
	#fitter_KGJP = fit(10**np.array(tot_shifted_freq), 10**np.array(tot_shifted_flux), np.array(tot_shifted_rms), 'CI_off', 'JP', constant_params = [z, B[0], inj_idx], fitted_params = [20., 1., 1.e-28])
	
	#print fitter.params[0], fitter.params[1], fitter.params[2]
	
	#s_jp = Synfit(z, 10**fitter_JP.params[0], 0., B[0], np.linspace(2.e7, 3.e9, 50) , inj_idx, 10**fitter_JP.params[2], 'delta', 'JP')
	#mod_tot_flux_JP = s_jp()
	
	#s_CIJP = Synfit(z, 10**fitter_CIJP.params[0], 10**fitter_CIJP.params[1], 4.e-6, np.linspace(1.e8, 2.e9, 100) , -1., 10**fitter_CIJP.params[2], 'CI', 'JP')
	#mod_tot_flux_CIJP = s_CIJP()
	
	#s_kgjp = Synfit(z, 10**fitter_KGJP.params[0], 10**fitter_KGJP.params[1], B[0], np.linspace(2.e7, 3.e9, 50) , inj_idx, 10**fitter_KGJP.params[2], 'CI_off', 'JP')
	#mod_tot_flux_KGJP = s_kgjp()
	

	# Leith fit
	#tot_fitter_JP = fit_leith(10**np.array(tot_shifted_freq), 10**np.array(tot_shifted_flux), np.array(tot_shifted_rms), constant_params = [z, B[0], vol, delta, 'JP', inj_idx], fitted_params = [30., 1.e3])

	#mod_tot_flux_JP = sl.get_fluxes(np.linspace(5.e7, 4.e10, 50), (10.**tot_fitter_JP.params[0]) * 1.e6, 1., 10.**tot_fitter_JP.params[1], gamma, B[0], vol, z, delta, 'JP')

	#tot_fitter_CI = fit_leith(10**np.array(tot_shifted_freq), 10**np.array(tot_shifted_flux), np.array(tot_shifted_rms), constant_params = [z, B[0], vol, delta, 'CI', inj_idx], fitted_params = [20., 2.e8])

	#mod_tot_flux_CI = sl.get_fluxes(np.linspace(2.e7, 3.e9, 50), (10.**tot_fitter_CI.params[0]) * 1.e6, 1., 1.e10, gamma, B[0], vol, z, delta, 'CI')

	tot_shifted_flux = []
	tot_shifted_freq = []
	tot_shifted_rms = []
	for idx in range(len(flux_tot)):
		fl = flux_tot[idx]
		fr = freq_tot[idx]
		rms = rms_tot[idx]
		for x in range(len(fl)):
			#print fl - flux_shift[idx]
			#print np.log10(fl[x]) - flux_shift[idx]
			tot_shifted_flux.append(np.log10(fl[x]) - flux_shift[idx])
			tot_shifted_freq.append(np.log10(fr[x]) - fr_shift[idx])
			tot_shifted_rms.append(rms[x])

	tot_fitter_CI_off = fit_leith(10.**np.array(tot_shifted_freq), 10.**np.array(tot_shifted_flux), np.array(tot_shifted_rms), constant_params = [z, B[0], vol, delta, 'CI_off', inj_idx], fitted_params = [30., 5., 1.e-6])

	mod_tot_flux_CI_off = sl.get_fluxes(np.linspace(5.e7, 1.e10, 50), (10.**tot_fitter_CI_off.params[0]) * 1.e6, (10.**tot_fitter_CI_off.params[0] + 10.**tot_fitter_CI_off.params[1]) / (10.**tot_fitter_CI_off.params[0]), 10.**tot_fitter_CI_off.params[2], gamma, B[0], vol, z, delta, 'CI_off')


	#print 'JP model fluxes: ', mod_tot_flux_JP
	#print 'CI_off (KGJP) model fluxes: ', mod_tot_flux_KGJP
	#print 'CIJP model fluxes: ', mod_tot_flux_CIJP

	print B
	print vol
	print delta
	print gamma
	#print 10.**tot_fitter_JP.params[0]
	
	#print mod_tot_flux_JP
	#print mod_tot_flux_CI_off
	#axis.loglog(np.linspace(5.e7, 1.e10, 50), mod_tot_flux_JP, '-c', linewidth=2, zorder=3)
	#axis.loglog(np.linspace(2.e7, 3.e9, 50), mod_tot_flux_CI, '-g')
	axis.loglog(np.linspace(5.e7, 1.e10, 50), mod_tot_flux_CI_off, '-c')
	
	#plt.ylim(3.e-5, 3.e-2)
	#plt.xlim(4.e7, 5.e10)
	
	xlabel(r'Frequency [Hz]', fontsize=18, fontweight='bold', color='#000000')
	ylabel(r'Flux density [Jy]', fontsize=18, fontweight='bold', color='#000000')
	
	
	title('', fontsize=20, fontweight='bold')
	#plt.savefig('/Users/users/shulevski/Desktop/shift_model_fit.eps', bbox_inches='tight')
	plt.show()
	'''
	
def spec_index():
	from astropy.io import fits
	#import pyfits
	import matplotlib.pyplot as plt
	import aplpy
	#from kapteyn import wcs
	#from astropy import wcs
	#from kapteyn import maputils
	import copy
	
	'''
	# 4C 35.06
	path = '/Users/users/shulevski/brooks/Research_Vault/A407_spix/4C3506_spectral_studies/images/'
	LOFAR = 'LOFAR_08G_regrid_crop.fits'
	VLA_4G = 'VLA_5G_crop.fits'
	VLA_1G = 'VLA_1G_regrid_crop.fits'
	WSRT = 'WSRT_1G_smooth_crop.fits'
	'''
	
	'''
	# VLSS J1431.8+1331
	path = '/Users/shulevski/Documents/Kapteyn/1431+1331_spix/'
	LOFAR = 'J1431+1331_116-170MHz_smooth_regrid_crop_allBW_regrid.fits'
	GMRT1 = 'J1431+1331_325MHz_smooth.fits'
	GMRT2 = 'J1431+1331_610MHz_smooth.fits'
	VLA = 'J1431_1400MHz_smooth.fits'
	'''
	
	'''
	# NGC 6251
	path = '/Users/shulevski/Documents/Research/Posters_and_talks/NAC_2015_Poster/'
	
	#lofar_im = 'NGC6251_80SB_avg.fits'
	lofar_im = 'lo.fits'
	#wsrt_im = 'NGC6251_92_regrid.fits'
	wsrt_im = 'wr.fits'
	'''
	
	''' # Attempt to solve for the rotation by reprojecting to the LOFAR header. Unsuccesfull.
	template = maputils.FITSimage(path + lofar_im)
	rotated = maputils.FITSimage(path + wsrt_im)
	corrected = rotated.reproject_to(template.hdr, interspatial=False)
	
	corrected.writetofits(path + "NGC6251_92_corrected_regridtoJ2000_derot.fits", clobber=True, append=False)
	'''
	
	'''
	# B2 0924+30
	path = '/Users/shulevski/Documents/Kapteyn/B20924+30_spix/'
	
	lofar_im = 'SB_000-300_7set_smoothtofinal_regrid.fits'
	wsrt_im = 'B20924+30_WSRT_608MHz_smoothtofinal_regrid.fits'
	nvss_im = 'B20924+30_NVSS_smoothtofinal.fits'
	'''

	'''
	# Savini

	path = '/Users/shulevski/Desktop/Savini/'
	lofar_im = 'LOFAR_crop.fits'
	gmrt_im = 'GMRT_crop.fits'
	'''

	'''
	# Giant
	path = '/Users/shulevski/Documents/Research/Projects/Multicore_cDs_LOFAR_Survey/TierI_data/P198+57/prefactor/subtract/'
	
	lofar_im = 'LOFAR_rg.fits'
	nvss_im = 'NVSS.fits'
	'''
	
	'''
	lof_hdu = fits.open(path + LOFAR, do_not_scale_image_data=True)
	lof_data = lof_hdu[0].data[0][0]
	lof_header = lof_hdu[0].header
	
	vla_1_hdu = fits.open(path + VLA_1G, do_not_scale_image_data=True)
	vla_1_data = vla_1_hdu[0].data[0][0]
	vla_1_header = vla_1_hdu[0].header
	
	wsrt_hdu = fits.open(path + WSRT, do_not_scale_image_data=True)
	wsrt_data = wsrt_hdu[0].data[0][0]
	wsrt_header = wsrt_hdu[0].header
	
	vla_4_hdu = fits.open(path + VLA_4G, do_not_scale_image_data=True)
	vla_4_data = vla_4_hdu[0].data[0][0]
	vla_4_header = vla_4_hdu[0].header
	'''
	'''
	lof_hdu = fits.open(path + lofar_im, do_not_scale_image_data=True)
	lof_data = lof_hdu[0].data[0][0]
	lof_header = lof_hdu[0].header
	
	wsrt_hdu = fits.open(path + wsrt_im, do_not_scale_image_data=True)
	wsrt_data = wsrt_hdu[0].data[0][0]
	wsrt_header = wsrt_hdu[0].header
	
	nvss_hdu = fits.open(path + nvss_im, do_not_scale_image_data=True)
	nvss_data = nvss_hdu[0].data
	nvss_header = nvss_hdu[0].header
	'''
	
	#'''
	#nvss_hdu = fits.open(path + nvss_im, do_not_scale_image_data=True)
	#nvss_data = nvss_hdu[0].data
	
	##wsrt_hdu = fits.open(path + wsrt_im, do_not_scale_image_data=True)
	##wsrt_data = wsrt_hdu[0].data
	##wsrt_header = wsrt_hdu[0].header
	
	##wsrt_image = maputils.FITSimage(path + wsrt_im)
	##lofar_image = maputils.FITSimage(path + lofar_im)
	#wsrt_image = maputils.prompt_fitsfile(defaultfile=path + wsrt_im, prompt=False, hnr=0)
	
	#print lof_header
	#print wsrt_header
	
	##wsrt_header['CTYPE1'] = 'RA---SIN'
	##wsrt_header['CTYPE2'] = 'DEC--SIN'
	
	#lof_header['CROTA1'] = 0.0
	#lof_header['CROTA2'] = 0.0
	
	#w = wcs.WCS(wsrt_header)
	#wsrt_data_rpr = w.wcs_pix2world(wsrt_data, 1)
	#new_hdu = fits.PrimaryHDU(wsrt_data_rpr.dat)
	#new_hdu.header = wsrt_header
	#new_hdu.header = w.to_header()
	#new_hdu = w.to_fits()[0]
	#new_hdu.writeto(path + "NGC6251_92_regrid_SIN.fits", clobber=True)
	#'''
	
	'''
	wsrt_hdu = fits.open(path + "NGC6251_92_regrid_SIN.fits", do_not_scale_image_data=True)
	wsrt_data = wsrt_hdu[0].data[0][0]
	
	specdata = wsrt_data
	factor = 3.5
	#freqs = np.array([140.e6, 1400.e6])
	freqs = np.array([140.e6, 325.e6])
	noise = factor * np.array([2.56e-3, 4.48e-4])
	print lof_data
	print wsrt_data
	'''
	
	'''
	path = '/Users/users/shulevski/brooks/Research_Vault/1431+1331_spix/Images_Feb_2014/'
	lofar_hdu0 = fits.open(path + "116-123MHz_regrid_crop.fits", do_not_scale_image_data=True)
	lofar_hdu1 = fits.open(path + "123-131MHz_regrid_crop.fits", do_not_scale_image_data=True)
	lofar_hdu2 = fits.open(path + "131-139MHz_regrid_crop.fits", do_not_scale_image_data=True)
	lofar_hdu3 = fits.open(path + "140-150MHz_regrid_crop.fits", do_not_scale_image_data=True)
	lofar_hdu4 = fits.open(path + "150-158MHz_regrid_crop.fits", do_not_scale_image_data=True)
	lofar_hdu5 = fits.open(path + "158-170MHz_regrid_crop.fits", do_not_scale_image_data=True)
	
	factor = 4.0
	freqs = np.array([120., 125., 135., 145., 155., 160.]) * 1.e6
	noise = factor * np.array([3.4, 2.7, 2.3, 1.8, 1.6, 1.2]) * 1.e-3
	print freqs
	
	lof_data0 = lofar_hdu0[0].data[0][0]
	lof_data1 = lofar_hdu1[0].data[0][0]
	lof_data2 = lofar_hdu2[0].data[0][0]
	lof_data3 = lofar_hdu3[0].data[0][0]
	lof_data4 = lofar_hdu4[0].data[0][0]
	lof_data5 = lofar_hdu5[0].data[0][0]
	specdata = lof_data0
	lof_header = lofar_hdu0[0].header
	'''
	'''
	path = '/Users/users/shulevski/brooks/Research_Vault/B20924+30/June-2014/Low_res/'
	#lofar_hdu0 = fits.open(path + "SB_020-040_nvss_regrid.fits", do_not_scale_image_data=True)
	lofar_hdu1 = fits.open(path + "SB_060-080_nvss_regrid.fits", do_not_scale_image_data=True)
	lofar_hdu2 = fits.open(path + "SB_100-120_nvss_regrid.fits", do_not_scale_image_data=True)
	lofar_hdu3 = fits.open(path + "SB_120-140_nvss_regrid.fits", do_not_scale_image_data=True)
	lofar_hdu4 = fits.open(path + "SB_240-260_nvss_regrid.fits", do_not_scale_image_data=True)
	lofar_hdu5 = fits.open(path + "SB_260-280_nvss_regrid.fits", do_not_scale_image_data=True)
	lofar_hdu6 = fits.open(path + "SB_280-300_nvss_regrid.fits", do_not_scale_image_data=True)
	wenss_hdu = fits.open(path + "B20924+30_WENSS_regrid.fits", do_not_scale_image_data=True)
	nvss_hdu = fits.open(path + "B20924+30_NVSS.fits", do_not_scale_image_data=True)
	
	factor = 0.2
	freqs = np.array([124.3, 132.1, 136., 159.5, 163.4, 167.3, 325., 1400.]) * 1.e6
	noise = factor * np.array([7.7, 4.5, 4.7, 3.2, 2.8, 2.5, 4., 0.8]) * 1.e-3
	#print freqs
	
	#lof_data0 = lofar_hdu0[0].data[0][0]
	lof_data1 = lofar_hdu1[0].data[0][0]
	lof_data2 = lofar_hdu2[0].data[0][0]
	lof_data3 = lofar_hdu3[0].data[0][0]
	lof_data4 = lofar_hdu4[0].data[0][0]
	lof_data5 = lofar_hdu5[0].data[0][0]
	lof_data6 = lofar_hdu5[0].data[0][0]
	wenss_data = wenss_hdu[0].data
	nvss_data = nvss_hdu[0].data
	specdata = nvss_data
	lof_header = lofar_hdu1[0].header
	'''
	'''
	path = '/Users/users/shulevski/brooks/Research_Vault/B20924+30/June-2014/High_res/'

	lofar_hdu = fits.open(path + "SB_000-300_7set.fits", do_not_scale_image_data=True)
	first_hdu = fits.open(path + "FIRST_smooth_regrid.fits", do_not_scale_image_data=True)
	'''
	
	'''
	# 4C 35.06
	
	factor = 5.
	freqs = np.array([61.6, 1357., 1415., 4885.]) * 1.e6
	noise = factor * np.array([105., 1.43, 0.29, 0.04]) * 1.e-3
	'''
	
	'''
	# VLSS J1431.8+1331
	
	lof_hdu = fits.open(path + LOFAR, do_not_scale_image_data=True)
	lof_data = lof_hdu[0].data[0][0]
	lof_header = lof_hdu[0].header
	
	gmrt_1_hdu = fits.open(path + GMRT1, do_not_scale_image_data=True)
	gmrt_1_data = gmrt_1_hdu[0].data[0]
	gmrt_1_header = gmrt_1_hdu[0].header
	
	gmrt_2_hdu = fits.open(path + GMRT2, do_not_scale_image_data=True)
	gmrt_2_data = gmrt_2_hdu[0].data[0][0]
	gmrt_2_header = gmrt_2_hdu[0].header
	
	vla_hdu = fits.open(path + VLA, do_not_scale_image_data=True)
	vla_data = vla_hdu[0].data[0]
	vla_header = vla_hdu[0].header
	
	#print len(vla_data[0]), len(vla_data[1])
	
	factor = 3.
	freqs = np.array([144., 325., 610., 1425.]) * 1.e6
	noise = factor * np.array([0.5, 0.1, 0.1, 0.05]) * 1.e-3
	'''
	
	'''
	#B2 0924+30
	
	lof_hdu = fits.open(path + lofar_im, do_not_scale_image_data=True)
	lof_data = lof_hdu[0].data[0][0]
	lof_header = lof_hdu[0].header
	
	wsrt_hdu = fits.open(path + wsrt_im, do_not_scale_image_data=True)
	wsrt_data = wsrt_hdu[0].data[0][0]
	wsrt_header = wsrt_hdu[0].header
	
	nvss_hdu = fits.open(path + nvss_im, do_not_scale_image_data=True)
	nvss_data = nvss_hdu[0].data
	nvss_header = nvss_hdu[0].header
	
	factor = 3.
	freqs = np.array([140., 608.5, 1400.]) * 1.e6
	noise = np.array([4., 1.6, 0.74]) * 1.e-3

	x_range = [80, 170]
	y_range = [100, 200]

	'''

	'''
	#Savini
	
	lof_hdu = fits.open(path + lofar_im, do_not_scale_image_data=True)
	lof_data = lof_hdu[0].data[0][0]
	lof_header = lof_hdu[0].header
	
	gmrt_hdu = fits.open(path + gmrt_im, do_not_scale_image_data=True)
	gmrt_data = gmrt_hdu[0].data[0][0]
	nvss_header = gmrt_hdu[0].header

	print len(gmrt_data[0]), len(gmrt_data[1])
	
	factor = 3.
	freqs = np.array([144., 607.]) * 1.e6
	noise = np.array([290., 350.]) * 1.e-6

	x_range = [10, 240]
	y_range = [10, 240]

	'''

	'''
	# Giant
	lof_hdu = fits.open(path + lofar_im, do_not_scale_image_data=True)
	lof_data = lof_hdu[0].data[0][0]
	print lof_data.shape
	lof_header = lof_hdu[0].header
	
	nvss_hdu = fits.open(path + nvss_im, do_not_scale_image_data=True)
	nvss_data = nvss_hdu[0].data
	print nvss_data.shape
	nvss_header = nvss_hdu[0].header
	
	factor = 1.
	freqs = np.array([126., 1400.]) * 1.e6
	noise = np.array([0.7, 0.74]) * 1.e-3 * factor

	x_range = [0, 61]
	y_range = [0, 61]
	'''
	
	#'''
	#NGC6251 & 3C236
	
	# 3C 236
	#path = '/Users/shulevski/Documents/Research/Projects/3C236/LOFAR_Factor_images_Feb_2017/dataset/'
	#path = '/Users/shulevski/Documents/Research/Projects/3C236/'
	#path = '/Users/shulevski/Documents/Research/Projects/3C236/LOFAR_Factor_images_Feb_2017/dataset/re-regridding/'

	path = '/home/shulevski/Documents/Research/Projects/3C236/LOFAR_Factor_images_Feb_2018/'
	
	#lofar_im = 'LOFAR_mean_regrid.fits'
	#wsrt_im = '3C236_WSRT_regrid_smooth.fits'
	#nvss_im = '3C236_NVSS_regird_smooth.fits'


	#lofar_im = '3C236_LOFAR_high_smooth.fits'
	#wsrt_im = '3C236_WSRT_smooth.fits'
	#nvss_im = '3C236_NVSS_smooth.fits'

	#lofar_im = 'LOFAR_mean_rgr.fits'
	#wsrt_im = '3C236_WSRT_j2_sm.fits'
	#nvss_im = '3C236_NVSS_sin_sm_rgr.fits'

	#lofar_bm_im = '3C236_LOFAR_sm.fits'
	#lofar_bm_im = '3C236_LOFAR_48asecorig_sm.fits'
	lofar_bm_im = '3C236_LOFAR_AGES.fits'

	#lofar_im = 'LSC.fits'
	lofar_im = '3C236_LOFAR_AGES.fits'
	#lofar_im = '3C236_LOFAR_48asecorig_sm.fits'
	wsrt_im = '3C236_WSRT_AGES.fits'
	#wsrt_im = '3C236_NVSS_sm.fits'
	nvss_im = 'NSC.fits'
	
	# NGC6251
	#path = '/Users/users/shulevski/brooks/Research_Vault/NGC_6251/LOFAR_Aug_2014/HBA_Low_Res/'
	
	#lofar_im = 'NGC6251_avg_9set_smooth_regrid.fits'
	#wsrt_im = 'NGC6251_92_corrected_regridtoJ2000.fits'
	
	lof_bm_hdu = fits.open(path + lofar_bm_im, do_not_scale_image_data=True)
	lof_bm_data = lof_bm_hdu[0].data[0][0]
	lof_bm_header = lof_bm_hdu[0].header

	lof_hdu = fits.open(path + lofar_im, do_not_scale_image_data=True)
	lof_data = lof_hdu[0].data[0][0]
	#lof_data = lof_hdu[0].data
	lof_header = lof_hdu[0].header
	print len(lof_data[:,0])

	wsrt_hdu = fits.open(path + wsrt_im, do_not_scale_image_data=True)
	wsrt_data = wsrt_hdu[0].data[0][0]
	#wsrt_data = wsrt_hdu[0].data
	wsrt_header = wsrt_hdu[0].header
	print len(wsrt_data[:,0])

	nvss_hdu = fits.open(path + nvss_im, do_not_scale_image_data=True)
	nvss_data = nvss_hdu[0].data
	nvss_header = nvss_hdu[0].header

	#x_range = [80, 400]
	#y_range = [80, 400]

	x_range = [80, 380]
	y_range = [100, 400]

	#x_range = [0, 238]
	#y_range = [0, 173]
	
	factor = 5.
	freqs = np.array([143.6, 608.5, 1400.]) * 1.e6
	#noise = np.array([1.6, 0.7, 0.4]) * 1.e-3
	noise = np.array([0.3, 0.7, 0.4]) * 1.e-3 # LSC image
	
	#freqs = np.array([144., 325]) * 1.e6
	#noise = factor * np.array([4., 2.3]) * 1.e-3
	
	#'''


	'''
	# A1318
	
	# 3C 236
	#path = '/Users/shulevski/Documents/Research/Projects/3C236/LOFAR_Factor_images_Feb_2017/dataset/'
	#path = '/Users/shulevski/Documents/Research/Projects/3C236/'
	#path = '/Users/shulevski/Documents/Research/Projects/3C236/LOFAR_Factor_images_Feb_2017/dataset/re-regridding/'

	path = '/home/shulevski/Documents/Research/Projects/A1318_relic/'
	
	#lofar_im = 'LOFAR_mean_regrid.fits'
	#wsrt_im = '3C236_WSRT_regrid_smooth.fits'
	#nvss_im = '3C236_NVSS_regird_smooth.fits'


	#lofar_im = '3C236_LOFAR_high_smooth.fits'
	#wsrt_im = '3C236_WSRT_smooth.fits'
	#nvss_im = '3C236_NVSS_smooth.fits'

	#lofar_im = 'LOFAR_mean_rgr.fits'
	#wsrt_im = '3C236_WSRT_j2_sm.fits'
	#nvss_im = '3C236_NVSS_sin_sm_rgr.fits'

	lofar_bm_im = 'LOFAR_smooth_regrid_wenss.fits'
	#lofar_bm_im = '3C236_LOFAR_48asecorig_sm.fits'
	#lofar_im = 'LSC.fits'
	lofar_im = 'LOFAR_smooth_regrid_wenss.fits'
	#wsrt_im = '3C236_WSRT_sm.fits'
	wsrt_im = 'A1318_WENSS.fits'
	nvss_im = 'I1136P56'
	
	lof_bm_hdu = fits.open(path + lofar_bm_im, do_not_scale_image_data=False)
	#lof_bm_data = lof_bm_hdu[0].data[0][0]
	lof_bm_data = lof_bm_hdu[0].data
	lof_bm_header = lof_bm_hdu[0].header

	lof_hdu = fits.open(path + lofar_im, do_not_scale_image_data=False)
	#lof_data = lof_hdu[0].data[0][0]
	lof_data = lof_hdu[0].data
	lof_header = lof_hdu[0].header
	print len(lof_data[:,0])

	wsrt_hdu = fits.open(path + wsrt_im, do_not_scale_image_data=True)
	wsrt_data = wsrt_hdu[0].data[0][0]
	wsrt_data = wsrt_hdu[0].data
	wsrt_header = wsrt_hdu[0].header
	print len(wsrt_data[:,0])

	nvss_hdu = fits.open(path + nvss_im, do_not_scale_image_data=False)
	nvss_data = nvss_hdu[0].data[0][0]
	nvss_header = nvss_hdu[0].header
	print len(nvss_data[:,0])

	#x_range = [450, 570] # nvss
	#y_range = [240, 320]

	x_range = [50, 300]
	y_range = [50, 300]
	
	factor = 3.
	freqs = np.array([143.65, 326.0, 1400.]) * 1.e6
	noise = np.array([461.5, 1192.7, 449.84]) * 1.e-6
	
	#freqs = np.array([144., 325]) * 1.e6
	#noise = np.array([4., 2.3]) * 1.e-3
	
	'''
	
	#lof_data = lofar_hdu[0].data[0][0]
	#first_data = first_hdu[0].data
	
	specdata = copy.deepcopy(lof_data)
	#lof_header = lofar_hdu[0].header

	for i in range(y_range[0], y_range[1]):
		for j in range(x_range[0], x_range[1]):
	#for i in range(len(gmrt_2_data[0,:])):
		#for j in range(len(gmrt_2_data[:,0])):
	#for i in np.linspace(0, 180, 181):
		#for j in np.linspace(0, 150, 151):
			#if lof_data[i, j] > noise[0] and nvss_data[i, j] > noise[1]:			
			#	fluxes = np.array([lof_data[i, j], nvss_data[i, j]])
			#if lof_data1[i, j] > noise[0] and lof_data2[i, j] > noise[1] and lof_data3[i, j] > noise[2] and lof_data4[i, j] > noise[3] and lof_data5[i, j] > noise[4] and lof_data5[i, j] > noise[5] and wenss_data[i, j] > noise[6] and nvss_data[i, j] > noise[7]:
			
			#if lof_data[i, j] > noise[0] and nvss_data[:, :, i, j] > noise[1]: # giant spix derivation
			#if vla_1_data[i, j] > noise[2] and vla_4_data[i, j] > noise[3]:
			
			#if lof_data[i, j] > noise[0] * factor and wsrt_data[i, j] > noise[1] * factor: # 3C236 low spix derivation
			if lof_data[i, j] > noise[0] * factor and wsrt_data[i, j] > noise[1] * factor: # 3C236 overall derivation
			#if lof_data[i, j] > noise[0] * factor and wsrt_data[i, j] > noise[1] * factor and nvss_data[i, j] > noise[2] * factor: # 3C236 curvature derivation
			#if lof_data[i, j] > noise[0] * factor and gmrt_data[i, j] > noise[1] * factor: # Savini
			#if lof_data[i, j] > noise[0] and wsrt_data[i, j] > noise[1] and nvss_data[i, j] > noise[2]: # NGC6251 spix derivation
			#if lof_data[i, j] > factor * noise[0] and wsrt_data[i, j] > factor * noise[1]: # spix derivation
			#if lof_data[i, j] > noise[0] and gmrt_1_data[i, j] > noise[1]: # J1431 spix derivation
			#if lof_data[i, j] > noise[0] and gmrt_2_data[i, j] > noise[2] and vla_data[i, j] > noise[3]: # J1431 curvature derivation
			#if nvss_data[i, j] > noise[1]:
				#fluxes = np.array([lof_data1[i, j], lof_data2[i, j], lof_data3[i, j], lof_data4[i, j], lof_data5[i, j], lof_data6[i, j], wenss_data[i, j], nvss_data[i, j]])
				#fluxes = np.array([vla_1_data[i, j], vla_4_data[i, j]])
				#fluxes = np.array([lof_data[i, j], wsrt_data[i, j], nvss_data[i, j]]) # NGC6251 curvature derivation
				fluxes = np.array([lof_data[i, j], wsrt_data[i, j]]) # spix derivation
				#fluxes = np.array([lof_data[i, j], nvss_data[:, :, i, j]]) # spix derivation
				#fluxes = np.array([lof_data[i, j], gmrt_data[i, j]]) # J1431 spix derivation
				#fluxes = np.array([lof_data[i, j], gmrt_2_data[i, j], vla_data[i, j]]) # J1431 curvature derivation
				
				#specdata[i, j] = np.polyfit(np.log10(freqs[0:2]), np.log10(fluxes[0:2]), 1)[0] - np.polyfit(np.log10(freqs[1:]), np.log10(fluxes[1:]), 1)[0]
				#specdata[i, j] = nvss_data[i, j]
				
				#errors = np.sqrt(np.array([noise[0]**2. + (fluxes[0] * 0.2)**2., noise[1]**2. + (fluxes[1] * 0.05)**2., noise[2]**2. + (fluxes[2] * 0.05)**2.])) # NGC6251 spix derivation
				errors = np.sqrt(np.array([noise[0]**2. + (fluxes[0] * 0.2)**2., noise[1]**2. + (fluxes[1] * 0.05)**2.]))
				#errors = np.sqrt(np.array([noise[0]**2., noise[1]**2.]))
				
				#errors = np.sqrt(np.array([noise[0]**2. + (fluxes[0] * 0.2)**2., noise[1]**2. + (fluxes[1] * 0.05)**2.])) # J1431 spix derivation
				#errors = np.sqrt(np.array([noise[0]**2. + (fluxes[0] * 0.2)**2., noise[1]**2. + (fluxes[1] * 0.05)**2., noise[2]**2. + (fluxes[2] * 0.05)**2.])) # J1431 curvature derivation
				
				# Likelyhood sampling to derive the fit error when fitting a straight line through two points
				#lower_samples =  np.random.uniform((fluxes - errors)[0], (fluxes + errors)[0], size = 100) # J1431 spix derivation
				#higher_samples =  np.random.uniform((fluxes - errors)[1], (fluxes + errors)[1], size = 100) # J1431 spix derivation

				#lower_samples =  np.random.uniform((fluxes - errors)[0], (fluxes + errors)[0], size = 100) # J1431 curv derivation
				#mid_samples =  np.random.uniform((fluxes - errors)[1], (fluxes + errors)[1], size = 100) # J1431 curv derivation
				#higher_samples =  np.random.uniform((fluxes - errors)[2], (fluxes + errors)[2], size = 100) # J1431 curv derivation
				#polys = []

				#for k in range(100):
				#	fit = np.polyfit(np.log10(freqs[[0, 1]]), np.log10([lower_samples[k], higher_samples[k]]), 1) # J1431 spix derivation
				#	polys.append(fit[0]) # J1431 spix derivation
					#polys.append(np.polyfit(np.log10(freqs[[0, 1]]), np.log10([lower_samples[k], mid_samples[k]]), 1)[0] - np.polyfit(np.log10(freqs[[1, 2]]), np.log10([mid_samples[k], higher_samples[k]]), 1)[0])
				#print np.std(polys)
				
				#specdata[i, j] = np.mean(polys)
				
				# Propagation of error calculation of spectral index (2 points)

				##con = 1. / np.log10(freqs[0] / freqs[1])
				##specdata[i, j] = con * np.log10(fluxes[0] / fluxes[1])

				#join_fl_err = np.sqrt(fluxes[0] / fluxes[1]) * ((noise[0] / fluxes[0])**2. + (noise[1] / fluxes[1])**2.)
				#spix_err = -1. * con * (join_fl_err / (np.log(10.) * (fluxes[0] / fluxes[1])))

				#specdata[i, j] = spix_err

				# ############

				specdata[i, j] = np.abs(1. / np.log(freqs[0] / freqs[1])) * np.sqrt(np.power(errors[0] / fluxes[0],2.) + np.power(errors[1] / fluxes[1], 2.)) # correct expression !!!!!!!!

				#spix_lo_err = np.abs(1. / np.log(freqs[0] / freqs[1])) * np.sqrt(np.power(errors[0] / fluxes[0],2.) + np.power(errors[1] / fluxes[1], 2.))
				#spix_hi_err = np.abs(1. / np.log(freqs[1] / freqs[2])) * np.sqrt(np.power(errors[1] / fluxes[1],2.) + np.power(errors[2] / fluxes[2], 2.))

				#specdata[i, j] = np.sqrt(np.power(spix_lo_err,2.) + np.power(spix_hi_err,2.))

				##############

				# Propagation of error calculation of spectral curvature (3 points)

				#con1 = 1. / np.log10(freqs[0] / freqs[1])
				#con2 = 1. / np.log10(freqs[1] / freqs[2])
				#specdata[i, j] = con1 * np.log10(fluxes[0] / fluxes[1]) - con2 * np.log10(fluxes[1] / fluxes[2])

				#join_fl_err1 = np.sqrt(fluxes[0] / fluxes[1]) * ((noise[0] / fluxes[0])**2. + (noise[1] / fluxes[1])**2.)
				#join_fl_err2 = np.sqrt(fluxes[1] / fluxes[2]) * ((noise[1] / fluxes[1])**2. + (noise[2] / fluxes[2])**2.)

				#spix_err1 = -1. * con1 * (join_fl_err1 / (np.log(10.) * (fluxes[0] / fluxes[1])))
				#spix_err2 = -1. * con2 * (join_fl_err2 / (np.log(10.) * (fluxes[1] / fluxes[2])))

				#spix_err1 = np.abs(1. / np.log(freqs[0] / freqs[1])) * np.sqrt(np.power(noise[0] / fluxes[0],2.) + np.power(noise[1] / fluxes[1], 2.))
				#spix_err2 = np.abs(1. / np.log(freqs[1] / freqs[2])) * np.sqrt(np.power(noise[1] / fluxes[1],2.) + np.power(noise[2] / fluxes[2], 2.))

				#specdata[i, j] = np.sqrt(spix_err1**2. + spix_err2**2.)
				
				#specdata[i, j] = np.std(polys) # errors
				#fit = np.polyfit(np.log10(freqs[[0, 1]]), np.log10(fluxes[[0, 1]]), 1, w=np.divide(errors, np.multiply(fluxes, np.log(10.0)))) #no likelihood sampling

				#fit = np.polyfit(np.log10(freqs[[0, 1]]), np.log10(fluxes[[0, 1]]), 1) #no likelihood sampling
				#specdata[i, j] = fit[0]
				#print fit
				#print specdata[i, j]
	
				#print fluxes
			else:
				specdata[i, j] = NaN

	new_hdu = fits.PrimaryHDU(specdata)
	new_hdu.header = lof_bm_header
	
	new_hdu.writeto(path + "spix_err.fits", clobber=True)
	im = aplpy.FITSFigure(path + "spix_err.fits", dimensions=[0, 1], slices=[0, 0]) # Will not read <astropy.io.fits.hdu.image.PrimaryHDU object>, so we have to use a tmp fits file
	###im = aplpy.FITSFigure(new_hdu, dimensions=[0, 1], slices=[0, 0]) # Will not read <astropy.io.fits.hdu.image.PrimaryHDU object>, so we have to use a tmp fits file
	#im.show_grayscale()
	#im.recenter(217.958333, 13.533333, radius=0.02)
	#im.show_colorscale(cmap='jet', stretch='linear', vmin=0., vmax=1.)
	#im.show_colorscale(cmap='jet', stretch='linear', vmin=-1.1, vmax=-0.4)
	im.show_colorscale(cmap='jet', stretch='linear', vmin=0.1, vmax=0.2)
	#im.show_colorscale(cmap='jet', stretch='linear')
	im.add_colorbar()
	im.add_beam()
	im.beam.set_edgecolor('black')
	im.beam.set_facecolor('white')
	im.beam.set_hatch('/')
	#im.set_theme('publication')
	im.tick_labels.set_xformat('hh:mm:ss')
	im.tick_labels.set_yformat('dd:mm')
	im.set_tick_color('k')
	
	# J1431+1331
	#im.show_contour('/Users/shulevski/Documents/Kapteyn/1431+1331_spix/' + 'S6.sphr.1400.fits', levels=4.e-5 * np.array([-3., 3., 9., 12., 20.]), colors=['gray'], linewidths=3)
	
	# 4C 35.06
	#im.show_contour(path+VLA_4G, levels=np.array([-5., 18., 27., 36., 81.]) * (noise[3]/factor), colors=['grey'])
	#im.show_contour(path+LOFAR, levels=1.*(noise[0]/factor)*np.array([-3.0, 6.0, 9.0, 12., 15.0, 21.0, 27.0, 33.0, 39.0, 45.0, 51.0, 57.0, 63.0, 69.0, 75.0, 81.0, 87.0, 93.0, 99.0, 105.0, 111.0, 117.0, 123.0, 129.0, 135.0, 141.0, 147.0, 153.0, 159., 165., 171., 177., 183.]), colors=['grey'])
	
	# B2 0924+30
	#im.show_contour(path+lofar_im, levels=1.*(noise[0])*np.array([-10., 10., 20., 30., 40., 50., 60.]), colors=['grey'], linewidths=2)
	#im.show_markers([141.97014], [29.98575], layer='core_pos', marker='x', edgecolor='black', s = 70, zorder = 3)
	
	# Savini
	#im.show_contour(path+lofar_im, levels=1.*(noise[0])*np.array([3., 10., 20., 50., 70.]), colors=['grey'], linewidths=2)
	
	# 3C 236

	levno = 10
	levbase = 2.
	levels = np.power(np.sqrt(levbase), range(levno))
	levels = np.insert(levels, 0, -levels[0], axis=0)
	im.show_contour(path+lofar_im, levels=factor * 6. * noise[0] * levels, colors=['grey'], linewidths=0.5)
	#im.show_contour(path+wsrt_im, levels=factor * noise[1] * levels, colors=['green'], linewidths=0.5)
	
	# Giant
	#im.show_contour(path+lofar_im, levels=(noise[0] / factor)*np.array([-3, 3, 6, 9, 15]), colors=['grey'], linewidths=1)

	plt.show()
	
def KGJP_model_LBA_HBA_GMRT_VLA():
	
	fig = plt.figure()
	axis = fig.add_subplot(111)
	#axis.grid()
	axis.tick_params(axis='x', labelsize=17)
	axis.tick_params(axis='y', labelsize=17)
	axis.tick_params(length=10)
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	
	freq = np.array([61.4, 132.5, 140.3, 326., 610., 1425.]) * 1.e6
	flux = np.array([0.71, 0.43, 0.37, 8.9e-2, 2.4e-2, 2.2e-3])
	rms = np.array([563.8, 86.5, 75., 19.1, 5.3, 0.53]) * 1.e-3
	
	rms_imp = [np.sqrt(rms[0]**2. + (0.1*flux[0])**2.), np.sqrt(rms[1]**2. + (0.1*flux[1])**2.), np.sqrt(rms[2]**2. + (0.1*flux[2])**2.), np.sqrt(rms[3]**2. + (0.05*flux[3])**2.), np.sqrt(rms[4]**2. + (0.05*flux[4])**2.), rms[5]]
	
	log_rms_imp = np.divide(rms_imp, np.multiply(flux, np.log(10.0)))
	
	axis.errorbar(freq, flux, rms_imp, marker='o', linestyle='none')
	
	#fitter = fit(freq, flux, rms_imp, 'CI_off', 'JP', constant_params = [0.1599, 4.e-6, -0.8], fitted_params = [40., 10., 1.e-25])
	
	#s = Synfit(0.1599, fitter.params[0], fitter.params[1], 4.e-6, np.linspace(10.e6, 1.5e9, 100) , -0.8, fitter.params[2], 'CI_off', 'JP')
	
	s = Synfit(0.1599, 135., 3.7, 4.e-6, np.linspace(10.e6, 1.5e9, 100) , -0.8, 4.6e-27, 'CI_off', 'JP')
	mod_tot_flux_KGJP = s()
	#s = Synfit(0.1599, 0., 135., 4.e-6, np.linspace(10.e6, 1.5e9, 100) , -0.8, 4.6e-27, 'CI', 'JP')
	#mod_tot_flux_CI = s()
	#s = Synfit(0.1599, 135., 0., 4.e-6, np.linspace(10.e6, 1.5e9, 100) , -0.8, 4.6e-27, 'delta', 'JP')
	#mod_tot_flux_JP = s()
	
	#print mod_tot_flux
	#print tot_shifted_flux
	axis.loglog(np.linspace(10.e6, 1.5e9, 100), mod_tot_flux_KGJP, '-r')
	#axis.loglog(np.linspace(10.e6, 1.5e9, 100), mod_tot_flux_CI, '-g')
	#axis.loglog(np.linspace(10.e6, 1.5e9, 100), mod_tot_flux_JP, '-b')
	
	xlabel(r'Frequency [Hz]', fontsize=18, fontweight='bold', color='#000000')
	ylabel(r'Flux density [Jy]', fontsize=18, fontweight='bold', color='#000000')
	plt.show()
	
def model_regions_KGJP():
	from scipy.stats import chi2
	
	'''
	# J1431.8+1331
	freq = [135.e6, 145.e6, 325.e6, 610.e6, 1425.e6]
	path = '/Users/users/shulevski/brooks/Research_Vault/1431+1331_spix/Images_Feb_2014/'
	fluxfile = 'region_data.txt'
	rmsfile = 'region_rms.txt'
	flux_regions = np.genfromtxt(path + fluxfile, comments='#')
	rms_regions = np.genfromtxt(path + rmsfile, comments='#')
	##alpha_inj = -0.6
	z = 0.1599
	'''
	#alpha_injs = np.array([-0.65, -0.75, -0.85])
	
	#'''
	#B2 0924+30
	path = '/Users/shulevski/Documents/Kapteyn/B20924+30_spix/'
	fluxfile = 'region_data_new.txt'
	'''
	freq = np.array([113., 132., 136., 159., 163., 167., 609., 1400.]) * 1.e6
	flux_regions = np.genfromtxt(path + fluxfile, comments='#')
	
	mask = np.array([1., 1., 1., 1., 1., 1., 0., 0.]) * 1.e-3
	beam_pix_num = np.array([23., 23., 23., 23., 23., 23., 27., 28.])
	rms_pix_num = np.array([1392., 1392., 1392., 1392., 1392.,1392., 1757., 1757.])
	'''
	freq = np.array([113., 132., 136., 159., 163., 167., 609., 1400.]) * 1.e6
	flux_regions = np.genfromtxt(path + fluxfile, comments='#')
	
	mask = np.array([1., 1., 1., 1., 1., 1., 0., 0.]) * 1.e-3
	beam_pix_num = np.array([28., 28., 28., 28., 28., 28., 27., 28.])
	rms_pix_num = np.array([1008., 1008., 1008., 1008., 1008., 1008., 1008., 1008.])

	##alpha_inj = -0.5
	z = 0.026141
	B = 1.35e-6 #B20924+20
	fit_arrs = []
	#'''
	#for alpha_inj in alpha_injs:
	##print "For injection index ", alpha_inj
	print " "
	##for region, rms in zip(flux_regions, rms_regions):
	for region in flux_regions:
		##rms = np.array([8.3, 5.9, 7.8, 2.9, 3.1, 2.6, 1.38, 0.72]) * 1.e-3 #B20924+20
		##norms = np.array([1.108621, 1.08504573, 1.07699903, 0.95513001, 0.92609498, 0.89703054, 1., 1.]) # beam normalization factors

		rms = np.array([8.5, 4.7, 4.8, 5.3, 3.2, 2.8, 1.4, 0.9]) * 1.e-3 #B20924+20
		###norms = np.array([1.108621, 1.08504573, 1.07699903, 0.95513001, 0.92609498, 0.89703054, 1., 1.]) # beam normalization factors
		norms = np.array([1., 1., 1., 1., 1., 1., 1., 1.]) # no beam normalization
		#if region[0] in range(5, 6):
		#if region[0] in range(7, 8):
		##if region[0] in [18, 19]:
		##if (region[0] not in [16, 23, 30, 36, 37, 41, 42, 44, 45, 50, 56]): #B20924+20
		for i in range(len(rms)):
			rms[i] = np.sqrt(np.power(np.divide(np.multiply(region[1]/beam_pix_num[i], rms[i]), np.sqrt(rms_pix_num[i]/beam_pix_num[i])),2.0) + np.power(np.divide(rms[i], np.sqrt(region[1]/beam_pix_num[i])),2.0) + np.power(region[i+2] * mask[i] * 0.2, 2.)) #B20924+20
		##flux = np.divide(np.array(region[2:]), norms) #B20924+20
		flux = np.divide(np.array(region[2:]) * 1.e-3, norms) #B20924+20
		print "Fluxes: ", flux
		
		##flux = np.array(region[[4, 5, 8, 9, 10]])
		##rms = np.array(rms[[4, 5, 8, 9, 10]])
		#B = region[13]
		##B=4.4e-6
		##B=3.92e-6
		#B=2.52e-6
		fits = []
		alpha_inj = -0.92
		
		print "Region ", region[0]
		print " "
		#'''
		# Instantaneous injection JP fit
		fitter_delJP = fit(freq, flux, rms, 'delta', 'JP', constant_params = [z, B, alpha_inj], fitted_params = [20., 1., 1.e-12])
		###fitter_delJP = fit(freq, flux, rms, 'delta', 'JP', constant_params = [z, B], fitted_params = [20., 1., 1.e-12, 0.6])
		fits.append(fitter_delJP)
		s_delJP = Synfit(z, 10.**fitter_delJP.params[0], 10.**fitter_delJP.params[1], B, np.linspace(1.e8, 1.5e9, 50) , alpha_inj, 10.**fitter_delJP.params[2], 'delta', 'JP')
		###s_delJP = Synfit(z, 10.**fitter_delJP.params[0], 10.**fitter_delJP.params[1], B, np.linspace(1.e8, 1.5e9, 50) , 10.**fitter_delJP.params[3] * -1, 10.**fitter_delJP.params[2], 'delta', 'JP')
		###mod_tot_flux_delJP = s_delJP()
		prms = np.array(fitter_delJP.params)
		fit_errors = (fitter_delJP.xerror * [10.**i for i in prms]) * np.log(10.)
		rv = chi2(fitter_delJP.dof)
		alpha = 0.05           # Select a p-value
		chi2_threshold = rv.ppf(1-alpha)
		reg_par_arr = np.array([region[0], alpha_inj, 10.**fitter_delJP.params[0], fit_errors[0], fitter_delJP.chi2_min if fitter_delJP.chi2_min <= chi2_threshold else 100.])
		fit_arrs.append(reg_par_arr)
		
		# CIJP fit
		#fitter_CIJP = fit(freq, flux, rms, 'CI', 'JP', constant_params = [z, region[12], alpha_inj], fitted_params = [20., 1., 1.e-15])
		#fits.append(fitter_CIJP)
		#s_CIJP = Synfit(z, 10.**fitter_CIJP.params[0], 10.**fitter_CIJP.params[1], B, np.linspace(1.e8, 1.5e9, 50) , alpha_inj, 10.**fitter_CIJP.params[2], 'CI', 'JP')
		#mod_tot_flux_CIJP = s_CIJP()
		
		#  KGJP fit
		##fitter_KGJP = fit(freq, flux, rms, 'CI_off', 'JP', constant_params = [z, B, alpha_inj], fitted_params = [20., 1., 1.e-28])
		##fits.append(fitter_KGJP)
		##s_KGJP = Synfit(z, 10.**fitter_KGJP.params[0], 10.**fitter_KGJP.params[1], B, np.linspace(1.e8, 1.5e9, 50) , alpha_inj, 10.**fitter_KGJP.params[2], 'CI_off', 'JP')
		##mod_tot_flux_KGJP = s_KGJP()
		
		# Model shape tests
		#s_KGJP = Synfit(z, 30., 20., B, np.linspace(1.e8, 1.5e9, 100) , alpha_inj, 1.e-28, 'CI_off', 'JP')
		#mod_tot_flux_KGJP = s_KGJP()
		#s_delJP = Synfit(z, 80., 10., B, np.linspace(1.e8, 2.e9, 100) , alpha_inj, 1.e-14, 'delta', 'JP')
		#mod_tot_flux_delJP = s_delJP()
		
		'''
		fig = plt.figure()
		#axis = fig.add_subplot(221)
		axis = fig.add_subplot(111)
		#axis.grid()
		axis.set_aspect('equal')
		axis.tick_params(axis='x', labelsize=17)
		axis.tick_params(axis='y', labelsize=17)
		axis.tick_params(length=10)
		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')
		
		axis.errorbar(freq, flux, rms, marker='o', linestyle='none')
		axis.loglog(np.linspace(1.e8, 1.5e9, 50), mod_tot_flux_delJP, '-r')
		#axis.loglog(np.linspace(1.e8, 1.5e9, 50), mod_tot_flux_CIJP, '-g')
		##axis.loglog(np.linspace(1.e8, 1.5e9, 50), mod_tot_flux_KGJP, '-k')
		
		xlabel('Frequency [Hz]', fontsize=18, fontweight='bold', color='#000000')
		ylabel('Flux density [Jy]', fontsize=18, fontweight='bold', color='#000000')
		title('Region ' + str(region[0]), fontsize=18, fontweight='bold', color='#000000')
		
		# Kardashev 1962, for delta JP breaks
		#nub_del = (3.4e8 * B**-3. * (10.**fitter_delJP.params[0] * 1.e6)**-2.) / 1.e9
		##nub_del = break_estimate(B * 1.e6, z, 10.**fitter_KGJP.params[0] + 10.**fitter_KGJP.params[1]) # KGJP, but with t_active = 0
		##print 'delJP break: ', nub_del, 'GHz'
		##print " "
		'''
		# B [\muG], nu_b [GHz], t_s = t_a + t_i [Myr]
		###nub_ci = break_estimate(region[12] * 1.e6, z, fitter_CIJP.params[1])
		###print  'CIJP break: ', nub_ci, 'GHz'
		'''
		#nub_kg_lo = break_estimate(B * 1.e6, z, 10.**fitter_KGJP.params[0] + 10.**fitter_KGJP.params[1])
		#nub_kg_hi = nub_kg_lo * ((10.**fitter_KGJP.params[0] + 10.**fitter_KGJP.params[1]) / 10.**fitter_KGJP.params[0])**2.
		#print  'KGJP break low: ', nub_kg_lo, 'GHz', 'KGJP break high: ', nub_kg_hi, 'GHz'
		#print " "
		
		##plt.axvline(nub_del * 1.e9, color='k', linewidth=2, linestyle='--')
		#plt.axvline(nub_ci * 1.e9, color='g', linewidth=2, linestyle='--')
		#plt.axvline(nub_kg_lo * 1.e9, color='b', linewidth=2, linestyle='--')
		#plt.axvline(nub_kg_hi * 1.e9, color='b', linewidth=2, linestyle='--')
	
		#xlabel(r'Frequency [Hz]', fontsize=18, fontweight='bold', color='#000000')
		#ylabel(r'Flux density [Jy]', fontsize=18, fontweight='bold', color='#000000')
		
		plt.xlim(1.e8, 2.e9)
		#plt.ylim(1.e-4, 1.e-1)
		'''
		
		'''
		#for f in range(len(fits)):
		# We want to plot the chi2 landscape
		# for a range of values of mu and sigma.
		model, loss = 'CI_off', 'JP'
		#if f == 0:
			#model = 'delta'
		#	model='CI_off'
		#elif f == 1:
		#	model = 'CI'
		#else:
		#	model = 'CI_off'
		
		print 'Model: ', model	
		#tr = 10.**fits[f].params[0] # relic age
		#ti = 10.**fits[f].params[1] # injection age
		tr_plot = 100.
		ti_plot = 10.
		sc_plot = 2.e-28
		ntr = 20.
		nti = 20.
		nsc = 20.
		Dtr = 100.; Dti = 10.; Dsc = 5.e-29
		tr_arr = np.linspace(tr_plot - Dtr, tr_plot + Dtr, ntr)
		ti_arr = np.linspace(ti_plot - Dti, ti_plot + Dti, nti)
		sc_arr = np.linspace(sc_plot - Dsc, sc_plot + Dsc, nsc)
		#Z = np.zeros((ntr, nti))
		#Z = np.zeros((nti, nsc))
		Z = np.zeros((ntr, nsc))

		# Get the Chi^2 landscape.
		#pars = 10.**fits[f].params
		
		#pars = np.log10(np.array([10., 0.1, 1.92849362405e-28])) # for ti, tr free
		#pars = np.log10(np.array([91., 0.1, 1.92849362405e-28])) # for ti, sc free
		pars = np.log10(np.array([91., 5., 1.92849362405e-28])) # for tr, sc free
		i = -1
		for sc in sc_arr:
		#for tr in tr_arr:
			i += 1
			j = -1
			#for ti in ti_arr:
			for tr in tr_arr:
				j += 1
				pars[0] = np.log10(tr)
				#pars[1] = np.log10(ti)
				pars[2] = np.log10(sc)
				#print pars
				Z[j, i] = (residuals(pars, (freq, flux, rms, z, region[13], alpha_inj, model, loss))**2).sum()
				#print tr, ti
				#print ti, sc
				print tr, sc
				print Z[j, i]
		#Z /= 100000.0
		#XY = np.meshgrid(tr_arr, ti_arr)
		#XY = np.meshgrid(ti_arr, sc_arr)
		XY = np.meshgrid(tr_arr, sc_arr)
		#print Z
		
		#contlevs = [1.0, 0.1, 0.5, 1.5, 2.0, 5, 10, 15, 20, 100, 200]
		contlevs = np.linspace(1., 200., 300)
		
		fig = plt.figure()
		#axis = fig.add_subplot(221)
		#axis = fig.add_subplot(111)
		#axis.grid()
		#axis.set_aspect('equal')
		#axis.tick_params(axis='x', labelsize=17)
		#axis.tick_params(axis='y', labelsize=17)
		#axis.tick_params(length=10)
		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')
		
		frame = fig.add_subplot(111)
		cs = frame.contour(XY[0], XY[1], Z, contlevs)
		zc = cs.collections[0]
		zc.set_color('red')
		zc.set_linewidth(2)
		frame.clabel(cs, contlevs, inline=False, fmt='%1.1f', fontsize=10, color='k')
		frame.set_title("Chi-squared contours of Synfit model", fontsize=10)
		frame.set_xlabel("Relic age [Myr]")
		#frame.set_xlabel("Active age [Myr]")
		frame.set_ylabel("Scale factor")
		#plt.savefig('/Users/users/shulevski/Desktop/chisqr_landscape.png', bbox_inches='tight')
		'''
		
		#J1431.8+1331
		
		#plt.savefig('/Users/users/shulevski/Desktop/J1431+1331/' + str(region[0]) + str(alpha_inj) + '_fit.png', bbox_inches='tight')
		
		# B20924+30
		#plt.savefig('/Users/users/shulevski/brooks/Research_Vault/B20924+30/August_2014/HBA_Low_Res/Spix/B29824+30_age_fits/' + str(region[0]) + '_fit.png', bbox_inches='tight')
		###plt.savefig('/Users/shulevski/Documents/Kapteyn/ExtEnD/B20924+30/' + 'inj_free_' + str(region[0]) + '_fit.eps', bbox_inches='tight')
		##plt.show()
		###plt.close()
	np.save(path + str(alpha_inj) + '_fit_results_JP_nocorrection', fit_arrs)

def region_age_maps():

	from astropy.io import fits
	#import pyfits
	import matplotlib.pyplot as plt
	import aplpy
	#from kapteyn import wcs
	#from astropy import wcs
	from kapteyn import maputils
	import copy
	
	#'''
	# B2 0924+30
	path = '/Users/shulevski/Documents/Kapteyn/B20924+30_spix/'
	
	#lofar_im1 = 'SB_000-020_regrid.fits'
	#lofar_im2 = 'SB_100-120_regrid.fits'
	#lofar_im3 = 'SB_120-140_regrid.fits'
	#lofar_im4 = 'SB_240-260_regrid.fits'
	#lofar_im5 = 'SB_260-280_regrid.fits'
	#lofar_im6 = 'SB_280-300_regrid.fits'

	lofar_im = 'SB_000-300_7set_smoothtofinal_regrid.fits'
	wsrt_im = 'B20924+30_WSRT_608MHz_smoothtofinal_regrid.fits'
	nvss_im = 'B20924+30_NVSS_smoothtofinal.fits'
	#'''
	
	#'''
	#B2 0924+30
	
	'''
	lof_hdu1 = fits.open(path + lofar_im1, do_not_scale_image_data=True)
	lof_data1 = lof_hdu1[0].data[0][0]
	lof_header1 = lof_hdu1[0].header

	lof_hdu2 = fits.open(path + lofar_im2, do_not_scale_image_data=True)
	lof_data2 = lof_hdu2[0].data[0][0]
	lof_header2 = lof_hdu2[0].header

	lof_hdu3 = fits.open(path + lofar_im3, do_not_scale_image_data=True)
	lof_data3 = lof_hdu3[0].data[0][0]
	lof_header3 = lof_hdu3[0].header

	lof_hdu4 = fits.open(path + lofar_im4, do_not_scale_image_data=True)
	lof_data4 = lof_hdu4[0].data[0][0]
	lof_header4 = lof_hdu4[0].header

	lof_hdu5 = fits.open(path + lofar_im5, do_not_scale_image_data=True)
	lof_data5 = lof_hdu5[0].data[0][0]
	lof_header5 = lof_hdu5[0].header

	lof_hdu6 = fits.open(path + lofar_im6, do_not_scale_image_data=True)
	lof_data6 = lof_hdu6[0].data[0][0]
	lof_header6 = lof_hdu6[0].header
	'''

	lof_hdu = fits.open(path + lofar_im, do_not_scale_image_data=True)
	lof_data = lof_hdu[0].data[0][0]
	lof_header = lof_hdu[0].header
	
	wsrt_hdu = fits.open(path + wsrt_im, do_not_scale_image_data=True)
	wsrt_data = wsrt_hdu[0].data[0][0]
	wsrt_header = wsrt_hdu[0].header
	
	nvss_hdu = fits.open(path + nvss_im, do_not_scale_image_data=True)
	nvss_data = nvss_hdu[0].data
	nvss_header = nvss_hdu[0].header
	
	factor = 3.
	
	#freqs = np.array([113., 132., 136., 159., 163., 167., 608.5, 1400.]) * 1.e6
	#noise = np.array([9.4, 5.8, 5.1, 4.7, 2.7, 2.5, 1.3, 0.8]) * 1.e-3

	freqs = np.array([140., 608.5, 1400.]) * 1.e6
	noise = np.array([4., 1.3, 0.8]) * 1.e-3
	#'''
	'''
	#age = copy.deepcopy(lof_data1)
	#ageerr = copy.deepcopy(lof_data1)
	age = copy.deepcopy(lof_data)
	ageerr = copy.deepcopy(lof_data)
	z = 0.026141
	B = 1.35e-6
	a_inj = -0.85

	delta = 1.
	B = B * 1.e-4 # [T]
	vol = 2.4e65 # m^3

	blc = [90, 160]
	trc = [130, 200]

	for i in range(trc[0], trc[1]):
		for j in range(blc[0], blc[1]):
			#if lof_data1[i, j] > factor * noise[0] and lof_data2[i, j] > factor * noise[1] and lof_data3[i, j] > factor * noise[2] and lof_data4[i, j] > factor * noise[3] and lof_data5[i, j] > factor * noise[4] and lof_data6[i, j] > factor * noise[5] and wsrt_data[i, j] > factor * noise[6] and nvss_data[i, j] > factor * noise[7]:
			if lof_data[i, j] > factor * noise[0] and wsrt_data[i, j] > factor * noise[1] and nvss_data[i, j] > factor * noise[2]:

				#fluxes = np.array([lof_data1[i, j], lof_data2[i, j], lof_data3[i, j], lof_data4[i, j], lof_data5[i, j], lof_data6[i, j], wsrt_data[i, j], nvss_data[i, j]])
				fluxes = np.array([lof_data[i, j], wsrt_data[i, j], nvss_data[i, j]])

				#corrections = np.array([1.108621, 1.08504573, 1.07699903, 0.95513001, 0.92609498, 0.89703054, 1., 1.])

				corrections = np.array([1.07699903, 1., 1.])
				fluxes = fluxes / corrections
				#errors = np.sqrt(np.array([noise[0]**2. + (fluxes[0] * 0.2)**2., noise[1]**2. + (fluxes[1] * 0.2)**2., noise[2]**2. + (fluxes[2] * 0.2)**2., noise[3]**2. + (fluxes[3] * 0.2)**2., noise[4]**2. + (fluxes[4] * 0.2)**2., noise[5]**2. + (fluxes[5] * 0.2)**2., noise[6]**2. + (fluxes[6] * 0.05)**2., noise[7]**2. + (fluxes[7] * 0.05)**2.]))

				errors = np.sqrt(np.array([noise[0]**2. + (fluxes[0] * 0.2)**2., noise[1]**2. + (fluxes[1] * 0.05)**2., noise[2]**2. + (fluxes[2] * 0.05)**2.]))

				print "Fitting pixel: ", i, j
				print "Fluxes: ", fluxes
				print "Errors: ", errors
				
				###fitter_JP = fit(freqs, fluxes, errors, 'delta', 'JP', constant_params = [z, B, a_inj], fitted_params = [80., 10., 1.e-10]) # do the fit
				###fitter_JP = fit(freqs, fluxes, errors, 'delta', 'JP', constant_params = [z, B, a_inj], fitted_params = [80., 10., 1.e-41]) # do the fit with Synfit_Eint.py

				# Leith fit
				fitter_JP = fit_leith(freqs, fluxes, errors, constant_params = [z, B, vol, delta, 'JP', a_inj], fitted_params = [60., 2.e8])
				
				fit_errors = (fitter_JP.xerror * [10.**k for k in fitter_JP.params]) * np.log(10.)
				
				from scipy.stats import chi2
				rv = chi2(fitter_JP.dof)
				
				xc = fitter_JP.chi2_min
				
				f = lambda x: -rv.pdf(x)
				x_max = fminbound(f,1,200)
				
				alpha = 0.001           # Select a p-value
				chi2max = max(3*x_max, fitter_JP.chi2_min)
				chi2_threshold = rv.ppf(1-alpha)
				
				if xc <= chi2_threshold:
					print "Success, T_off: ", 10.**fitter_JP.params[0]
					#age[i, j] = 10.**fitter_JP.params[0]
					age[i, j] = fitter_JP.chi2_min
					ageerr[i, j] = fit_errors[0]
					xc = 11.
					chi2_threshold = 10.
				else:
				   age[i, j] = NaN
				   ageerr[i, j] = NaN
			else:
				age[i, j] = NaN
				ageerr[i, j] = NaN
				
	new_hdu1 = fits.PrimaryHDU(age)
	new_hdu1.header = lof_header
	new_hdu1.writeto(path + "ages.fits", clobber=True)
	
	new_hdu2 = fits.PrimaryHDU(ageerr)
	new_hdu2.header = lof_header
	new_hdu2.writeto(path + "ageerror.fits", clobber=True)
	'''
	#'''
	im = aplpy.FITSFigure(path + "ages.fits", dimensions=[0, 1], slices=[0, 0]) # Will not read <astropy.io.fits.hdu.image.PrimaryHDU object>, so we have to use a tmp fits file
	
	#im.show_colorscale(cmap='gist_heat', stretch='log', vmin=40., vmax=180.)
	#im.show_colorscale(cmap='gist_earth', stretch='log', vmin=5., vmax=25.)
	im.show_colorscale(cmap='bone', stretch='log', vmin=0.01, vmax=15.)

	im.add_colorbar()
	#im.colorbar.set_axis_label_text('Age [Myr]')
	#im.colorbar.set_axis_label_text('Age error [Myr]')
	im.colorbar.set_axis_label_text('$\chi^{2}$ of model fit')
	
	###im.colorbar.set_axis_label_font(size=12, weight='bold')
	im.add_beam()
	im.beam.set_edgecolor('black')
	im.beam.set_facecolor('white')
	im.beam.set_hatch('/')
	#im.set_theme('publication')
	im.tick_labels.set_xformat('hh:mm:ss')
	im.tick_labels.set_yformat('dd:mm')
	im.set_tick_color('k')
	
	im.show_contour(path+lofar_im, levels=1.*(noise[0])*np.array([-10., 10., 20., 30., 40., 50., 60.]), colors=['grey'], linewidths=2)
	im.show_markers([141.97014], [29.98575], layer='core_pos', marker='x', edgecolor='black', s = 70, zorder = 3)
	
	plt.show()
	#'''
	
def spec_tomography():
	from astropy.io import fits
	import pyfits
	import aplpy
	import matplotlib.pyplot as plt
	#from kapteyn import wcs
	from astropy import wcs
	from kapteyn import maputils
	import copy
	
	'''
	# NGC 6251
	freq_arr = [140.e6, 325.e6]
	spix_arr = linspace(-0.5, -1.8, 40)
	path = '/Users/users/shulevski/brooks/Research_Vault/NGC_6251/LOFAR_Aug_2014/HBA_Low_Res/'
	lofar_hdu = fits.open(path + "NGC6251_avg_9set_smooth_regrid.fits", do_not_scale_image_data=True)
	vla_hdu = fits.open(path + "NGC6251_92_corrected_regridtoJ2000.fits", do_not_scale_image_data=True)
	lof_data = lofar_hdu[0].data[0][0]
	vla_data = vla_hdu[0].data[0][0]
	lof_header = lofar_hdu[0].header
	
	noise = factor * np.array([2.5, 1.]) * 1.e-3
	hdudata = np.empty([40, 526, 526])
	'''
	
	# B2 0924+30
	freq_arr = np.array([140., 608., 1420.]) * 1.e6
	spix_arr = linspace(-0.5, -2., 80)
	
	path = '/Users/users/shulevski/brooks/Research_Vault/B20924+30/August_2014/HBA_Low_Res/Spix/'
	lofar_im = 'SB_000-300_7set_smoothtofinal_regrid.fits'
	wsrt_im = 'B20924+30_WSRT_608MHz_smoothtofinal_regrid.fits'
	nvss_im = 'B20924+30_NVSS_smoothtofinal.fits'
	
	lofar_hdu = fits.open(path + lofar_im, do_not_scale_image_data=True)
	wsrt_hdu = fits.open(path + wsrt_im, do_not_scale_image_data=True)
	nvss_hdu = fits.open(path + nvss_im, do_not_scale_image_data=True)
	
	lof_data = lofar_hdu[0].data[0][0]
	wsrt_data = wsrt_hdu[0].data[0][0]
	nvss_data = nvss_hdu[0].data
	
	lof_header = lofar_hdu[0].header
	
	factor = 3.
	noise = factor * np.array([4., 1.6, 0.74]) * 1.e-3
	
	hdudata = np.empty([80, 300, 300])
	tomdata = copy.deepcopy(nvss_data)
	
	print 'LOFAR image size: ', len(lof_data[0,:]), ' x ', len(lof_data[:,0]), ' pix'
	print 'WSRT image size: ', len(nvss_data[0,:]), ' x ', len(nvss_data[:,0]), ' pix'
	
	for k in range(len(spix_arr)):
		for i in range(len(nvss_data[0,:])):
			for j in range(len(nvss_data[:,0])):
				if lof_data[i, j] > noise[0] and wsrt_data[i, j] > noise[1]:
					tomdata[i, j] = lof_data[i, j] - nvss_data[i, j] * (freq_arr[0] / freq_arr[2])**spix_arr[k]
					#print 'LOFAR value: ', lof_data[i, j], 'JVLA value: ', vla_data[i, j]
					#print 'Derived spix: ', np.log10(lof_data[i, j] / vla_data[i, j]) / np.log10(freq_arr[0] / freq_arr[1])
				else:
					tomdata[i, j] = 0.2
		hdudata[k] = tomdata
		#if k == 5:
		#	print 'Spix:', spix_arr[k]
		#	plt.imshow(hdudata[k], vmin=-5., vmax=0.)
		#	plt.show()
	new_hdu = fits.PrimaryHDU(hdudata)
	new_hdu.header = lof_header
	new_hdu.writeto(path + "tom.fits", clobber=True)
	'''
	im = aplpy.FITSFigure(path + "temp_1.fits") # Will not read <astropy.io.fits.hdu.image.PrimaryHDU object>, so we have to use a tmp fits file
	#im.show_grayscale()
	im.show_colorscale(cmap='jet', stretch='linear', vmin=0., vmax=0.09)
	#im.show_colorscale(cmap='jet', stretch='linear')
	im.add_colorbar()
	im.add_beam()
	plt.show()
	'''

def fit_wrapper(args):

	return fit(*args[0])

def integrated_spectra_model_fit():
	
	# 4C35.06
	
	#frequencies_int = np.array([26.3, 61.6, 74., 81.5, 178., 232., 327., 408., 1360., 1400., 1490., 2700., 4850.]) * 1.e6
	#flux_int = np.array([47., 17.04, 13.92, 13.8, 5.01, 4.43, 3.27, 2.3, 0.76, 0.75, 0.71, 0.37, 0.23])
	#rms_int = np.array([6., 3.41, 0.34, 1.8, 0.87, 0.05, 0.01, 0.2, 0.03, 0.002, 0.003, 0.02, 0.03])
	
	# A
	#frequencies_int = np.array([61.6, 74., 325., 1360.]) * 1.e6
	#flux_int = np.array([0.77, 0.51, 0.075, 0.0023])
	#rms_int = np.array([0.16, 0.12, 0.005, 0.0007])
	
	# C
	#frequencies_int_c = np.array([61.6, 325., 1550., 4700.]) * 1.e6
	#flux_int_c = np.array([9.1, 1.6, 0.57, 0.2])
	#rms_int_c = np.array([1.7 * 1.1, 0.4, 0.1, 0.01])
	
	'''
	# E
	frequencies_int = np.array([61.6, 74., 325., 1360.]) * 1.e6
	flux_int = np.array([0.96, 0.32, 0.022, 0.0015])
	rms_int = np.array([0.20, 0.12, 0.005, 0.0007])
	'''
	
	'''
	# Marisa's blob
	
	frequencies_int_c = np.array([116., 155., 325., 1400., 4850.]) * 1.e6
	flux_int_c = np.array([1.4, 1.2, 0.8, 0.27, 0.03])
	rms_int_c = np.array([1.4 * 0.2, 1.2 * 0.2, 0.8 * 0.2, 0.27 * 0.1, 0.03 * 0.1])	
	'''
	
	'''
	freq = [135.e6, 145.e6, 325.e6, 610.e6, 1425.e6]
	
	frequencies_int = np.array([74., 135., 325., 610., 1425.]) * 1.e6
	flux_int = np.array([3.72, 1.95, 0.36, 0.11, 14.7e-3])
	rms_int = np.array([0.42, 0.39, 0.07, 5.5e-3, 2.3e-4])
	
	path = '/Users/users/shulevski/brooks/Research_Vault/1431+1331_spix/Images_Feb_2014/'
	fluxfile = 'region_data.txt'
	rmsfile = 'region_rms.txt'
	flux_regions = np.genfromtxt(path + fluxfile, comments='#')
	rms_regions = np.genfromtxt(path + rmsfile, comments='#')
	z = 0.1599
	B = 3.925e-6
	models_kgjp = []
	models_jp = []
	##t_on = [10., 10., 0.5, 10., 10., 0.8, 10., 5.3, 10., 1.4, 10., 0.5, 10., 0.5, 0.5, 0.5, 0.5]
	##t_off = [44., 132., 91.5, 77.8, 91., 119.2, 96.2, 150., 124., 103.5, 139.4, 150., 129., 123.2, 113.8, 131.5, 126.2]
	##sf = [8.62249868331e-29, 5.06843357843e-28, 2.42781173571e-26, 6.80499029696e-28, 1.92849362405e-28, 7.42392516848e-25, 5.93660745548e-28, 1.64493268121e-27, 1.81387943902e-27, 9.84927318334e-27, 3.86536584909e-28, 1.08340941581e-27, 6.3853280233e-29, 1.00944931637e-25, 9.63162814078e-26, 2.36518445057e-25, 1.4129917958e-25]
	
	t_on = [0., 0., 20.4, 0., 0., 22.9, 0., 0., 0., 0., 0., 0., 0., 0., 0., 7.8, 0.]
	t_off = [0., 0., 108.4, 60., 0., 117.4, 48.6, 91.3, 113.2, 104.7, 69.8, 97.2, 60.6, 121.4, 131.6, 132.4, 123.1]
	sf = [0., 0., 6.22562852638e-28, 1.51287573605e-14, 0., 2.63598339506e-27, 1.48384682275e-16, 1.67568057692e-16, 1.2155481949e-13, 2.70376227236e-13, 6.16391083476e-17, 3.45215520373e-17, 3.34619502658e-17, 1.52507826182e-12, 1.46437100648e-12, 1.43050398374e-26, 2.10461171914e-12]
	a_inj = [0., 0., -0.88, -1.12, 0., -0.74, -1.5, -1.5, -1.03, -0.93, -1.5, -1.5, -1.5, -0.6, -0.6, -0.6, -0.6]
	
	'''
	
	#'''
	# B2 0924+30
	#frequencies_int = np.array([63.2, 112.6, 124.3, 132.1, 136.0, 159.5, 163.4, 167.3, 151., 325., 609., 1400., 4750., 10550.]) * 1.e6
	#frequencies_int = np.array([63.2, 112.6, 132.1, 136.0, 151., 159.5, 163.4, 167.3, 325., 609., 1400., 4750., 10550.]) * 1.e6
	frequencies_int = np.array([63.2, 112.6, 132.1, 140.0, 151., 159.5, 163.4, 167.3, 325., 609., 1400., 4750., 10550.]) * 1.e6
	#flux_int = np.array([6501.5, 8384.4, 8937.3, 6774.5, 6738.5, 5214.0, 4751.2, 4701.7, 4600., 2425., 1094., 420., 60., 10.]) * 1.e-3
	#rms_int = np.array([19.5, 7.6, 4.5, 4.4, 360., 3.5, 2.2, 1.9, 124., 56., 43., 7., 4.]) * 1.e-3
	rms_int = np.array([19.5, 7.6, 4.5, 2.7, 360., 3.5, 2.2, 1.9, 124., 56., 43., 7., 4.]) * 1.e-3

	#flux_int = np.array([6501.5, 8384.4 / 1.108621, 6774.5 / 1.08504573, 6738.5 / 1.07699903, 4600., 5214.0 / 0.95513001, 4751.2 / 0.92609498, 4701.7 / 0.89703054, 2425., 1094., 420., 60., 10.]) * 1.e-3 # beam normalization correction

	flux_int = np.array([6501.5, 8384.4 / 1.108621, 6774.5 / 1.08504573, 6306, 4600., 5214.0 / 0.95513001, 4751.2 / 0.92609498, 4701.7 / 0.89703054, 2425., 1094., 420., 60., 10.]) * 1.e-3 # no correction at 140 MHz

	#LOFAR
	#flux_pix_num = np.array([1065., 1619., 1619., 1619., 1619., 1619., 1619., 1619.])
	#beam_pix_num = np.array([10.4, 16.4, 16.6, 16.6, 16.6, 16.6, 16.6, 16.6])
	#rms_pix_num = np.array([916., 1398., 1398., 1398., 1398., 1398., 1398., 1398.])

	#flux_pix_num = np.array([1065., 1619., 1619., 1619., 1619., 1619., 1619., 1619.])
	flux_pix_num = np.array([1065., 1619., 1619., 1653., 1619., 1619., 1619., 1619.])
	#beam_pix_num = np.array([10.4, 16.4, 16.6, 16.6, 16.6, 16.6, 16.6, 16.6])
	beam_pix_num = np.array([10.4, 16.4, 16.6, 28.29, 16.6, 16.6, 16.6, 16.6])
	#rms_pix_num = np.array([916., 1398., 1398., 1398., 1398., 1398., 1398., 1398.])
	rms_pix_num = np.array([916., 1398., 1398., 1327., 1398., 1398., 1398., 1398.])
	#'''
	
	'''
	# Quasar (Ekers)
	
	frequencies_int = np.array([112.6, 124.3, 132.1, 136.0, 159.5, 163.4, 167.3, 151., 325., 609., 1400., 4750., 5000., 10550.]) * 1.e6
	flux_int = np.array([157., 236., 143., 149., 142.0, 117.2, 117.5, 150., 101., 73., 54., 53., 32., 22.]) * 1.e-3
	rms_int = np.array([51., 70., 46., 48., 46., 40., 40., 22., 6., 2., 1., 2., 1., 2.]) * 1.e-3
	#LOFAR
	flux_pix_num = np.array([1619., 1619., 1619., 1619., 1619., 1619., 1619.])
	beam_pix_num = np.array([16.4, 16.6, 16.6, 16.6, 16.6, 16.6, 16.6])
	rms_pix_num = np.array([1398., 1398., 1398., 1398., 1398., 1398., 1398.])
	'''
	
	for i in [0, 1, 2, 3, 5, 6, 7]:
		rms_int[i] = np.sqrt(np.power(np.divide(np.multiply(flux_pix_num[i]/beam_pix_num[i], rms_int[i]), np.sqrt(rms_pix_num[i]/beam_pix_num[i])),2.0) + np.power(np.divide(rms_int[i], np.sqrt(flux_pix_num[i]/beam_pix_num[i])),2.0) + np.power(flux_int[i] * 0.2, 2.))
	
	fig = plt.figure()
	axis = fig.add_subplot(111)
	#axis.grid()
	axis.set_aspect('equal')
	axis.tick_params(axis='x', labelsize=17)
	axis.tick_params(axis='y', labelsize=17)
	axis.tick_params(length=10)
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	
	#'''
	# B20924+30
	models = []
	z = 0.026141
	B = 1.35e-6
	#B = 1.6e-6 # from Jamrozy's paper
	#a_inj = 0.85
	#'''
	
	'''
	# 4C35.06
	z = 0.046276
	B=5.34e-6
	a_inj = -0.8
	'''
	
	'''
	# Marisa's blob
	
	z = 0.05
	B=1.e-6
	#a_inj = -0.8
	'''
	
	'''
	# J1431.8+1331
	for idx in range(len(flux_regions)):
		if flux_regions[idx][0] in [3, 6, 16]:
			s_KGJP = Synfit(z, t_off[idx], t_on[idx], B, np.linspace(8.e7, 2.e9, 100) , a_inj[idx], sf[idx], 'CI_off', 'JP')
			mod_tot_flux_KGJP = s_KGJP()
			models_kgjp.append(mod_tot_flux_KGJP)
			models_jp.append(0.)
		else:
			s_JP = Synfit(z, t_off[idx], t_on[idx], B, np.linspace(8.e7, 2.e9, 100) , a_inj[idx], sf[idx], 'delta', 'JP')
			mod_tot_flux_JP = s_JP()
			models_jp.append(mod_tot_flux_JP)
			models_kgjp.append(0.)
		if idx in [3, 15]:
			flux = np.array([flux_regions[idx][4], flux_regions[idx][5], flux_regions[idx][8], flux_regions[idx][9], flux_regions[idx][10]])
			rms = np.array([rms_regions[idx][4], rms_regions[idx][5], rms_regions[idx][8], rms_regions[idx][9], rms_regions[idx][10]])
			axis.scatter(freq, flux, marker='o', c='r', s = 22, zorder=-100)
			print 'Flux', flux
			print 'RMS', rms
			axis.errorbar(freq, flux, rms, marker=None, fmt=None, ecolor='gray')
			
	axis.loglog(np.linspace(8.e7, 2.e9, 100), models_jp[3], '-b')
	axis.loglog(np.linspace(8.e7, 2.e9, 100), models_kgjp[15], '-r')
	axis.loglog(np.linspace(8.e7, 2.e9, 100), [sum(x) for x in zip(models_kgjp[2],models_jp[3],models_kgjp[5],models_jp[6],models_jp[7],models_jp[8],models_jp[9],models_jp[10],models_jp[11],models_jp[12],models_jp[13],models_jp[14],models_kgjp[15],models_jp[16])], '--g')
	'''
	
	'''
	# 4C35.06
	#axis.scatter(frequencies_int, flux_int, marker='o', c='r', s = 22, zorder=-100)
	fitter_KGJP_int = fit(frequencies_int_c, flux_int_c, rms_int_c, 'CI_off', 'JP', constant_params = [z, B, a_inj], fitted_params = [20., 5., 1.e-12])
	s_KGJP_int = Synfit(z, 10**fitter_KGJP_int.params[0], 10**fitter_KGJP_int.params[1], B, np.linspace(5.e7, 5.e9, 100) , a_inj, 10**fitter_KGJP_int.params[2], 'CI_off', 'JP')
	mod_tot_flux_KGJP_int = s_KGJP_int()
	
	#axis.scatter(frequencies_int, flux_int, marker='o', c='b', s = 22, zorder=-100)
	#axis.errorbar(frequencies_int, flux_int, np.array([0.42, 0.41, 0.07, 5.5e-3, 2.3e-4]), marker=None, fmt=None, ecolor='gray')
	#axis.loglog(np.linspace(5.e7, 10.e10, 100), mod_tot_flux_KGJP_int, '--k')
	
	#axis.scatter(frequencies_int[0:2], flux_int[0:2], marker='o', c='b', s = 30, zorder=-100)
	#axis.scatter(frequencies_int[0], flux_int[0], marker='o', c='r', s = 35, zorder=-100) # LOFAR points
	#axis.errorbar(frequencies_int[0:2], flux_int[0:2], rms_int[0:2], marker=None, fmt=None, ecolor='gray')
	#axis.loglog(frequencies_int, flux_int, marker=None, linestyle='None')
	#axis.errorbar(frequencies_int[2:], flux_int[2:], xerr=None, yerr=[[2.e-2, 1.4e-3], [0., 0.]], markerfacecolor='none', ecolor='gray', marker='h', markersize=8, linestyle='none', lolims=True)
	
	#flux_int[2] = flux_int[2] - 0.8e-2
	#flux_int[3] = flux_int[3] - 1.3e-3
	#pfit, cov = np.polyfit(np.log10(frequencies_int[0:2]), np.log10(flux_int[0:2]), 1, cov=True)
	#pfit = np.polyfit(np.log10(frequencies_int[0:2]), np.log10(flux_int[0:2]), 1)
	#pfit2 = np.polyfit(np.log10(frequencies_int), np.log10(flux_int), 2)
	
	#print "Spectral index: ", pfit[0]
	#print "Fit error: ", np.sqrt(-1. * cov[0][0])
	
	#polynom = np.poly1d(pfit)
	#quad = np.poly1d(pfit2)
	#axis.loglog(linspace(2.e7, 8.e9, 1000), 10**polynom(np.log10(linspace(2.e7, 8.e9, 1000))), '-g')
	#axis.loglog(linspace(5.e7, 2.e9, 1000), 10**quad(np.log10(linspace(5.e7, 2.e9, 1000))), '--b')
	
	#plt.xlim(1.e7, 1.e10)
	#plt.ylim(1.e-1, 1.e2)
	
	# Model
	axis.scatter(frequencies_int_c, flux_int_c, marker='*', c='g', s = 40, zorder=-100)
	axis.scatter(frequencies_int_c[0], flux_int_c[0], marker='*', c='r', s = 40, zorder=-100)
	axis.errorbar(frequencies_int_c, flux_int_c, rms_int_c, marker=None, fmt=None, ecolor='gray')
	axis.loglog(np.linspace(5.e7, 5.e9, 100), mod_tot_flux_KGJP_int, '--b')
	'''
	
	'''
	# Marisa's blob
	#axis.scatter(frequencies_int, flux_int, marker='o', c='r', s = 22, zorder=-100)
	fitter_KGJP_int = fit(frequencies_int_c, flux_int_c, rms_int_c, 'delta', 'JP', constant_params = [z, B], fitted_params = [40., 10., 1.e-10, 0.6])
	s_KGJP_int = Synfit(z, 10**fitter_KGJP_int.params[0], 10**fitter_KGJP_int.params[1], B, np.linspace(5.e7, 5.e9, 100) , -1. * 10**fitter_KGJP_int.params[3], 10**fitter_KGJP_int.params[2], 'delta', 'JP')
	mod_tot_flux_KGJP_int = s_KGJP_int()
	
	# Model
	axis.scatter(frequencies_int_c, flux_int_c, marker='*', c='g', s = 40, zorder=-100)
	axis.scatter(frequencies_int_c[0], flux_int_c[0], marker='*', c='r', s = 40, zorder=-100)
	axis.errorbar(frequencies_int_c, flux_int_c, rms_int_c, marker=None, fmt=None, ecolor='gray')
	axis.loglog(np.linspace(5.e7, 5.e9, 100), mod_tot_flux_KGJP_int, '--b')
	'''
	
	#'''
	# B2 0924+30
	
	#rms_int = rms_int / 2.

	'''# Cython realization - no speedup
	import pyximport
	pyximport.install(setup_args={'include_dirs':[np.get_include()]})
	from fitter import fit as cy_fit

	fitter_CI_off_int = cy_fit(frequencies_int[1:], flux_int[1:], rms_int[1:], 'CI_off', 'JP', constant_params = [z, B], fitted_params = [40., 40., 1.e-12, 0.60])
	'''
	''' # Python multiprocessing realization - no speedup
	from multiprocessing import Pool
	from functools import partial
	import itertools
	arguments = [frequencies_int[1:], flux_int[1:], rms_int[1:], 'CI_off', 'JP', [z, B], [40., 40., 1.e-12, 0.60]]
	arg_it = itertools.repeat(arguments, 10)
	pool = Pool(processes=10)

	fitter_CI_off_int = pool.map_async(fit_wrapper, itertools.izip(itertools.repeat((frequencies_int[1:], flux_int[1:], rms_int[1:], 'CI_off', 'JP', [z, B], [40., 40., 1.e-12, 0.60])),range(10)))
	#fitter_CI_off_int = pool.map(fit, [arguments], 1)
	pool.close()
	pool.join()

	print fitter_CI_off_int
	'''
	''' # Kapteyn package fitter
	###fitter_CI_off_int = fit(frequencies_int[1:], flux_int[1:], rms_int[1:], 'CI_off', 'JP', constant_params = [z, B], fitted_params = [40., 40., 1.e-12, 0.60])
	fitter_CI_off_int = fit(frequencies_int[1:], flux_int[1:], rms_int[1:], 'CI_off', 'JP', constant_params = [z, B], fitted_params = [40., 40., 1.e-9, 0.60]) # with Synfit.py - integration by parts
	###fitter_CI_off_int = fit(frequencies_int[1:], flux_int[1:], rms_int[1:], 'CI_off', 'JP', constant_params = [z, B], fitted_params = [40., 40., 1.e-41, 0.60]) # with Synfit_Eint.py
	#fitter_CI_off_int = fit(frequencies_int[1:], flux_int[1:], rms_int[1:], 'CI_off', 'JP', constant_params = [z, B], fitted_params = [40., 40., 1.e-6, 0.60]) # with Synfit_Eint.py - integration by parts

	s_CI_off_int = Synfit(z, 10**fitter_CI_off_int.params[0], 10**fitter_CI_off_int.params[1], B, np.linspace(1.e8, 11.5e9, 100) , 10**fitter_CI_off_int.params[3] * -1., 10**fitter_CI_off_int.params[2], 'CI_off', 'JP')
	mod_tot_flux_CI_off_int = s_CI_off_int()
	'''
	
	#''' Leith code fit with the Kapteyn package
	
	delta = 1.
	B = B * 1.e-4 # [T]
	vol = 2.4e65 # m^3
	
	
	# CI_off model
	
	#fitter_CI_off_int = fit_leith(frequencies_int[1:], flux_int[1:], rms_int[1:], constant_params = [z, B, vol, delta, 'CI_off'], fitted_params = [30., 15., 5.e-6, 0.60])
	
	fitter_CI_off_int = fit_leith(frequencies_int[[3, 4, 8, 9, 10, 11, 12]], flux_int[[3, 4, 8, 9, 10, 11, 12]], rms_int[[3, 4, 8, 9, 10, 11, 12]], constant_params = [z, B, vol, delta, 'CI_off'], fitted_params = [30., 15., 5.e-6, 0.60])
	
	
	#fitter_CI_off_int = fit_leith(frequencies_int[[3, 4, 8, 9, 10, 11, 12]], flux_int[[3, 4, 8, 9, 10, 11, 12]], rms_int[[3, 4, 8, 9, 10, 11, 12]], constant_params = [z, B, vol, delta, 'CI_off', a_inj], fitted_params = [30., 60., 5.e-2])
	
	gamma = 1.0 - 2.0 * (10**fitter_CI_off_int.params[3] * -1.)
	
	#gamma = 1.0 - 2.0 * (-a_inj)

	ts_over_ta = (10**fitter_CI_off_int.params[0] + 10**fitter_CI_off_int.params[1]) / 10**fitter_CI_off_int.params[0]

	#print 'Fitted a_inj: ', 10**fitter_CI_off_int.params[3] * -1.
	print 'GAMMA: ', gamma

	mod_tot_flux_CI_off_int = sl.get_fluxes(np.linspace(1.e8, 11.5e9, 100), (10**fitter_CI_off_int.params[0]) * 1.e6, ts_over_ta, 10**fitter_CI_off_int.params[2], gamma, B, vol, z, delta, 'CI_off')
	
	'''
	
	# JP model, a_int fitted for
	
	fitter_JP_int = fit_leith(frequencies_int[1:], flux_int[1:], rms_int[1:], constant_params = [z, B, vol, delta, 'JP'], fitted_params = [5., 2.e8, 0.60])

	#fitter_JP_int = fit_leith(frequencies_int[[3, 4, 8, 9, 10, 11, 12]], flux_int[[3, 4, 8, 9, 10, 11, 12]], rms_int[[3, 4, 8, 9, 10, 11, 12]], constant_params = [z, B, vol, delta, 'JP'], fitted_params = [5., 2.e8, 0.60])

	gamma = 1.0 - 2.0 * (10**fitter_JP_int.params[2] * -1.)

	ts_over_ta = 1.

	print 'Fitted a_inj: ', 10**fitter_JP_int.params[2] * -1.
	print 'GAMMA: ', gamma

	mod_tot_flux_JP_int = sl.get_fluxes(np.linspace(1.e8, 11.5e9, 100), (10**fitter_JP_int.params[0]) * 1.e6, ts_over_ta, 10**fitter_JP_int.params[1], gamma, B, vol, z, delta, 'JP')
	'''
	
	#'''

	''' scipy.optimize fit
	#optimized_CI_off_int, cov = curve_fit(synfit_opt_func, np.log10(np.array(frequencies_int[1:])), np.log10(np.array(flux_int[1:])), p0=np.log10(np.array([40., 40., 1.e-12, 0.60])), sigma=np.divide(np.array(rms_int[1:]), np.multiply(np.array(flux_int[1:]), np.log(10.))), maxfev=10000)

	optimized_CI_off_int, cov = curve_fit(synfit_opt_func, np.log10(np.array(frequencies_int[[3, 4, 8, 9, 10, 11, 12]])), np.log10(np.array(flux_int[[3, 4, 8, 9, 10, 11, 12]])), p0=np.log10(np.array([10., 10., 1.e-10, 0.60])), sigma=np.divide(np.array(rms_int[[3, 4, 8, 9, 10, 11, 12]]), np.multiply(np.array(flux_int[[3, 4, 8, 9, 10, 11, 12]]), np.log(10.))), maxfev=10000)

	s_CI_off_int = Synfit(z, 10.**optimized_CI_off_int[0], 10.**optimized_CI_off_int[1], B, np.linspace(1.e8, 11.5e9, 100) , 10.**optimized_CI_off_int[3] * -1., 10.**optimized_CI_off_int[2], 'CI_off', 'JP')
	mod_tot_flux_CI_off_int = s_CI_off_int()
	print "Fitted values: ta = ",  10.**optimized_CI_off_int[0], " Myr; ti = ", 10.**optimized_CI_off_int[1], " Myr; N = ", 10.**optimized_CI_off_int[2], "a_inj = ", 10.**optimized_CI_off_int[3] * -1.
	print "Fit errors: ", 10.**np.sqrt(np.diag(cov))
	'''

	###fitter_JP_int = fit(frequencies_int[1:], flux_int[1:], rms_int[1:], 'delta', 'JP', constant_params = [z, B, a_inj], fitted_params = [80., 10., 1.e-10])
	###s_JP_int = Synfit(z, 10**fitter_JP_int.params[0], 10**fitter_JP_int.params[1], B, np.linspace(1.e8, 11.5e9, 100) , a_inj, 10**fitter_JP_int.params[2], 'delta', 'JP')
	###s_JP_int = Synfit(z, 93., 44., B, np.linspace(1.e8, 11.5e9, 100) , 0.9 * -1., 9.38e-11, 'CI_off', 'JP')
	###mod_tot_flux_JP_int = s_JP_int()

	print "RMS int: ", rms_int
	print "Flux int: ", flux_int
	print "Frequencies: ", frequencies_int

	#print "Plotted model parameters: t_off: ", 10**fitter_JP_int.params[0], " t_on: ", 10**fitter_JP_int.params[1], " injection index: ", 10**fitter_JP_int.params[3] * -1., " Scaling factor: ", 10**fitter_JP_int.params[2]
	
	#axis.scatter(np.array([112.6, 124.3, 132.1, 136.0, 159.5, 163.4, 167.3]) * 1.e6, np.array([8384.4, 8937.3, 6774.5, 6738.5, 5214.0, 4751.2, 4701.7]) * 1.e-3, marker='o', c='g', s = 22)
	'''
	axis.errorbar(frequencies_int[1:], flux_int[1:], rms_int[1:], marker=None, fmt=None, ecolor='gray', zorder=0)
	axis.scatter(frequencies_int[[4, 8, 9, 10, 11, 12]], flux_int[[4, 8, 9, 10, 11, 12]], marker='o', c='b', s = 22)
	axis.scatter(frequencies_int[[1, 2, 3, 5, 6, 7]], flux_int[[1, 2, 3, 5, 6, 7]], marker='^', c='r', s = 40)
	axis.loglog(np.linspace(1.e8, 11.5e9, 100), mod_tot_flux_CI_off_int, '--g')
	###axis.loglog(np.linspace(1.e8, 11.5e9, 100), mod_tot_flux_JP_int, '--g')
	'''
	
	axis.errorbar(frequencies_int[[3, 4, 8, 9, 10, 11, 12]], flux_int[[3, 4, 8, 9, 10, 11, 12]], rms_int[[3, 4, 8, 9, 10, 11, 12]], marker=None, fmt=None, ecolor='gray', zorder=0)
	axis.scatter(frequencies_int[[4, 8, 9, 10, 11, 12]], flux_int[[4, 8, 9, 10, 11, 12]], marker='o', c='b', s = 22)
	axis.scatter(frequencies_int[[3]], flux_int[[3]], marker='^', c='r', s = 40)
	axis.loglog(np.linspace(1.e8, 11.5e9, 100), mod_tot_flux_CI_off_int, '--g')
	###axis.loglog(np.linspace(1.e8, 11.5e9, 100), mod_tot_flux_JP_int, '--g')
	
	#'''
	
	'''
	# J1431.8+1331
	a_inj = 1.
	fitter_KGJP_int = fit(frequencies_int, flux_int, rms_int, 'CI_off', 'JP', constant_params = [z, B, a_inj], fitted_params = [25., 4., 1.e-12, 0.9])
	s_KGJP_int = Synfit(z, 10**fitter_KGJP_int.params[0], 10**fitter_KGJP_int.params[1], B, np.linspace(8.e7, 2.e9, 100) , 10**fitter_KGJP_int.params[3] * -1., 10**fitter_KGJP_int.params[2], 'CI_off', 'JP')
	mod_tot_flux_KGJP_int = s_KGJP_int()
	
	fitter_CIJP_int = fit(frequencies_int, flux_int, rms_int, 'CI', 'JP', constant_params = [z, B, a_inj], fitted_params = [25., 4., 1.e-12, 0.9])
	s_CIJP_int = Synfit(z, 10**fitter_CIJP_int.params[0], 10**fitter_CIJP_int.params[1], B, np.linspace(8.e7, 2.e9, 100) , 10**fitter_CIJP_int.params[3] * -1., 10**fitter_CIJP_int.params[2], 'CI', 'JP')
	mod_tot_flux_CIJP_int = s_CIJP_int()
	
	axis.scatter(frequencies_int, flux_int, marker='o', c='b', s = 22)
	axis.errorbar(frequencies_int, flux_int, rms_int, marker=None, fmt=None, ecolor='gray')
	axis.loglog(np.linspace(8.e7, 2.e9, 100), mod_tot_flux_KGJP_int, '-k')
	axis.loglog(np.linspace(8.e7, 2.e9, 100), mod_tot_flux_CIJP_int, '-g')
	'''
	
	xlabel(r'Frequency [Hz]', fontsize=18, fontweight='bold', color='#000000')
	ylabel(r'Flux density [Jy]', fontsize=18, fontweight='bold', color='#000000')
	
	plt.show()
	#plt.savefig('/Users/users/shulevski/Desktop/'+'4C_spectral_fits.eps', bbox_inches='tight')

def simple_int_spectrum_plot(frequencies, fluxes, flux_errors):

	fig = plt.figure()
	axis = fig.add_subplot(111)
	#axis.grid()
	#axis.set_aspect('equal')
	axis.tick_params(axis='x', labelsize=15)
	axis.tick_params(axis='y', labelsize=15)
	axis.tick_params(length=8)
	#plt.rc('text', usetex=True)
	#plt.rc('font', family='serif')

	#axis.set_xscale("log", nonposx='clip')
	#axis.set_yscale("log", nonposy='clip')

	axis.errorbar(frequencies, fluxes, yerr=flux_errors, linestyle='None', marker=None, ecolor='gray', zorder=0)
	axis.scatter(frequencies, fluxes, marker='o', c='b', s = 22)
	axis.scatter(frequencies[0], fluxes[0], marker='o', c='r', s = 22)
	axis.loglog(frequencies, fluxes, linestyle='None', marker=None)

	xlabel(r'Frequency [Hz]', fontsize=15, color='#000000')
	ylabel(r'Flux density [Jy]', fontsize=15, color='#000000')
	
	plt.show()

def plot_spix_profile():

	import pandas as pd
	from matplotlib import rc
	rc('text', usetex=True)

	fig = plt.figure()
	axis = fig.add_subplot(111)
	#axis.grid()
	#axis.set_aspect('equal')
	axis.tick_params(axis='x', labelsize=17)
	axis.tick_params(axis='y', labelsize=17)
	axis.tick_params(length=10)
	#plt.rc('text', usetex=True)
	plt.rc('font', family='serif')

	'''
	#axis.set_xscale("log", nonposx='clip')
	#axis.set_yscale("log", nonposy='clip')

	path='/home/shulevski/Documents/Research/Projects/3C236/LOFAR_Factor_images_Feb_2018/regions_new_2019/'
	profile = pd.read_csv(path + 'spix_profile_SE.txt', header=3, names=['Distance','Spix'], comment='#', delim_whitespace=True, usecols=[0,1])
	errors = pd.read_csv(path + 'spixerr_profile_SE.txt', header=3, names=['Distance','Spixerr'], comment='#', delim_whitespace=True, usecols=[0,1])

	#axis.errorbar(frequencies, fluxes, yerr=flux_errors, linestyle='None', marker=None, ecolor='gray', zorder=0)
	axis.plot(profile.get_values()[:,0] * 1.8, profile.get_values()[:,1], marker=None, c='k')
	#axis.loglog(frequencies, fluxes, linestyle='None', marker=None)

	plt.fill_between(profile.get_values()[:,0] * 1.8, profile.get_values()[:,1] - errors.get_values()[:,1], profile.get_values()[:,1] + errors.get_values()[:,1], alpha=0.8)
	#plt.xlabel(r'Distance along profile [kpc]', fontsize=18, fontweight='bold', color='#000000')
	#plt.ylabel(r'Spectral index', fontsize=18, fontweight='bold', color='#000000')
	'''

	path='/home/shulevski/Documents/Research/Projects/3C236/'
	profile = pd.read_csv(path + '_MODELDATA.dat', header=-1, names=['Source','Frequency','Flux Density'], comment='#', delim_whitespace=False, usecols=[0,1,2])
	
	print profile.loc[profile['Source']=='SE'].get('Frequency') / 1.1
	axis.plot(profile.loc[profile['Source']=='SE'].get('Frequency') / 1.1, profile.loc[profile['Source']=='SE'].get('Flux Density'), marker=None, linestyle='--', c='r')
	axis.errorbar(np.array([143.e6, 608.6e6, 1.4e9]), np.array([3.58, 1.26, 0.58]), yerr=np.array([0.1, 0.25, 0.12]), linestyle='None', marker='.', color='k', ecolor='k', zorder=0)
	axis.loglog(np.array([143.e6, 608.6e6, 1.4e9]), np.array([5.03, 1.93, 0.73]), linestyle='None', marker=None)

	plt.xlabel(r'Frequency [Hz]', fontsize=18, fontweight='bold', color='#000000')
	plt.ylabel(r'Flux Density [Jy]', fontsize=18, fontweight='bold', color='#000000')
	plt.title('SE lobe: CI model vs. observed flux density')
	plt.show()

	
# From Stroe et al. 2014, Eilek 2014
# B - current lobe mag. field, muG
# nu - observing frequency, GHz
def maximum_age(z, B, nu):
	
	B_cmb = 3.25 * (1. + z)**2. * 1.e-6
	
	print 'B_CMB = ', B_cmb, ' G'
	
	print 'B_IC = ', np.sqrt(2. / 3.) * B_cmb, ' G'
	
	print 'B_min = ', np.sqrt(1. / 3.) * B_cmb, ' G'
	
	print 't_max_tribble = ', (np.sqrt(B_cmb) / ((4./3.) * (3.**(1./4.)) * B_cmb**2.)) / 1.e6, ' Myr'
	
	print 't_max_eilek = ', 57. * (B**0.5 / nu**0.5), ' Myr'
	
def pretty_plots():

	import aplpy
	from astropy.coordinates import SkyCoord
	from astropy.coordinates import FK5
	from astropy import units as u

	#'''
	# 3C 236
	#path = '/Users/shulevski/Documents/Research/Projects/3C236/LOFAR_Factor_images_Feb_2017/dataset/re-regridding/'
	path = '/home/shulevski/Documents/Research/Projects/3C236/LOFAR_Factor_images_Feb_2018/'
	path_hires = '/home/shulevski/Documents/Research/Projects/3C236/3C236_DDFacet_Nov2018/Image_SC3_iter3/'
	
	#lofar_im = '3C236_LOFAR.fits'
	#inset_im = 'imtry1_9-MFS-image.fits'
	#lofar_im = 'imtry1_9-MFS-image.fits'
	#inset_im = '3C236_man_img_SC2-MFS-image.fits'
	#lofar_im = '3C236_man_img_SC2-MFS-image.fits'
	inset_im = '3C236_man_img_SC3-MFS-image-pb.fits'
	lofar_im = '3C236_man_img_SC4_midres_default-MFS-image-pb.fits'

	lev_im = '3C236_LOFAR_AGES.fits'
	#lev_im = 'LSC.fits'
	#lev_im = '3C236_LOFAR_48asecorig_sm.fits'
	lofar_im = 'spix.fits'
	#inset_im = 'spix.fits'

	#lofar_im = 'curv_err.fits'
	#inset_im = 'curv_err.fits'
	#'''
	
	fig = plt.figure(figsize=(11, 9))
	
	'''
	# Regions of interest

	#lofar_im = 'LSC.fits'
	#lofar_im = '3C236_LOFAR.fits'

	im = aplpy.FITSFigure(path + lofar_im, figure=fig, dimensions=[0, 1], slices=[0, 0])
	im.recenter(151.55, 34.90, radius=0.4)
	im.show_grayscale(invert=True, stretch='linear', vmin=0.001, vmax=0.1)
	im.axis_labels.set_font(size=18)
	im.tick_labels.set_font(size=16)

	factor = 20.
	sigma = 3e-3
	levno = 10
	levbase = 2.
	levels = np.power(np.sqrt(levbase), range(levno))	
	im.show_contour(path+lofar_im, levels=factor * sigma * levels, colors=['black'], linewidths=1.1)

	#im.show_rectangles(np.array([151.885108, 151.787620, 151.697969, 151.455176, 151.358325, 151.341375, 151.319955, 151.254662]), np.array([34.710602, 34.752080, 34.788003, 34.904841, 35.020029, 34.993528, 34.974135, 35.041590]), np.array([96.80, 79.56, 97.02, 92.67, 57.50, 56.80, 56.82, 93.13])/3600., np.array([76.52, 142.15, 94.78, 88.50, 79.49, 71.35, 75.56, 117.17])/3600., edgecolor='black', linewidth=2)

	#im.show_lines([np.array([[151.651464, 151.727017, 151.771241, 151.801138, 151.829530, 151.867973, 151.929169], [34.814827, 34.772158, 34.768355, 34.749319, 34.713779, 34.700485, 34.678661]]), np.array([[151.438014, 151.354995, 151.293367, 151.254679, 151.223139], [34.940664, 34.987262, 35.029079, 35.041712, 35.054354]])], edgecolor='black', linestyle='dashed', linewidth=2)

	#im.show_polygons([np.array([[151.983333333, 151.507416667, 151.421458333, 151.252875, 152.017958333], [34.724166667, 34.974416667, 34.995083333, 35.289305556, 35.278805556]]), np.array([[152.019166667, 151.86525, 151.522875, 151.183375, 151.183833333, 151.070833333, 151.075916667, 152.029958333], [34.578472222, 34.646472222, 34.848611111, 34.980055556, 35.219027778, 35.214527778, 34.506166667, 34.508444444]])], facecolor='white', zorder=2)

	#im.show_ellipses(np.array([151.483531]), np.array([35.046612]), 0.1, 0.15, facecolor='black')

	######
	'''

	'''
	im = aplpy.FITSFigure(path + lofar_im, figure=fig, dimensions=[0, 1], slices=[0, 0], subplot=[0.15, 0.1, 0.75, 0.8])
	im_1 = aplpy.FITSFigure(path_hires + inset_im, figure=fig, dimensions=[0, 1], slices=[0, 0], subplot=[0.16, 0.59, 0.35, 0.30])
	im_2 = aplpy.FITSFigure(path_hires + inset_im, figure=fig, dimensions=[0, 1], slices=[0, 0], subplot=[0.54, 0.11, 0.35, 0.30])

	im.recenter(151.55, 34.90, radius=0.3)
	#im.show_colorscale(cmap='jet', stretch='linear', vmin=-1.6, vmax=-0.4)

	im_1.recenter(151.28584, 35.02501, radius=0.065)
	#im_1.show_colorscale(cmap='jet', stretch='linear', vmin=-1.6, vmax=-0.4)
	#im_1.show_colorscale(cmap='jet', stretch='log', vmin=1e-5, vmax=1e-1)
	#im_1.show_colorscale(cmap='jet', stretch='linear', vmin=0, vmax=1)
	
	im_1.show_grayscale(invert=True, stretch='log', vmin=0.001, vmax=0.1)
	#im_1.show_grayscale(invert=True, stretch='log', vmin=0.001, vmax=0.01)
	#im_1.axis_labels.set_font(size=18)
	#im_1.tick_labels.set_font(size=16)

	im_2.recenter(151.87969, 34.700476, radius=0.065)
	#im_2.show_colorscale(cmap='jet', stretch='linear', vmin=-1.6, vmax=-0.4)
	#im_1.show_colorscale(cmap='jet', stretch='log', vmin=1e-5, vmax=1e-1)
	#im_1.show_colorscale(cmap='jet', stretch='linear', vmin=0, vmax=1)
	
	#im_2.show_grayscale(invert=False, stretch='linear', vmin=0.001, vmax=0.06)
	im_2.show_grayscale(invert=True, stretch='log', vmin=0.001, vmax=0.5)
	#im_2.axis_labels.set_font(size=18)
	#im_2.tick_labels.set_font(size=16)

	#im.show_colorscale(cmap='jet', stretch='linear', vmin=0., vmax=1.)
	
	#im.show_grayscale(invert=False, stretch='linear', vmin=-0.001, vmax=0.07)
	im.show_grayscale(invert=True, stretch='linear', vmin=0.000001, vmax=0.15)
	im.axis_labels.set_font(size=18)
	im.tick_labels.set_font(size=16)
	'''
	#'''
	# Spix maps
	im = aplpy.FITSFigure(path + lofar_im, figure=fig, dimensions=[0, 1], slices=[0, 0], subplot=[0.15, 0.1, 0.8, 0.8])
	#im_1 = aplpy.FITSFigure(path + inset_im, figure=fig, dimensions=[0, 1], slices=[0, 0], subplot=[0.23, 0.62, 0.27, 0.27])
	#im_2 = aplpy.FITSFigure(path + inset_im, figure=fig, dimensions=[0, 1], slices=[0, 0], subplot=[0.56, 0.11, 0.30, 0.25])

	im.recenter(151.55, 34.90, radius=0.32)
	im.show_colorscale(cmap='jet', stretch='linear', vmin=-1.1, vmax=-0.4)
	#im.show_colorscale(cmap='jet', stretch='linear', vmin=0, vmax=1)
	#im.show_colorscale(cmap='jet', stretch='linear', vmin=0.1, vmax=0.25)

	#im_1.recenter(151.28584, 35.02501, radius=0.07)
	#im_1.show_colorscale(cmap='jet', stretch='linear', vmin=-1.6, vmax=-0.4)
	#im_1.show_colorscale(cmap='jet', stretch='log', vmin=1e-5, vmax=1e-1)
	#im_1.show_colorscale(cmap='jet', stretch='linear', vmin=0, vmax=1)
	#im_1.show_colorscale(cmap='jet', stretch='linear', vmin=0.1, vmax=0.25)
	#im_1.axis_labels.set_font(size=18)
	#im_1.tick_labels.set_font(size=16)

	#im_2.recenter(151.87969, 34.700476, radius=0.07)
	#im_2.show_colorscale(cmap='jet', stretch='linear', vmin=-1.6, vmax=-0.4)
	#im_1.show_colorscale(cmap='jet', stretch='log', vmin=1e-5, vmax=1e-1)
	#im_2.show_colorscale(cmap='jet', stretch='linear', vmin=0, vmax=1)
	#im_2.show_colorscale(cmap='jet', stretch='linear', vmin=0.1, vmax=0.25)
	#im_2.axis_labels.set_font(size=18)
	#im_2.tick_labels.set_font(size=16)

	#im.show_colorscale(cmap='jet', stretch='linear', vmin=0., vmax=1.)
	#im.show_grayscale(invert=True, stretch='linear', vmin=-0.01, vmax=0.1)
	#im.axis_labels.set_font(size=18)
	#im.tick_labels.set_font(size=16)
	
	im.add_colorbar()
	im.colorbar.show()
	im.colorbar.set_location('right')
	im.colorbar.set_width(0.1)
	im.colorbar.set_pad(0.03)
	im.colorbar.set_axis_label_text(r'$\alpha_{143}^{609}$')
	##im.colorbar.set_axis_label_text(r'$\alpha_{143}^{609}$ - $\alpha_{609}^{1400}$ spectral curvature' )
	##im.colorbar.set_axis_label_text('Spectral curvature error')
	##im.colorbar.set_axis_label_text('Spectral index error')
	im.colorbar.set_axis_label_font(size=20)
	im.colorbar.set_axis_label_pad(20)
	im.colorbar.set_font(size=16)
	
	#im_1.add_colorbar()
	#im_1.colorbar.show(log_format=True)
	#im_1.colorbar.set_location('right')
	#im_1.colorbar.set_width(0.1)
	#im_1.colorbar.set_pad(0.03)
	#im_1.colorbar.set_axis_label_font(size=16)
	#im_1.colorbar.set_font(size=13)
	#im_1.colorbar.set_ticks([1e-2,2e-2])
	#im_1.set_title('Spectral index error')
	#im_1.set_tick_color('k')
	#'''
	'''
	im.add_scalebar(0.2)
	im.scalebar.show(0.2)  # length in degrees
	from astropy import units as u
	im.scalebar.set_length(8.721 * u.arcminute)
	im.scalebar.set_label('5 pc')
	im.scalebar.set_corner('top right')
	im.scalebar.show(8.721 * u.arcminute, label="1 Mpc", corner="top right", color='black')
	
	im.add_beam()
	im.beam.set_edgecolor('white')
	im.beam.set_facecolor('black')
	im.beam.set_hatch('/')
	##im.set_theme('publication')
	im.tick_labels.set_xformat('hh:mm:ss')
	im.tick_labels.set_yformat('dd:mm')
	im.set_tick_color('k')
	im.set_nan_color('white')
	#im.axis_labels.set_font(size=18)
	#im.tick_labels.set_font(size=16)
	
	im_1.add_beam()
	im_1.beam.set_edgecolor('white')
	im_1.beam.set_facecolor('black')
	im_1.beam.set_hatch('/')
	im_1.set_tick_color('k')
	aplpy.Ticks(im_1).hide_y()
	aplpy.TickLabels(im_1).hide_y()
	im_1.axis_labels.hide()
	im_1.tick_labels.set_xformat('hh:mm')
	im_1.set_nan_color('black')

	im_2.add_beam()
	im_2.beam.set_edgecolor('white')
	im_2.beam.set_facecolor('black')
	im_2.beam.set_hatch('/')
	im_2.set_tick_color('k')
	aplpy.Ticks(im_2).hide_y()
	aplpy.TickLabels(im_2).hide_y()
	im_2.axis_labels.hide()
	im_2.tick_labels.set_xformat('hh:mm')
	im_2.tick_labels.set_xposition('top')
	im_2.set_nan_color('black')
	'''
	
	#im.show_arrows(np.array([151.725, 151.579]), np.array([35., 34.69]), np.array([-0.3, 0.25]), np.array([0., 0.]), width=0.5, head_width=5., color='black')
	#im.show_arrows(np.array([151.579]), np.array([34.69]), np.array([0.25]), np.array([0.]), width=0.5, head_width=5., color='black')

	#im.show_ellipses(np.array([151.3083]), np.array([35.016]), np.array([0.05]), np.array([0.02]), angle=30, color='black', linestyle='--')

	#im.show_circles(np.array([151.925, 151.854, 151.821, 151.383, 151.325]), np.array([34.65, 34.75, 34.75, 34.96, 35.05]), np.array([0.015, 0.015, 0.01, 0.01, 0.01]), facecolor='white', zorder=1000)
	#im_1.show_circles(np.array([151.325]), np.array([35.05]), np.array([0.01]), facecolor='white', zorder=1000) 
	
	# Manipulate subplot axes via the figure, tick label colors not acessible via AplPy FitsFigure

	#fig.axes[1].tick_params(colors='black')
	#fig.axes[1].spines['top'].set_color('black')
	#fig.axes[1].spines['bottom'].set_color('black')
	#fig.axes[1].spines['left'].set_color('black')
	#fig.axes[1].spines['right'].set_color('black')
	#fig.axes[2].tick_params(colors='black')
	#fig.axes[2].spines['top'].set_color('black')
	#fig.axes[2].spines['bottom'].set_color('black')
	#fig.axes[2].spines['left'].set_color('black')
	#fig.axes[2].spines['right'].set_color('black')
	#'''
	#plt.setp(plt.gca().get_xticklabels(), color='red')
	#'''
	factor = 5.
	sigma = 0.6e-3
	sigma1 = 0.5e-3
	sigma_sm = 1.6e-3
	sigma_spix = 3.e-3
	sigma_ddf = 0.26e-3
	#sigma_ddf_1 = 380.e-6
	levno = 10
	levbase = 2.
	levels = np.power(np.sqrt(levbase), range(levno))
	levels = np.insert(levels, 0, -levels[0], axis=0)
	print "Levels: ", factor * sigma_spix * levels
	im.show_contour(path+lev_im, levels=factor * sigma_spix * levels, colors=['gray'], linewidths=0.5, overlap=True)
	#im_1.show_contour(path_hires+inset_im, levels=factor * sigma_ddf * levels, colors=['gray'], linewidths=0.5)
	#im_2.show_contour(path_hires+inset_im, levels=factor * sigma_ddf * levels, colors=['gray'], linewidths=0.5)
	#'''

	pos = SkyCoord(["10:05:01.451 +35:02:25.384", "10:05:18.036 +34:58:31.189", "10:05:21.152 +34:59:59.059", "10:05:25.165 +35:01:26.910", "10:05:39.715 +  34:57:23.725", "10:05:49.552 +34:54:03.678", "10:06:23.471 +34:51:07.540", "10:06:45.753 +34:47:56.557", "10:07:10.459 +34:44:55.307", "10:07:40.046 +  34:42:16.965", "10:07:38.235 +34:40:48.925"], frame=FK5, unit=(u.hourangle, u.degree))

	im.show_rectangles(np.array(pos.ra.degree), np.array(pos.dec.degree), np.array([160.072, 60.6729, 71.6399, 71.6161, 99.0416, 87.9601, 87.6856, 109.442, 92.1936, 76.3952, 87.2109])/3600., np.array([156.318, 70.6044, 70.5003, 70.5303, 98.1844, 82.1228, 104.968, 111.356, 205.588, 62.6333, 84.9923])/3600., edgecolor='black', linewidth=1)

	#im.show_rectangles(np.array([151.885108, 151.787620, 151.697969, 151.455176, 151.358325, 151.341375, 151.319955, 151.254662]), np.array([34.710602, 34.752080, 34.788003, 34.904841, 35.020029, 34.993528, 34.974135, 35.041590]), np.array([96.80, 79.56, 97.02, 92.67, 57.50, 56.80, 56.82, 93.13])/3600., np.array([76.52, 142.15, 94.78, 88.50, 79.49, 71.35, 75.56, 117.17])/3600., edgecolor='black', linewidth=1)

	lin_SE = SkyCoord(["10:06:37.85571 +34:48:30.6278", "10:06:46.85853 +34:47:08.5337", "10:06:56.46752 +34:46:41.8236", "10:07:03.97766 +34:46:41.1244", "10:07:18.04733 +34:43:30.4109", "10:07:28.82627 +34:41:52.4792", "10:07:43.51362 +34:40:58.2728"], frame=FK5, unit=(u.hourangle, u.degree))

	lin_NW = SkyCoord(["10:05:43.21060 +34:56:35.0412", "10:05:10.22578 +35:01:46.5361", "10:04:54.08773 +35:02:55.9048"], frame=FK5, unit=(u.hourangle, u.degree))

	im.show_lines([np.array([lin_NW.ra.degree, lin_NW.dec.degree]), np.array([lin_SE.ra.degree, lin_SE.dec.degree])], edgecolor='black', linestyle='dashed', linewidth=1)

	#im.show_lines([np.array([[151.438500,151.359650,151.294741,151.252644,151.221930], [34.940965,34.987429,35.028743,35.042122,35.054816]]), np.array([[151.650049,151.721693,151.771522,151.800626,151.830863,151.868466,151.930297], [34.814092,34.771757,34.768451,34.749226,34.712928,34.700573,34.678900]])], edgecolor='black', linestyle='dashed', linewidth=1)

	im.show_circles(np.array([151.925, 151.854, 151.821, 151.383, 151.325]), np.array([34.65, 34.75, 34.75, 34.96, 35.05]), np.array([0.02, 0.015, 0.01, 0.01, 0.01]), facecolor='white', zorder=1000)
	#im_1.show_circles(np.array([151.925, 151.854, 151.821, 151.383, 151.325]), np.array([34.65, 34.75, 34.75, 34.96, 35.05]), np.array([0.015, 0.015, 0.01, 0.01, 0.01]), facecolor='white', zorder=1000)
	#im_2.show_circles(np.array([151.925, 151.854, 151.821, 151.383, 151.325]), np.array([34.65, 34.75, 34.75, 34.96, 35.05]), np.array([0.015, 0.015, 0.01, 0.01, 0.01]), facecolor='white', zorder=1000)

	lab = SkyCoord(["10:07:40.36 +34:43:33.3", "10:07:35.35 +34:39:16.7", "10:07:11.86 +34:47:28.4", "10:06:47.72 +34:49:38.9", "10:06:23.05 +34:52:39.7", "10:05:49.5 +34:52:56.0", "10:05:38.05 +34:55:45.3", "10:05:26.55 +35:02:36.7", "10:05:26.57 +35:00:00.4", "10:05:17.48 +34:57:16.9", "10:05:00.73 +35:04:55.1", "10:07:39 +34:41:49.2", "10:07:37 +34:41:07.7", "10:07:37 +34:40:43.5"], frame=FK5, unit=(u.hourangle, u.degree))
	labdegra = np.array(lab.ra.degree)
	labdegdec = np.array(lab.dec.degree)

	im.add_label(labdegra[0], labdegdec[0], text='1', size='large')
	im.add_label(labdegra[1], labdegdec[1], text='2', size='large')
	im.add_label(labdegra[2], labdegdec[2], text='3', size='large')
	im.add_label(labdegra[3], labdegdec[3], text='4', size='large')
	im.add_label(labdegra[4], labdegdec[4], text='5', size='large')
	im.add_label(labdegra[5], labdegdec[5], text='6', size='large')
	im.add_label(labdegra[6], labdegdec[6], text='7', size='large')
	im.add_label(labdegra[7], labdegdec[7], text='8', size='large')
	im.add_label(labdegra[8], labdegdec[8], text='9', size='large')
	im.add_label(labdegra[9], labdegdec[9], text='10', size='large')
	im.add_label(labdegra[10], labdegdec[10], text='11', size='large')
	#im_2.add_label(labdegra[11], labdegdec[11], text='< H1', size='small')
	#im_2.add_label(labdegra[12], labdegdec[12], text='< H2', size='small')
	#im_2.add_label(labdegra[13], labdegdec[13], text='< H3', size='small')
	
	im.add_beam()
	im.tick_labels.set_xformat('hh:mm:ss')
	im.tick_labels.set_yformat('dd:mm')
	im.beam.set_edgecolor('white')
	im.beam.set_facecolor('black')
	im.beam.set_hatch('/')
	im.set_nan_color('white')
	im.axis_labels.set_font(size=18)
	im.tick_labels.set_font(size=16)
	im.set_tick_color('k')

	#'''

	'''
	# A1318 ##############################################################

	#path = '/home/shulevski/Documents/Research/Projects/Image_combiner/'
	path = '/media/shulevski/CORTEX5/A1318/'

	panstarrs = 'mosaic_r.fits'

	im = aplpy.FITSFigure(path + panstarrs, figure=fig, dimensions=[0, 1], slices=[0, 0]) 
	im.show_grayscale(vmin=None, vmid=None, vmax=None, pmin=90., pmax=99.7, stretch='linear', exponent=2, invert=True)
	#im = aplpy.FITSFigure('/home/shulevski/Desktop/P173+55-mosaic.fits', figure=fig, dimensions=[0, 1], slices=[0, 0]) 
	#im.show_grayscale(vmin=-0.0001, vmax=0.008, stretch='linear', invert=True)
	im.recenter(173.96412, 55.09193, radius=0.081)

	im.add_label(174.03341, 55.13569, 'A', color='r', size=24)
	im.add_label(173.95882, 55.10926, 'B', color='r', size=24)
	im.add_label(173.87653, 55.11302, 'C', color='r', size=24)
	im.add_label(174.00878, 55.07993, 'D', color='r', size=24)
	im.add_label(173.91691, 55.06021, 'E', color='r', size=24)
	im.add_label(173.95675, 55.08197, 'F', color='r', size=24)

	#im.add_beam()
	#im.beam.set_edgecolor('black')
	#im.beam.set_facecolor('white')
	#im.beam.set_hatch('/')
	im.set_theme('publication')

	#im = aplpy.FITSFigure(path + lofar_im, figure=fig, dimensions=[0, 1], slices=[0, 0], subplot=[0.15, 0.1, 0.75, 0.8])
	#im_1 = aplpy.FITSFigure(path + inset_im, figure=fig, dimensions=[0, 1], slices=[0, 0], subplot=[0.16, 0.59, 0.25, 0.30])
	#im_2 = aplpy.FITSFigure(path + inset_im, figure=fig, dimensions=[0, 1], slices=[0, 0], subplot=[0.54, 0.11, 0.35, 0.30])

	#im.recenter(151.55, 34.90, radius=0.3)
	#im.show_colorscale(cmap='jet', stretch='linear', vmin=-1.6, vmax=-0.4)

	#im_1.recenter(151.28584, 35.02501, radius=0.045)
	#im_1.show_colorscale(cmap='jet', stretch='linear', vmin=-1.6, vmax=-0.4)
	#im_1.show_colorscale(cmap='jet', stretch='log', vmin=1e-5, vmax=1e-1)
	#im_1.show_colorscale(cmap='jet', stretch='linear', vmin=0, vmax=1)
	#im_1.show_grayscale(invert=True, stretch='linear', vmin=-0.001, vmax=0.01)
	#im_1.axis_labels.set_font(size=18)
	#im_1.tick_labels.set_font(size=16)

	#im_2.recenter(151.87969, 34.700476, height=0.035, width=0.085)
	#im_2.show_colorscale(cmap='jet', stretch='linear', vmin=-1.6, vmax=-0.4)
	#im_1.show_colorscale(cmap='jet', stretch='log', vmin=1e-5, vmax=1e-1)
	#im_1.show_colorscale(cmap='jet', stretch='linear', vmin=0, vmax=1)
	#im_2.show_grayscale(invert=True, stretch='linear', vmin=-0.001, vmax=0.01)
	#im_2.axis_labels.set_font(size=18)
	#im_2.tick_labels.set_font(size=16)

	#im.show_colorscale(cmap='jet', stretch='linear', vmin=0., vmax=1.)
	#im.show_grayscale(invert=True, stretch='linear', vmin=-0.01, vmax=0.1)

	factor = 3.
	sigma = 1.2e-3
	sigma1 = 0.5e-3
	sigma_sm = 60.5e-6
	levno = 10
	levbase = 2.
	levels = np.power(np.sqrt(levbase), range(levno))
	levels = np.insert(levels, 0, -levels[0], axis=0)
	print levels
	#im.show_contour('/home/shulevski/Desktop/P173+55-mosaic.fits', levels=factor * sigma_sm * levels, colors=['grey'], linewidths=1.)
	im.show_contour('/media/shulevski/CORTEX5/A1318/A1318_LOFAR_crop.fits', levels=factor * sigma_sm * levels, colors=['grey'], linewidths=1.)

	######################################################################
	'''

	'''
	# Intro aging
	fig = plt.figure()
	axis = fig.add_subplot(111)
	axis.grid()
	#axis.set_aspect('equal')
	axis.tick_params(axis='x', labelsize=17)
	axis.tick_params(axis='y', labelsize=17)
	axis.tick_params(length=10)
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	
	z = 0.02
	B = 3.4e-6
	a_inj = -0.5
	sf = 1.e-27
	
	freq = np.linspace(5.e7, 5.e9, 10)
	
	s_JP_50 = Synfit(z, 50., 150., B, freq, a_inj, sf, 'delta', 'JP')
	s_JP_60 = Synfit(z, 60., 150., B, freq, a_inj, sf, 'delta', 'JP')
	#s_JP_60 = Synfit(z, 60., 150., B, freq, a_inj, 1.e-12, 'CI', 'JP')
	s_JP_80 = Synfit(z, 80., 150., B, freq, a_inj, sf, 'delta', 'JP')
	
	axis.loglog(freq, s_JP_50(), '-b', lw=3)
	axis.loglog(freq, s_JP_60(), '-g', lw=3)
	axis.loglog(freq, s_JP_80(), '-r', lw=3)
	
	plt.axvline(7.e7, color='y', linewidth=2, linestyle='-')
	plt.axvline(1.5e8, color='y', linewidth=2, linestyle='-')
	plt.axvline(2.e9, color='k', linewidth=2, linestyle='-')
	plt.axvline(4.5e9, color='k', linewidth=2, linestyle='-')
	
	xlabel(r'Frequency [Hz]', fontsize=18, fontweight='bold', color='#000000')
	ylabel(r'Flux density [arbitrary units]', fontsize=18, fontweight='bold', color='#000000')
	
	# Chapter 6 spectral indices
	
	MSSS_flux = np.array([78.5, 13.5, 10.9, 29.8, 77.2, 8.1, 32.6, 23.2, 25., 16.6, 14.6, 34., 21.8, 14.6, 34.7, 14., 11.3, 84.1, 9.6, 4.8, 7.2, 2.9, 1.4, 2., 2.4, 6.2, 3.4, 8., 1.5, 4.5, 3., 1.6, 1.3, 2.3, 0.7, 0.8, 0.6, 2.6, 0.05, 0.1]) # Jy
	NVSS_flux = np.array([16., 3., 2.4, 8.6, 22.9, 2., 4.2, 3.2, 6.5, 3.7, 2.3, 3.1, 14.9, 1.9, 7.5, 4.6, 3.7, 13.7, 2.7, 1.8, 2., 0.7, 0.4, 0.5, 0.4, 1.4, 0.6, 5.6, 2.4, 4.9, 5.4, 4.3, 5.1, 3.7, 2.3, 1.8, 4.9, 2.6, 0.04, 0.3]) # Jy
	
	#for src in range(len(MSSS_flux)):
		##print "Fluxes: ", [MSSS_flux[src], NVSS_flux[src]]
		##print "Frequencies: ", [140.e6, 1400.e6]
		#spix = np.polyfit(np.log10([140.e6, 1400.e6]), np.log10([MSSS_flux[src], NVSS_flux[src]]), 1)[0]
		#print "Spix: ", spix
	
	axis.scatter(MSSS_flux, NVSS_flux, marker='o', c='b', s = 22)
	
	xlabel(r'MSSS flux density [Jy]', fontsize=18, fontweight='bold', color='#000000')
	ylabel(r'NVSS flux density [Jy]', fontsize=18, fontweight='bold', color='#000000')
	'''
	
	plt.show()
	#plt.savefig('/home/shulevski/Desktop/'+'3C236_spix.eps', transparent=True, bbox_inches='tight')

def region_spix_calc(region_flux, map_err, freqs):

	error = [np.sqrt(np.power(map_err[0], 2.) + np.power(region_flux[0] * 0.2, 2.)), np.sqrt(np.power(map_err[1], 2.) + np.power(region_flux[1] * 0.05, 2.)), np.sqrt(np.power(map_err[2], 2.) + np.power(region_flux[2] * 0.05, 2.))]
	
	con = 1. / np.log10(freqs[0] / freqs[1])

	print "Errors: ", error
	print "Spectral index: ", con * np.log10(region_flux[0] / region_flux[1])

	print "Spectral index error: ", np.abs(1. / np.log(freqs[0] / freqs[1])) * np.sqrt(np.power(error[0] / region_flux[0],2.) + np.power(error[1] / region_flux[1], 2.))

def corrected_B_field(B_eq, gamma_min, sp_in):

	sp_in = -sp_in
	B_eq_corr = 1.18 * (gamma_min**((1. - 2. * sp_in) / (3. + sp_in))) * (B_eq**(7. / (6. + 2. * sp_in)))
	
	print 'Corrected equipartition magnetic fied: B_eq_corr = ', B_eq_corr, ' G'

if __name__=='__main__':	
	
	print 'Start.'

	path = '/Users/users/shulevski/brooks/Research_Vault/1431+1331_spix/Images_Feb_2014/'
	images = ['116-123MHz_regrid_crop.fits', '123-131MHz_regrid_crop.fits', '131-139MHz_regrid_crop.fits', '140-150MHz_regrid_crop.fits', '150-158MHz_regrid_crop.fits', '158-170MHz_regrid_crop.fits', 'J1431+1331_325MHz_smooth.fits', 'J1431+1331_610MHz_smooth.fits', 'J1431_1400MHz_smooth.fits']
	
	N_pix_beam = 66.
	N_pix_sigma = 19221.
	noise_arr = [2.8e-3, 2.2e-3, 1.7e-3, 1.5e-3, 1.3e-3, 1.e-3, 0.1e-3, 0.09e-3, 0.05e-3]
	freq_arr = [120.e6, 127.e6, 135.e6, 145.e6, 154.e6, 164.e6, 325.e6, 610.e6, 1425.e6]
	
	#result = fit_regions_age(path, 'region_data.txt', N_pix_beam, N_pix_sigma, noise_arr, freq_arr, 0.1599, 'CI_off', True)	
	
	####################### Fit space test ######################
	
	nu_arr = [131.e6, 325.e6, 610.e6, 1425.e6]
	s_arr = [6.90e-2, 1.61e-2, 6.86e-3, 1.38e-3]
	s_err = [0.014, 0.0032, 0.00035, 5.085e-05]
	
	#explore_fit_space(nu_arr, s_arr, s_err)
	#############################################################
	
	'''
	##### B2 1610+29 global spectral fit test, Murgia 2011 ######
	
	mfr = np.array([74., 151., 325., 408., 610., 1400., 1452., 1502., 2639., 4710., 4850., 8350.]) * 1.e6
	mfl = np.array([1055., 774., 412., 349., 240., 110., 101., 98.8, 49.8, 21.4, 19., 4.8]) * 1.e-3
	mfle = np.array([157., 9.5, 5.2, 35., 12.,11., 2., 2., 3., 0.3, 0.4, 0.6]) * 1.e-3
	
	#fit(mfr, mfl, mfle, 0.0318, -0.64, 3.2e-6, 'JP', True, 15., 10.)
	explore_fit_space(mfr, mfl, mfle)
	
	#############################################################
	'''
	
	############## Integrated flux fit ##########################
	frequencies = [131.e6, 325.e6, 610.e6, 1425.e6]
	flux_int = [1953.5e-3, 364.8e-3, 109.8e-3, 14.7e-3]
	flux_int_err = [0.39, 0.073, 0.0055, 0.00023]
	B = 4.25e-6 # G
	z = 0.1599
	
	#generate_model_fluxes(z, 80., 50., B, frequencies, -0.7, 2.e-26, 0., 'CI_off', True)
	
	flux_gen = [1.1862257027695642, 0.5463968470078916, 0.29062987170385302, 0.10432503728615294]
	flux_gen_err = [0.39, 0.073, 0.0055, 0.0023]
	
	#fitter = fit(frequencies, flux_gen, flux_gen_err, z, -0.7, B, 'JP', True)
	
	# Recovered parameters for t_a = 80, t_i = 50, sf = 2.e-26
	#
	# Fit input: 5., 5., 1.e-24
	# Fitted params: Source off time: 54.7699305938  [Myr], Source on time:  5.0705456171  [Myr], Scale factor:  1.91158776622e-25
	#
	# Initial params: Source off time:  81.0  [Myr], Source on time:  49.0  [Myr], Scale factor:  1e-24
	# Fitted params: Source off time:  80.0  [Myr], Source on time:  50.0  [Myr], Scale factor:  2e-26
	#
	# Initial params: Source off time:  70.0  [Myr], Source on time:  55.0  [Myr], Scale factor:  1e-24
	# Fitted params: Source off time:  80.0  [Myr], Source on time:  50.0  [Myr], Scale factor:  2e-26
	#
	# Initial params: Source off time:  30.0  [Myr], Source on time:  15.0  [Myr], Scale factor:  1e-24
	# Fitted params: Source off time:  58.512679625  [Myr], Source on time:  68.1736818804  [Myr], Scale factor:  -3.47764784424e-24

	#############################################################
	'''
	b_f = [3.59, 6.26, 5.68, 2.69]
	#br_f = [0.126, 0.186, 0.268, 0.575] # CIoff
	#toff_t = [0.2, 0.49, 0.51, 0.9] # for CIoff models
	br_f = [1., 0.493, 0.613, 0.648] # JP
	for i in range(len(b_f)):
		t = age_estimate(b_f[i], 0.1599, br_f[i])
		print 'N = ', i
		print 'T = ', t, '[Myr]'
		#print 'Toff = ', t * toff_t[i], '[Myr]'
	'''
	#freq_arr = [131.e6, 325.e6, 610.e6, 1425.e6]
	#plotModelFit(0.1599, freq_arr, -0.7, 'JP', True)
	
	#np.savetxt(path + 'Age_fit_results.txt', result)
	
	#fit_pixels(path, images, N_pix_beam, N_pix_sigma, noise_arr, freq_arr)
	
	#	Calculate the magnetic field in a plasma from equipartition assumptions (Miley, 1980)
	#	k		- ratio of the energy contained in heavy particles vs. that in electrons
	#	eta		- filling factor of the emitting regions
	#	z		- redshift
	#	th_x	- equivalent beam width or source component size in arcsec in direction x
	#	th_y	- equivalent beam width or source component size in arcsec in direction y
	#	s		- path length through the source (kpc) in the line of sight
	#	phi		- angle between the uniform magnetic field and the line of sight
	#	F_0		- flux density (Jy) or brightness (Jy / beam) of the source region at frequency nu_0
	#	nu_0	- frequency of measurement in GHz
	#	nu_1	- lower cutoff frequency in GHz
	#	nu_2	- upper cutoff frequency in GHz
	#	sp_in	- spectral index (F(nu) ~ nu^sp_in, nu_1 < nu < nu_2)
	#
	# returns the strength of the equipartition magnetic field in Gauss
	
	# B2 0924+30
	# z = 0.026141 as corrected to the Reference Frame defined by the 3K Microwave Background Radiation
	# Scale (Cosmology Corrected):     505 pc/arcsec =  0.505 kpc/arcsec
	
	#print 'The strength of the magnetic field in the integrated lobe derived from equipartition is: ', B_field_estimator(1., 1., 0.046276, 60., 60., 52.3, np.pi / 2., 0.96, 0.0616, 0.01, 100., -2.), ' G'
	
	#4C435.06 region C
	#print 'The strength of the magnetic field in the integrated lobe derived from equipartition is: ', B_field_estimator(1., 1., 0.046276, 55., 55., 48., np.pi / 2., 0.57, 1.36, 0.01, 100., -1.), ' G'
	
	#4C435.06 region A
	#print 'The strength of the magnetic field in the integrated lobe derived from equipartition is: ', B_field_estimator(1., 1., 0.046276, 45., 45., 39., np.pi / 2., 0.57, 0.061, 0.01, 100., -1.3), ' G'
	
	#print 'The break frequency of region E is: ', (1./(1. + 0.046276))*(1590.*((2.7**0.5)/((2.7**2.+(3.25*(1.+0.046276)**2.)**2.)*70.))), ' GHz'
	
	# J1431.8+1331
	#B_field_estimator_1(1., 0.1599, 4.798, 417.9 / 1.77, 0.14 * 1.e3, 10.e6, 100.e9, -1.0, 100.)
	
	#B_field_estimator(1., 1., 0.1599, 58., 35., 121, np.pi / 2., 8.5e-2, 0.61, 0.01, 100., -1.8, 700.) # NE
	
	#B_field_estimator(1., 1., 0.1599, 45., 26., 92.3, np.pi / 2., 1.9e-2, 0.61, 0.01, 100., -1.8, 700.) # SW
	
	#convert_data_synage()
	
	#read_synage_model_compute_ages()
	
	#color_shift()
	
	spec_index()
	
	#KGJP_model_LBA_HBA_GMRT_VLA()
	
	#model_regions_KGJP()
	
	#spec_tomography()
	
	#integrated_spectra_model_fit()

	#'''
	#Plot a log-log integrated spectrum. Supply: measurement frequencies, measured fluxes and flux errors in that order [Jy]

	freqs = np.array([143., 326., 609., 2695., 4750., 10550.])*1.e6
	
	#fluxes = np.array([17744., 11955.4, 7398.2, 2222.1, 1301.2, 434.3])*1.e-3
	#flux_errors = np.array([np.sqrt(np.power(376.2, 2.) + np.power(17744. * 0.2, 2.)), 128., 108.9, 73.8, 13.3, 16.0])*1.e-3
	##flux_errors = np.array([267.3, 128., 108.9, 73.8, 13.3, 16.0])*1.e-3

	fluxes = np.array([17744., 13132., 8227.7, 3652., 2353.5, 1274.7])*1.e-3
	flux_errors = np.array([np.sqrt(np.power(376.2, 2.) + np.power(17744. * 0.2, 2.)), 140., 90.8, 71.2, 41.7, 31.7])*1.e-3

	#print "Frequency [MHz]: ", freqs
	#print "Flux [mJy]: ", fluxes/1.e-3
	#print "Flux error [mJy]: ", flux_errors/1.e-3

	#simple_int_spectrum_plot(freqs, fluxes, flux_errors)
	#'''

	#'''
	# Calculates spectral index given a flux density and error
	box_1_flux = np.array([447.41, 136.54, 86.30])*1.e-3
	box_1_rms = np.array([248.36, 77.92, 47.04])*1.e-3

	box_2_flux = np.array([1338.81, 458.97, 212.38])*1.e-3
	box_2_rms = np.array([253.23, 154.80, 74.88])*1.e-3

	box_3_flux = np.array([164.24, 64.67, 13.32])*1.e-3
	box_3_rms = np.array([23.27, 9.11, 2.07])*1.e-3

	box_4_flux = np.array([40.04, 25.00, 8.20])*1.e-3
	box_4_rms = np.array([7.80, 4.87, 1.66])*1.e-3

	box_5_flux = np.array([27.40, 10.32, 10.87])*1.e-3
	box_5_rms = np.array([7.73, 2.96, 3.07])*1.e-3

	box_6_flux = np.array([35.75, 14.97, 2.12])*1.e-3
	box_6_rms = np.array([12.25, 4.91, 1.44])*1.e-3

	box_7_flux = np.array([191.0, 53.21, 8.10])*1.e-3
	box_7_rms = np.array([49.17, 13.65, 2.09])*1.e-3

	box_8_flux = np.array([80.34, 28.91, 7.12])*1.e-3
	box_8_rms = np.array([40.76, 14.12, 4.37])*1.e-3

	box_9_flux = np.array([322.15, 122.0, 49.64])*1.e-3
	box_9_rms = np.array([140.64, 53.51, 4.88])*1.e-3

	box_10_flux = np.array([99.94, 37.27, 14.49])*1.e-3
	box_10_rms = np.array([62.03, 23.14, 9.03])*1.e-3

	box_11_flux = np.array([1266.87, 541.59, 252.29])*1.e-3
	box_11_rms = np.array([138.33, 58.69, 27.92])*1.e-3

	freqs = np.array([143., 609.])*1.e6
	map_err = np.array([1.1, 0.88, 0.4])*1.e-3

	#region_spix_calc(box_11_flux, map_err, freqs)

	#'''
	
	#print "The (maximum) electron age is: ", age_estimate(0.5, 0.1005, 0.6), " [Myr]"

	#print "The break frequency is: ", break_estimate(4., 0.1005, 69.), " [GHz]"
	
	#print "The electron age for the strongest magnetic field is: ", age_estimate(10., 0.1599, 0.060), " [Myr]"
	
	#maximum_age(0.008556, 3., 0.300)

	#plot_spix_profile()
	
	#pretty_plots()

	# B20924+30
	#corrected_B_field(1.35e-6, 100, -1.2)

	#region_age_maps()

#	Calculate the magnetic field in a plasma from equipartition assumptions (Miley, 1980)
#	k		- ratio of the energy contained in heavy particles vs. that in electrons
#	eta		- filling factor of the emitting regions
#	z		- redshift
#	th_x	- equivalent beam width or source component size in arcsec in direction x
#	th_y	- equivalent beam width or source component size in arcsec in direction y
#	s		- path length through the source (kpc) in the line of sight
#	phi		- angle between the uniform magnetic field and the line of sight
#	F_0		- flux density (Jy) or brightness (Jy / beam) of the source region at frequency nu_0
#	nu_0	- frequency of measurement in GHz
#	nu_1	- lower cutoff frequency in GHz
#	nu_2	- upper cutoff frequency in GHz
#	sp_in	- spectral index (F(nu) ~ nu^sp_in, nu_1 < nu < nu_2)
#	gamma_min	- Lorentz factor of lowest energy electrons
#
# returns the strength of the equipartition magnetic field in Gauss 

#def B_field_estimator(k, eta, z, th_x, th_y, s, phi, F_0, nu_0, nu_1, nu_2, sp_in, gamma_min):

# 3C236
#B_field_estimator(1., 1., 0.1, 900., 300., 540, np.pi / 2., 5., 0.143, 0.01, 100., -0.85, 100.) # NW lobe

#B_field_estimator(1., 1., 0.1, 1340., 205., 369, np.pi / 2., 3.6, 0.143, 0.01, 100., -0.85, 100.) # SE lobe

#corrected_B_field(0.68e-6, 200., -0.75)

#print "errors LOFAR", np.sqrt(np.power(4. * 0.2, 2.) + np.power(3*1.e-3, 2.))
#print "errors WSRT", np.sqrt(np.power(1.26 * 0.2, 2.) + np.power(0.7*1.e-3, 2.))
#print "errors NVSS", np.sqrt(np.power(0.58 * 0.2, 2.) + np.power(0.4*1.e-3, 2.))