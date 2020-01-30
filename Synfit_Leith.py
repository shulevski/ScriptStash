from pylab import *
#from numpy import *
from scipy import *
from scipy import interpolate
from scipy import integrate
from scipy import special
import math
import numpy
import matplotlib.pyplot as plt
import cosmological_functions as cosmo
from Physical_constants import *
from scipy.special import gammaln




four_sigma_T_on_three_me_c = (4.0*sigma_T/(3.0*me*c))

seconds_per_year = 3.15569e7




# ===========================================================
# read in the fbar(y) data points and get the spline coefficients
# ===========================================================

def get_spline_tck():

    n_pts = 147

    ln_y_pts = zeros(n_pts)
    ln_f_pts = zeros(n_pts)
    ln_fbar_pts = zeros(n_pts)

    
    # First open the file file to read(r)
    inp = open("f_bar.dat","r")
    # read the file into a list then print
    # each item
    i = 0
    for line in inp:
        asdf = line.rstrip().split()
    
        if i>0:
            ln_y_str = asdf[0]
            ln_f_str = asdf[1]
            ln_fbar_str = asdf[2]
        
            ln_y_pts[i-1] = float(ln_y_str)
            ln_f_pts[i-1] = float(ln_f_str)
            ln_fbar_pts[i-1] = float(ln_fbar_str)
        
        i=i+1
    
    # Now close it again
    inp.close()

    spline_tck = interpolate.splrep(ln_y_pts,ln_fbar_pts,s=0)

    return spline_tck


#### for low values of y, we can use this asymptotic version of fbar(y)


def fbar_asymptotic(y):
    c1=1.808418021
    c2=-1.813799364
    c3=0.8476959474
    c4=-0.510131
    thd = 1.0/3.0
    twothd=2.0/3.0
    sevnthd = 7.0/3.0
    eightthd=8.0/3.0
    f_bar_0 = c1*(y**thd) + c2*y + c3*(y**sevnthd) + c4*y*y*y
    return f_bar_0


#### calculate the value of the synchrotron function fbar(y) using the spline interpolation

def fbar(y):
    
    if y < 1.0e-3:
        return fbar_asymptotic(y)
    elif y > 1.0e-3 and y < 20.0:
        ln_y = log(y)
        return exp(interpolate.splev(ln_y, tck, der=0))
    else:
        return 0.0



def N_CI_off(gamma, t_off, t_s_over_t_off, a, B, K_e, z, model):
    
    c_1 = four_sigma_T_on_three_me_c/(2.0*mu_0)
    
    B_CMB = 3.25e-10*(1.0+z)**2
    
    b = c_1 * (B**2 + B_CMB**2)
    
    t = t_off*t_s_over_t_off
    
    #    gamma_br = 1.0/(b*t*seconds_per_year)
    
    g_gamma = 0.0

    if model == 'CI_off':
        if gamma < 1.0/(b*t*seconds_per_year):
            g_gamma = ((1.0 - b*gamma*(t_off)*seconds_per_year)**(a-1.0)) - ((1.0 - b*gamma*t*seconds_per_year)**(a-1.0))
        elif gamma < (1.0/(b*(t_off)*seconds_per_year)):
            g_gamma = (1.0 - b*gamma*(t_off)*seconds_per_year)**(a-1.0)
    elif model == 'CI':
        if (gamma < 1. / (b * (t - t_off) * seconds_per_year)):
                g_gamma = ((t - t_off) * seconds_per_year * gamma**(-a)) / (gamma**(-a-1.0)/(b*(a-1.0)))
        if (gamma >= 1. / (b * (t - t_off) * seconds_per_year)):
            g_gamma = ((gamma**(-a - 1.)) / (b * (a - 1.))) / (gamma**(-a-1.0)/(b*(a-1.0)))
    else:
        if (gamma < 1. / (b * t_off * seconds_per_year)):
                g_gamma = ((gamma**-a) * (1.0 - b * gamma * (t_off) * seconds_per_year)**(a - 2.0)) / (gamma**(-a-1.0)/(b*(a-1.0)))
    
    N = (1.0/seconds_per_year)*g_gamma*K_e*gamma**(-a-1.0)/(b*(a-1.0))
    
    return N


#### integrand for the synchrotron integral

def synch_integrand(gamma, nu, t_off, t_s_over_t_off, a, B, K_e, z, model):
    Omega_0 = qe*B/me
    y = 4.0*math.pi*nu/(3.0*Omega_0*gamma*gamma)

    return N_CI_off(gamma, t_off, t_s_over_t_off, a, B, K_e, z, model)*fbar(y)




#### emissivity function

def synch_emissivity(nu, t_off, t_s_over_t_off, a, B, K_e, z, model):

    # Perform Integration: split the integration into two sectors, due to discontinuity at gamma_cr
    #    I_synch_low = integrate.quad(synch_integrand, gamma_0, gamma_cr, args=(nu, K_e, B, a, a_0, gamma_0, gamma_cr, gamma_b, gamma_2))[0]
    # I_synch_low = 0.0
    
    c_1 = four_sigma_T_on_three_me_c/(2.0*mu_0)
    
    B_CMB = 3.25e-10*(1.0+z)**2
    
    b = c_1 * (B**2 + B_CMB**2)
    
    gamma_min = 1.0
    
    t = t_off*t_s_over_t_off
    
    gamma_b = (1.0/(b*t*seconds_per_year))
    
    gamma_max = (1.0/(b*(t_off)*seconds_per_year))

    if model == 'JP':
        I_synch_I =integrate.quad(synch_integrand, gamma_min, gamma_b, args=(nu, t_off, t_s_over_t_off, a, B, K_e, z, model))[0]
    else:
        I_synch_I =integrate.quad(synch_integrand, gamma_min, gamma_b, args=(nu, t_off, t_s_over_t_off, a, B, K_e, z, model))[0]
    
        I_synch_II =integrate.quad(synch_integrand, gamma_b, gamma_max, args=(nu, t_off, t_s_over_t_off, a, B, K_e, z, model))[0]


    Omega_0 = qe*B/me

    if model == 'JP':
        j_nu = ((3.0**0.5)*me*c*re/(4.0*math.pi))*Omega_0*(I_synch_I)
    else:
        j_nu = ((3.0**0.5)*me*c*re/(4.0*math.pi))*Omega_0*(I_synch_I + I_synch_II)
 
    return j_nu


def synch_flux_density(nu, t_off, t_s_over_t_off, a, B, K_e, vol, z, delta, model):
    
    D_Lum = cosmo.D_Lum(z)
    
    F_nu = 1.0e26*(1.0/(D_Lum*D_Lum))*(delta**3.0)*(1.0+z)*vol*synch_emissivity(nu, t_off, t_s_over_t_off, a, B, K_e, z, model)
    
    return F_nu



###### Get the spline coefficients. This immediate code will run whenever the module is imported, and so tck is available immediately after import.

tck = get_spline_tck()

def synch_flux_density_for_fitting(nu, t_off, t_s_over_t_off, q, a, B, vol, z, delta, model):
   
    t_0 = t_off*(t_s_over_t_off - 1.0)

    if model == 'JP': # q is the number of injected particles, q = q_0 * t_i, so K_e = q0. q0 is the source in the KG paper
        K_e = q
    else:
        #K_e = q/t_0
        K_e = q

    F_nu = synch_flux_density(nu, t_off, t_s_over_t_off, a, B, K_e, vol, z, delta, model)

    return F_nu


def get_fluxes(freqs, t_off, t_s_over_t_off, q, a, B, vol, z, delta, model):

    model_fluxes = numpy.zeros(len(freqs))

    for i, freq in enumerate(freqs):
        
        model_fluxes[i] = synch_flux_density_for_fitting(freq, t_off, t_s_over_t_off, q, a, B, vol, z, delta, model)

    return model_fluxes




#### Test code for N(gamma) function: Plot an example electron distribution to check if it is working

test_N_gamma = False


if test_N_gamma == True:

    num_points = 1000

    gamma_array = numpy.logspace(3.0, 7.0, num_points)
    N_gamma_array = numpy.zeros(num_points)

    for i, gamma in enumerate(gamma_array):
        N_gamma_array[i] = N_CI_off(gamma, 6.0e7, 3.0e7, 2.01, 1.0e-10, 4.0e-4, 0.05)


    plt.plot(gamma_array, N_gamma_array, 'ro')

    plt.xscale('log')
    plt.yscale('log')

    plt.show()
