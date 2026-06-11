#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 22:05:25 2024
画图包，需结合RVAst_mcmc拟合程序使用
@author: xiaogy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import matplotlib.pyplot as pl
import sys
#import corner
#from matplotlib import rcParams, gridspec
#from matplotlib.ticker import MaxNLocator
from astropy.time import Time
#import copy
import random
import glob
import seaborn as sns
from math import sin, cos, tan, sqrt, atan2, fabs
from scipy.stats import multivariate_normal
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
#from scipy.spatial import ConvexHull
#from scipy.interpolate import interp1d
#from scipy import stats, signal
import os
from scipy.optimize import curve_fit
#from matplotlib.ticker import NullFormatter
from matplotlib.ticker import NullFormatter
import re
from astropy.io import fits
from scipy import stats, signal
from scipy.interpolate import interp1d

config = {
    "font.family":'serif', # sans-serif/serif/cursive/fantasy/monospace
    #"font.size": 14, # medium/large/small
    'font.style':'normal', # normal/italic/oblique
    'font.weight':'normal', # bold
    "mathtext.fontset":'cm',# 'cm' (Computer Modern)
    "font.serif": ['cmr10'], # 'Simsun'宋体  STIXGeneral  cmr10
    "axes.unicode_minus": False,# 用来正常显示负号
}
plt.rcParams.update(config)

pi = np.pi
auyr2kms = 4.74047

sns.set(style='ticks') #style='ticks'  darkgrid, whitegrid, dark, white, ticks

def kepler(Marr, eccarr):
    """Solve Kepler's Equation
    Args:
        Marr (array): input Mean anomaly
        eccarr (array): eccentricity
    Returns:
        array: eccentric anomaly
    """
    conv = 1.0e-12  # convergence criterion
    k = 0.85

    Earr = Marr + np.sign(np.sin(Marr)) * k * eccarr  # first guess at E
    # fiarr should go to zero when converges
    fiarr = ( Earr - eccarr * np.sin(Earr) - Marr)
    convd = np.where(np.abs(fiarr) > conv)[0]  # which indices have not converged
    nd = len(convd)  # number of unconverged elements
    count = 0

    while nd > 0:  # while unconverged elements exist
        count += 1
        M = Marr[convd]  # just the unconverged elements ...
        ecc = eccarr[convd]
        E = Earr[convd]

        fi = fiarr[convd]  # fi = E - e*np.sin(E)-M    ; should go to 0
        fip = 1 - ecc * np.cos(E)  # d/dE(fi) ;i.e.,  fi**(prime)
        fipp = ecc * np.sin(E)  # d/dE(d/dE(fi)) ;i.e.,  fi**(\prime\prime)
        fippp = 1 - fip  # d/dE(d/dE(d/dE(fi))) ;i.e.,  fi**(\prime\prime\prime)

        # first, second, and third order corrections to E
        d1 = -fi / fip
        d2 = -fi / (fip + d1 * fipp / 2.0)
        d3 = -fi / (fip + d2 * fipp / 2.0 + d2 * d2 * fippp / 6.0)
        E = E + d3
        Earr[convd] = E
        fiarr = ( Earr - eccarr * np.sin( Earr ) - Marr) # how well did we do?
        convd = np.abs(fiarr) > conv  # test for convergence
        nd = np.sum(convd is True)

    if Earr.size > 1:
        return Earr
    else:
        return Earr[0]

def true_anomaly(t, tp, per, e):
    """
    Calculate the true anomaly for a given time, period, eccentricity.

    Args:
        t (array): array of times in JD
        tp (float): time of periastron, same units as t
        per (float): orbital period in days
        e (float): eccentricity

    Returns:
        array: true anomoly at each time
    """

    # f in Murray and Dermott p. 27
    m = 2 * np.pi * (((t - tp) / per) - np.floor((t - tp) / per))
    eccarr = np.zeros(t.size) + e
    e1 = kepler(m, eccarr)
    n1 = 1.0 + e
    n2 = 1.0 - e
    nu = 2.0 * np.arctan((n1 / n2)**0.5 * np.tan(e1 / 2.0))
    return nu

def rv_calc(t, orbel):
    """RV Drive
    Args:
        t (array of floats): times of observations
        orbel (array of floats): [per, tp, e, om, K].\
            Omega is expected to be\
            in radians
        use_c_kepler_solver (bool): (default: True) If \
            True use the Kepler solver written in C, else \
            use the Python/NumPy version.
    Returns:
        rv: (array of floats): radial velocity model
    """

    # unpack array of parameters
    per, tp, e, om, k = orbel
    # Performance boost for circular orbits
    if e == 0.0:
        m = 2 * np.pi * (((t - tp) / per) - np.floor((t - tp) / per))
        return k * np.cos(m + om)

    if per < 0:
        per = 1e-4
    if e < 0:
        e = 0
    if e > 0.99:
        e = 0.99

    # Calculate the approximate eccentric anomaly, E1, via the mean anomaly  M.

    nu = true_anomaly(t, tp, per, e)
    rv = k * (np.cos(nu + om) + e * np.cos(om))

    return rv



def k2m(K,P,e,Ms,Inc=None,Niter=100,tol=1e-6): 
###If Inc is given, k2m will determine absolute mass                                                                                
###If Inc is not given, k2m will approximately give msini if m is small                                                             
    #Me2s = 3.003e-6#Earth mass in solar unit                                                                                       
    #Mj2s = 1/1048
    sinI = 1 if Inc is None else sin(Inc)
    
    K = K*0.001/4.74047#from m/s to au/yr                                                                                            
    P = P/365.25#yr                                                                                                                

    a1 = (K/sinI)**2/(4*pi**2)*(1-e**2)*P**(2/3)
    mp0 = mp = sqrt(a1*Ms**(4/3))
    
    for j in range(Niter):
        mp = sqrt(a1*(Ms+mp)**(4/3))
        if (fabs(mp-mp0)/mp0)<tol:
            break
        mp0 = mp

    return mp  # units: solar mass

def calc_eta(m1, m2, band='G', mlow_m22=0, mup_m22=99, mrl_m22=None, primary_type='single', Gmag=None): 
    ###https://iopscience.iop.org/article/10.1088/0004-6256/145/3/81/pdf
    ###Arenou et al. 2000
    ###m1: primary; m2: secondary
    ###eta=(a1-a0)/a0 where a0 and a1 are the angular semimajor axes of the photocentric and the primary orbits.
    eta = 0
    if (band=='Hp') and (m2>0.1) and (m1>0.1):
        dhp = -13.5*(np.log10(m2)-np.log10(m1))
        eta = (1+10**(0.074*dhp))/(10**(0.4*dhp)-10**(0.074*dhp))
    elif (band=='G') and (m1>mlow_m22) and (m1<mup_m22) and (m2>mlow_m22) and (m2<mup_m22):
        ###http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
        f = mrl_m22
        if primary_type=='single':
            dg = f(m2)-f(m1)
            eta = (1+m1/m2)/(10**(0.4*dg)-m1/m2)  # 1/(1+eta) = [1-m1F2/m2F1][1+F2/F1]^-1, 1-primary, 2-secondary
        elif primary_type=='binary':
            dg = Gmag-f(m2)
            eta = (1+m1/m2)/((10**(-0.4*dg)-1)-m1/m2)  # 1-binary MA+MB, 2-third body
            # print('system mag (from Gaia):',Gmag,'third-body mag:', f(m2), f(m1))
            # print('m1 & m2:',m1,m2,'eta:',eta, r'1/(1+eta):', 1/(1+eta))
            # sys.exit()
        else:
            return 0
    return eta

cc = ['c','r','g','y','b']

def plot_timeseries(ax, jd, rv, erv, offset, jitter, ins, orbel, best_fit, norbit=0):
    """
    Make a plot of the RV data and model in the current Axes.
    """
    ax.axhline(0, color='0.5', linestyle='--',lw=1.5)

    # plot orbit model
    tmin, tmax = min_max_jd(jd)#2464328.5-(2035-1995)*365.25, 2464328.5#min_max_jd(jd)#
    mplttimes = np.linspace(tmin, tmax, 10000)
    orbit_model = np.zeros_like(mplttimes,dtype=float)
    for i in range(len(orbel)):
        per, tp, e, om, k = orbel[i]
        orbit_model += rv_calc(mplttimes, [per[best_fit], tp[best_fit], e[best_fit], om[best_fit], k[best_fit]])
    ax.plot(mplttimes-2450000, orbit_model, 'k-', rasterized=False, lw=2.1, zorder=0)

    # years on upper axis
    axyrs = ax.twiny()
    xl = np.array(list(ax.get_xlim())) + 2450000
    decimalyear = Time(xl, format='jd', scale='utc').decimalyear
    axyrs.get_xaxis().get_major_formatter().set_useOffset(False)
    axyrs.set_xlim(*decimalyear)
    axyrs.set_xlabel('Year', fontsize=14)#fontweight='bold',
    axyrs.xaxis.set_minor_locator(AutoMinorLocator())
    axyrs.tick_params(direction='in', which='both',labelsize=13)
    plt.locator_params(axis='x', nbins=8)

    inds = random.sample(range(len(per)),norbit)
    for n in range(norbit):
        models = np.zeros_like(mplttimes,dtype=float)
        for i in range(len(orbel)):
            per, tp, e, om, k = orbel[i]
            models += rv_calc(mplttimes, [per[inds[n]], tp[inds[n]], e[inds[n]], om[inds[n]], k[inds[n]]])
        ax.plot(mplttimes-2450000, models, 'k-', rasterized=False, lw=1, zorder=1,alpha=0.2)
    
    # plot data
    
    if isinstance(ins, list):
        for i in range(len(ins)):
            ax.errorbar(jd[i]-2450000, rv[i]-offset[i], yerr=np.sqrt(erv[i]**2 + jitter[i]**2), 
                         fmt='o', ecolor=cc[i],capsize=3, color=cc[i],
                         capthick=1, elinewidth=1.2,ms=6,mec=cc[i], label=ins[i])
    else:
        ax.errorbar(jd-2450000, rv-offset, yerr=np.sqrt(erv**2 + jitter**2), 
                     fmt='o', ecolor=cc[i],capsize=3, 
                     capthick=1, elinewidth=1.2,ms=6,mec=cc[i], label=ins)
        
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_ylabel('RV [m/s]',fontsize=16) #weight='bold',
    ax.xaxis.grid(False)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(direction='in', which='both', left=True, right=True, bottom=True, top=False,labelsize=14)

    #ax.setp(ax.get_xticklabels(), visible=False)
    #text_name = 'New RV dataset (with DI):\nP = {} yr\ne = {}\nMc = {} $M_J$'.format(np.round(per[best_fit]/365.25,1),np.round(e[best_fit],2),np.round(mp[best_fit],2))
    #ax.text(0.5,0.25, text_name, horizontalalignment='left',verticalalignment='center', transform=ax.transAxes,fontsize=18)

    #ax.grid(False)
    ax.legend(fontsize=12,loc='upper right')#

def plot_residuals(ax, jd, rv, erv, offset, jitter, ins, orbel, best_fit, norbit=0):
    """
    Make a plot of the RV data and model in the current Axes.
    """

    ax.axhline(0, color='0.5', linestyle='--',lw=2,zorder=99)

    tmin, tmax = min_max_jd(jd)#2464328.5-(2035-1995)*365.25, 2464328.5#min_max_jd(jd)
    mplttimes = np.linspace(tmin, tmax, 1000)
    
    orbit_model = np.zeros_like(mplttimes,dtype=float)
    for i in range(len(orbel)):
        per, tp, e, om, k = orbel[i]
        orbit_model += rv_calc(mplttimes, [per[best_fit], tp[best_fit], e[best_fit], om[best_fit], k[best_fit]])

    inds = random.sample(range(len(per)),norbit)
    for n in range(norbit):
        models = np.zeros_like(mplttimes,dtype=float)
        for i in range(len(orbel)):
            per, tp, e, om, k = orbel[i]
            models += rv_calc(mplttimes, [per[inds[n]], tp[inds[n]], e[inds[n]], om[inds[n]], k[inds[n]]])
        ax.plot(mplttimes-2450000, models-orbit_model, 'k-', rasterized=False, lw=1, zorder=1,alpha=0.2)

    # plot data
    #cc = ['c','r','g','y','b']
    if isinstance(ins, list):
        for i in range(len(ins)):
            orbit_model = np.zeros_like(jd[i],dtype=float)
            for j in range(len(orbel)):
                per, tp, e, om, k = orbel[j]
                orbit_model += rv_calc(jd[i], [per[best_fit], tp[best_fit], e[best_fit], om[best_fit], k[best_fit]])
            ax.errorbar(jd[i]-2450000, (rv[i]-offset[i]-orbit_model), yerr=np.sqrt(erv[i]**2 + jitter[i]**2), 
                         fmt='o', ecolor=cc[i],capsize=3, color=cc[i],
                         capthick=1, elinewidth=1.2,ms=6,mec=cc[i], label=ins[i])
            #flat_samples = np.array([jd[i], (rv[i]-offset[i]-orbit_model), erv[i]]).T
            #df0 = pd.DataFrame(flat_samples, columns=['BJD','dRV','eRV'])
            #df0.to_csv('RV_Residuals_{}.dat'.format(ins[i]),sep=' ',mode='w',index=False)
            print(ins[i], 'RMS:', np.std(rv[i]-offset[i]-orbit_model))
    else:
        orbit_model = np.zeros_like(jd,dtype=float)
        for i in range(len(orbel)):
                per, tp, e, om, k = orbel[i]
                orbit_model += rv_calc(jd, [per[best_fit], tp[best_fit], e[best_fit], om[best_fit], k[best_fit]])
        ax.errorbar(jd-2450000, rv-offset-orbit_model, yerr=np.sqrt(erv**2 + jitter**2), 
                     fmt='o', ecolor=cc[i],capsize=3, 
                     capthick=1, elinewidth=1.2,ms=6,mec=cc[i], label=ins)

    ax.set_xlabel('JD-2450000',fontsize=16)#, weight='bold'
    ax.set_ylabel('O-C [m/s]', fontsize=16)#weight='bold',
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(direction='in', which='both', left=True, right=True, bottom=True, top=True,labelsize=14)


def adjust_figs_equal(ax):
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    dx, dy = np.abs(xmax-xmin), np.abs(ymax-ymin)
    diff = dy - dx
    if diff>0:
        xmin, xmax = xmin+diff/2, xmax-diff/2
    else:
        ymin, ymax = ymin+diff/2, ymax-diff/2
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
def circle_direction(ax, inc_deg=0, loc='lower left'):
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    dx = np.abs(xmax-xmin)
    dy = np.abs(ymax-ymin)
    diff = dy - dx
    if diff>0:
        xmin, xmax = xmin+diff/2, xmax-diff/2
    else:
        ymin, ymax = ymin+diff/2, ymax-diff/2
    dx = np.abs(xmax-xmin)
    dy = np.abs(ymax-ymin)        
    pos1, pos2 = loc.split()
    a = 0.04*dx
    b = 0.02*dy    
    if pos1=='upper':
        yc = ymax-0.28*dy
    elif pos1=='lower':
        yc = ymin+0.25*dy
    else:
        print('error location!')
        sys.exit()
        
    if pos2=='left':
        xc = xmin-0.1*dx
    elif pos2=='right':
        xc = xmax+0.1*dx
    else:
        print('error location!')
        sys.exit()        

    theta = np.arange(0,2*pi,0.01)
    xs = xc+a*np.cos(theta)
    ys = yc+b*np.sin(theta)
    ax.plot(xs,ys,'grey',alpha=0.6,lw=1)
    w, h = a, b
    if inc_deg<90:
        w, h = -w, -h
    ax.arrow(xc,yc+b,-w,0,head_width=0.3*a, head_length=0.5*b, color='grey',alpha=0.5,lw=1)
    ax.arrow(xc,yc-b,w,0,head_width=0.3*a, head_length=0.5*b, color='grey',alpha=0.5,lw=1)
    ax.arrow(xc+a,yc,0,h,head_width=0.3*a, head_length=0.5*b, color='grey',alpha=0.5,lw=1)
    ax.arrow(xc-a,yc,0,-h,head_width=0.3*a, head_length=0.5*b, color='grey',alpha=0.5,lw=1)
    #ax.set_xlim(xmin, xmax)
    #ax.set_ylim(ymin, ymax)

##contour of observed position as a function of time given proper motion uncertainty
def conPM(dra,ddec,dpmra,dpmdec,era,edec,epmra,epmdec,dt):
##https://byjus.com/jee/equation-of-tangent-to-ellipse/#:~:text=A%20line%20which%20intersects%20the%20ellipse%20at%20a,b%202%5D%20represent%20the%20tangents%20to%20the%20ellipse.
    a = np.sqrt(era**2+(epmra*dt)**2)
    b = np.sqrt(edec**2+(epmdec*dt)**2)
    m = dpmdec/dpmra
    c = np.sqrt(a**2*m**2+b**2)
    x = a**2*m/c
    y = -b**2/c
    x0 = dra+dpmra*dt
    y0 = ddec+dpmdec*dt
    x1 = x0+x
    y1 = y0+y
    x2 = x0-x
    y2 = y0-y
    return {'upper':np.array([x1,y1]).T,'lower':np.array([x2,y2]).T}

def min_max_jd(jds):
    mins, maxs = [], []
    for jd in jds:
        mins.append(np.min(jd))
        maxs.append(np.max(jd))
    return np.min(mins), np.max(maxs)

def predict_positon(MCMCtab, out, pre_epoch='2024-01-01T0:0:0', iplanet=1):
    #### calculate orbital motion of 1 planets
    pi = 3.141592653589793
    i = iplanet
    jd_ref = Time(pre_epoch, format='isot', scale='utc').jd
    Mstar = MCMCtab['Mstar'].values
    mp = MCMCtab['Mc%d'%i].values
    M0s = MCMCtab['Mo%d'%i].values
    ps = MCMCtab['Pd%d'%i].values
    MA = (M0s+2*pi*((jd_ref-out['tmin'])%ps)/ps)%(2*pi)  # Mean anomaly
    es = MCMCtab['e%d'%i].values
    E = kepler(MA, es)  # eccentric anomaly
    inc = MCMCtab['Inc%d'%i].values
    omega = MCMCtab['omega%d'%i].values
    Omega = MCMCtab['Omega%d'%i].values
    Ks = MCMCtab['K%d'%i].values
    plx = out['plx']-MCMCtab['dplx'].values  #calc_astro(pp, E, plx, Mstar, out=None, Zero_planet=False)
    
    sininc, cosinc = np.sin(inc), np.cos(inc)
    sqrt1_e2 = np.sqrt(1-es**2)
    cosOmega, sinOmega = np.cos(Omega), np.sin(Omega)
    cosomega, sinomega = np.cos(omega), np.sin(omega)
    beta0 = ps/365.25*(Ks*0.001/4.74047)*sqrt1_e2/(2*pi)/sininc# a_star: au 

    ##semi-major axis is the astrometric signature in micro-arcsec
    A = cosOmega*cosomega-sinOmega*sinomega*cosinc
    B = sinOmega*cosomega+cosOmega*sinomega*cosinc
    F = -cosOmega*sinomega-sinOmega*cosomega*cosinc
    G = -sinOmega*sinomega+cosOmega*cosomega*cosinc

    ###calculate POS
    X = np.cos(E)-es
    Y = sqrt1_e2*np.sin(E)

    beta = beta0*plx#mas
    raP = beta*(B*X+G*Y)
    decP = beta*(A*X+F*Y)

    ##    rvP.epoch = alpha*(C*Vx+H*Vy)
    eta = 0
    if not np.isnan(out['eta']):
        eta = out['eta']
    elif not np.isnan(out['flux_ratio']):
        xi = (1-Mstar/mp*out['flux_ratio'])*(1+out['flux_ratio'])**(-1)
        eta = 1/xi-1
    else:
        Gmag = out['Gmag']-10+5*np.log10(plx)-out['G_extinction']
        eta = calc_eta(Mstar, mp, mlow_m22=out['mlow_m22'],mup_m22=out['mup_m22'], mrl_m22=out['mrl_m22'], primary_type=out['primary_type'], Gmag=Gmag)  #xiao
    xi = 1/(eta+1)
    raP = raP*xi      # photocentric motion; xiao
    decP = decP*xi

    xis = mp/(Mstar+mp)
    dra_mc = -raP/xis
    ddec_mc = -decP/xis
    pa_mc = np.arctan2(dra_mc,ddec_mc)*180/pi#deg
    sep_mc = np.sqrt(dra_mc**2+ddec_mc**2)*1e-3#as
    print('sep=',np.mean(sep_mc),'+-',np.std(sep_mc),'arcsec;pa=',np.mean(pa_mc),'+-',np.std(pa_mc),'deg\n')
    return dra_mc, ddec_mc, pa_mc, sep_mc    

def nodes_periastron(par_opt, iplanet=1, plx=None, types='star', out=None):
    i = iplanet-1
    norbpar = 7
    e = par_opt[i*norbpar+2]
    K = par_opt[i*norbpar+1]
    P = par_opt[i*norbpar+0]
    #M0 = par_opt[i*norbpar+4]
    inc = par_opt[i*norbpar+6]
    omega = par_opt[i*norbpar+3]
    Omega = par_opt[i*norbpar+5]
    eccterm = np.sqrt((1 - e)/(1 + e))
    EA = [-1e-3, 0, 2*np.arctan(eccterm*np.tan((np.pi - omega)/2.)),
          2*np.arctan(eccterm*np.tan(-omega/2.))]
    EA = np.array(EA)
    plx = plx-par_opt[-4] # dplx
    
    mstar = par_opt[-1]
    
    sininc, cosinc = sin(inc), cos(inc)
    sqrt1_e2 = sqrt(1-e**2)
    cosOmega, sinOmega = cos(Omega), sin(Omega)
    cosomega, sinomega = cos(omega), sin(omega)
    beta0 = P/365.25*(K*0.001/4.74047)*sqrt1_e2/(2*pi)/sininc#au

    A = cosOmega*cosomega-sinOmega*sinomega*cosinc
    B = sinOmega*cosomega+cosOmega*sinomega*cosinc
    F = -cosOmega*sinomega-sinOmega*cosomega*cosinc
    G = -sinOmega*sinomega+cosOmega*cosomega*cosinc

    X = np.cos(EA)-e
    Y = sqrt1_e2*np.sin(EA)

    beta = beta0*plx#mas
    raP = beta*(B*X+G*Y)
    decP = beta*(A*X+F*Y)
    
    eta = 0
    if not np.isnan(out['eta']):
        eta = out['eta']
    else:
        mp = k2m(K,P,e,mstar,Inc=inc)  # in unit of solar mass
        eta = calc_eta(mstar, mp, mlow_m22=out['mlow_m22'],mup_m22=out['mup_m22'], mrl_m22=out['mrl_m22'], primary_type=out['primary_type'], Gmag=out['Gmag'])  #xiao
    xi = 1/(eta+1)
    raP = raP*xi      # photocentric motion; xiao
    decP = decP*xi
    xis = 1
    if types=='planet':
        mp = k2m(K,P,e,mstar,Inc=inc)
        xis = -mp/(mstar+mp)
    return raP/xis, decP/xis

def plot_predicted_position(tab, out, astrometry_epoch, pre_epoch='2027-01-01T0:0:0', iplanet=1, savefig=False):
    
    Np = out['nplanet']
    predict_time = pre_epoch
    #predict_time = Time(2466160.9679852305, format='jd', scale='utc').isot
    lnp = tab['logpost'].values
    best_fit = np.argmax(lnp)                # Maximum lnL index
    par_opt = tab.iloc[[best_fit]].values[0]

    names = list(tab.keys())
    npar = names.index('logpost') # number of free parameters, the order should be same as MCMC pars
    pars = np.copy(par_opt[:npar])
    
    norbpar = 7  

    for i in range(Np):  # reconstruct the free parameters
        pday = np.e**pars[i*norbpar]  # logP->P
        pars[i*norbpar] = pday
        pars[i*norbpar+1] = np.e**pars[i*norbpar+1] # logK->K
        esino, ecoso = pars[i*norbpar+2], pars[i*norbpar+3]
        ecc = esino**2 + ecoso**2
        pars[i*norbpar+2] = ecc
        pars[i*norbpar+3] = np.arctan2(esino, ecoso)
        
    inc_deg = tab['Inc%d'%iplanet].values[best_fit]*180/np.pi #par_opt['Inc%d'%iplanet].values[0]*180/np.pi
    mstar = tab['Mstar'].values[best_fit]#par_opt['Mstar'].values   # Msun
    mp = tab['Mc%d'%iplanet].values[best_fit]#par_opt['Mc%d'%iplanet].values/1048.0        # MJ -> Msun
    
    dra_mp, ddec_mp, _, _ = predict_positon(tab, out, pre_epoch=pre_epoch, iplanet=iplanet)
    dra_mp, ddec_mp = dra_mp/1000, ddec_mp/1000

    plt.figure(figsize=(7,6))
    ax = plt.gca()
    itmin = 2455000
    tsim = np.linspace(itmin,itmin+3*pday,1000)
    #tsim_yr = (tsim-2451544.5)/365.25+2000  # JD to yr
    #reflex_sim = astrometry_epoch(par_opt,tt=tsim,iplanet=iplanet)['epoch']                # pd.DataFrame
    reflex_sim = astrometry_epoch(pars,tt=tsim, out=out, iplanet=iplanet)
    xis = -mp/(mstar+mp)

    dra_sim, ddec_sim = reflex_sim[0,:]/xis, reflex_sim[1,:]/xis
    pa_sim = np.arctan2(dra_sim,ddec_sim)*180/pi#deg
    #print(np.min(pa_sim),np.max(pa_sim))
    pa_sim[pa_sim<0] += 360
    sep_sim = np.sqrt(dra_sim**2+ddec_sim**2)*1e-3#as
    
    #################### plot contour (predict location)
    ramin = stats.scoreatpercentile(dra_mp, 0)
    ramax = stats.scoreatpercentile(dra_mp, 100)
    decmin = stats.scoreatpercentile(ddec_mp, 0)
    decmax = stats.scoreatpercentile(ddec_mp, 100)
    
    diff = max(ramax - ramin, decmax - decmin)*1.7
    xmin = 0.5*(ramin + ramax) - diff/2.
    xmax = xmin + diff
    ymin = 0.5*(decmin + decmax) - diff/2.
    ymax = ymin + diff
    nbins = 500
    x = np.linspace(xmin, xmax, nbins)
    y = np.linspace(ymin, ymax, nbins)
    
    # Bin it up, then smooth it.  More points -> less smoothing.
    dens = np.histogram2d(dra_mp, ddec_mp, bins=[x, y])[0].T
    _x, _y = np.mgrid[-20:21, -20:21]
    window = np.exp(-(_x**2 + _y**2)/20.*len(dra_mp)/len(x)**2)
    dens = signal.convolve2d(dens, window, mode='same')
    dens /= np.amax(dens)
    # Make one-, two-, and three-sigma contours.
    dens_sorted = np.sort(dens.flatten())
    cdf = np.cumsum(dens_sorted)/np.sum(dens_sorted)
    cdf_func = interp1d(cdf, dens_sorted)
    
    ic = ax.imshow(dens[::-1], extent=(xmin, xmax, ymin, ymax),interpolation='nearest',aspect='auto', cmap=cm.hot_r)# 

    x = 0.5*(x[1:] + x[:-1])
    y = 0.5*(y[1:] + y[:-1])
    levels = [cdf_func(p) for p in [1 - 0.9973, 1 - 0.954, 1 - 0.683]]
    
    ccs = ['k','b','r']
    ax.contour(x, y, dens, levels=levels, colors=[ccs[0], ccs[1], ccs[2]],zorder=1,alpha=0.9)
    #text_name = 'Location on {}'.format(predict_time.split('T')[0])
    #ax.text(0.05,0.94, text_name, horizontalalignment='left',verticalalignment='center', transform=ax.transAxes,fontsize=14)
    
    # plot complete orbit (best-fit)
    ax.plot(dra_sim/1000,ddec_sim/1000,ccs[0]+'-',lw=2,zorder=2)#,label=labs[ifn]
    ax.plot(0,0,color='orange',marker='*',ms=12,zorder=99)
    
    
    # plot nodes & periastron
    dra_node, ddec_node = nodes_periastron(pars, plx=out['plx'], types='planet',iplanet=iplanet, out=out)
    dra_node, ddec_node =  dra_node/1000, ddec_node/1000

    ax.plot(dra_node[2:], ddec_node[2:],'k--',lw=1,alpha=0.7)
    ax.plot([0,dra_node[1]], [0,ddec_node[1]],'grey',lw=1.8,alpha=0.7)
              
    pname = '{}{}'.format(out['target'], chr(iplanet+97))
    print('{} prediction on {}'.format(pname, predict_time.split('T')[0]))
    
    text_name = 'Position of {}, {}'.format(pname, predict_time.split('T')[0])
    ax.text(0.07,1.04, text_name, horizontalalignment='left',verticalalignment='center', transform=ax.transAxes,fontsize=16)#,fontweight='bold'
    
    ax.invert_xaxis()
    #ax.legend(loc='upper right')
    #ax.set_aspect('equal')
    ax.set_xlabel(r'$\Delta \alpha*$ [arcsec]',fontsize=20)
    ax.set_ylabel(r'$\Delta \delta$ [arcsec]',fontsize=20)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(direction='in', which='both', left=True, right=True, bottom=True, top=True,labelsize=18)
    
    #circle_direction(ax, inc_deg=inc_deg, loc='lower left')
    print('Planet {}'.format(chr(iplanet+97)),'Inc:', inc_deg)
    if savefig:
        plt.savefig(pname+'_predicted_position.png')#,bbox_inches='tight',transparent=False)


def align_ylabels(fig, axes, pad=0.02):
    xmin = np.inf
    for axis in axes:
        xmin = min(xmin, axis.yaxis.get_ticklabel_extents(fig.canvas.get_renderer())[0].get_points()[0, 0])
    for axis in axes:
        bb = axis.get_window_extent(fig.canvas.get_renderer()).get_points()
        dx = bb[1, 0] - bb[0, 0]
        x0 = bb[0, 0]
        axis.yaxis.set_label_coords((xmin - x0)/dx - pad, 0.5)
    return

#cc = ['r','b','purple','y','g']

def plot_phase(ax, jd, rv, erv, offset, jitter, ins, orbel, best_fit, norbit=0, iplanet=0):
    
    ax.axhline(0, color='0.5', linestyle='--',lw=1.2)
    # plot best-fit orbit model
    tmin, tmax = min_max_jd(jd)
    tmin -= 75
    
    i = iplanet
    per, tp, e, om, k = orbel[i]
    bestper = per[best_fit]
    mplttimes = np.linspace(tmin, tmin+bestper, 1000)
    orbit_model = rv_calc(mplttimes, [per[best_fit], tp[best_fit], e[best_fit], om[best_fit], k[best_fit]])
    ax.plot(mplttimes-tmin, orbit_model, 'k-', rasterized=False, lw=2.2, zorder=99)
    ax.set_xlim(0,bestper)

    # plot n orbits ramdomly draw from posterior
    inds = random.sample(range(len(per)),norbit)
    for n in range(norbit):
        models = rv_calc(mplttimes, [per[inds[n]], tp[inds[n]], e[inds[n]], om[inds[n]], k[inds[n]]])
        ax.plot(mplttimes-tmin, models, 'k-', rasterized=False, lw=1, zorder=1,alpha=0.2)   

    # subtract other signal
    # plot data
    
    if isinstance(ins, list):
        for i in range(len(ins)):
            orbit_model = np.zeros_like(jd[i],dtype=float)
            for j in range(len(orbel)):
                if j==iplanet:continue
                per, tp, e, om, k = orbel[j]
                orbit_model += rv_calc(jd[i], [per[best_fit], tp[best_fit], e[best_fit], om[best_fit], k[best_fit]])
            
            ax.errorbar((jd[i]-tmin)%bestper, rv[i]-offset[i]-orbit_model, yerr=np.sqrt(erv[i]**2 + jitter[i]**2), 
                         fmt='o', ecolor='black',capsize=3, color=cc[i],
                         capthick=1, elinewidth=1.2,ms=8,mec='k', label=ins[i], zorder=999)
    else:
        orbit_model = np.zeros_like(jd,dtype=float)
        for j in range(len(orbel)):
            if j==iplanet:continue
            per, tp, e, om, k = orbel[j]
            orbit_model += rv_calc(jd, [per[best_fit], tp[best_fit], e[best_fit], om[best_fit], k[best_fit]])
        ax.errorbar((jd-tmin)%bestper, rv-offset, yerr=np.sqrt(erv**2 + jitter**2), 
                     fmt='o', ecolor='black',capsize=3, 
                     capthick=1, elinewidth=1.2,ms=8,mec='k', label=ins, zorder=999)
    
    ax.set_ylabel('RV [m/s]',fontsize=16) #weight='bold',
    ax.set_xlabel('Phase [days]',fontsize=16) #weight='bold',
    ax.xaxis.grid(False)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(direction='in', which='both', left=True, right=True, bottom=True, top=True,labelsize=14)    
    plt.setp(ax.spines.values(), linewidth=1.8)

def plot_RV_phase(tab, out, norbit=0, savefig=False):
    
    jds, rvs, ervs, ins = out['rv_jds'], out['rv_data'], out['rv_err'], out['rv_ins']
    offsets, jitters = [], []
    
    lnp = tab['logpost'].values
    best_fit = np.argmax(lnp)                # Maximum lnL index
    #par_opt = tab.iloc[[best_fit]]
    nplanet = out['nplanet']
    
    orbel = []
    for i in range(nplanet):
        per = tab['Pd%d'%(i+1)].values           # day
        try:
            k1 = tab['K%d'%(i+1)].values              # m/s
        except KeyError:
            k1 = np.exp(tab['logK%d'%(i+1)].values)
        e1 = tab['e%d'%(i+1)].values
        arg1 = tab['omega%d'%(i+1)].values        # rad
        tp1 = tab['Tp%d'%(i+1)].values        # day
        orbel.append([per, tp1, e1, arg1, k1])
    
    # calculate RV offsets for different instruments if they were marginalized during fitting 
    for n, tel in enumerate(ins):   
        J = tab['J_'+tel].values[best_fit]
        try:
            b = tab['b_'+tel].values[best_fit]
        except KeyError:
            jd, rv, erv = jds[n], rvs[n], ervs[n]
            model_rv = np.zeros_like(jd, dtype=float)
            for ip in range(nplanet):
                per, tp, e, om, k = orbel[ip]
                model_rv += rv_calc(jd, [per[best_fit], tp[best_fit], e[best_fit], om[best_fit], k[best_fit]])
            jit = J
            ivar = 1./(erv**2+jit**2)  # adopt from orvara, marginalize rv offsets
            dRv = rv - model_rv        
            A = np.sum(ivar)
            B = np.sum(2*dRv*ivar)   # gamma = -B/2/A
            b = B/2/A
        offsets.append(b)
        jitters.append(J)

    #plot_timeseries(ax1, jds, rvs, ervs, offsets, jitters, ins, orbel, best_fit, norbit=norbit)
    #plot_residuals(ax2, jds, rvs, ervs, offsets, jitters, ins, orbel, best_fit, norbit=norbit)
    
    fig, axs = plt.subplots(nplanet, 1, figsize=(6, nplanet*4))
    if nplanet==1:
        plot_phase(axs, jds, rvs, ervs, offsets, jitters, ins, orbel, best_fit, norbit=norbit, iplanet=0)
    else:
        fig.subplots_adjust(hspace=0.20, wspace=0.50)
        for i, iax in enumerate(axs):
            plot_phase(iax, jds, rvs, ervs, offsets, jitters, ins, orbel, best_fit, norbit=norbit, iplanet=i)
    if savefig:
        target = out['target']
        plt.savefig(target+'_RV_phase.png')#,bbox_inches='tight',transparent=False)


def plot_RV_OC(tab, out, ax1=None, ax2=None, norbit=0, savefig=False):
    '''
    Parameters
    ----------
    ax1 : axix for RV fitting.
    ax2 : axis for RV residual (O-C).
    tab : mcmc posteriors. type: DataFrame.
    out : data dict. Containing RV epoch data and n planet.
    norbit : number of orbits. The default is 0.
    '''
    if ax1 is None or ax2 is None:
        fig = plt.figure(figsize=(6.5, 5.8),dpi=120)
        ax1 = fig.add_axes((0.16, 0.33, 0.78, 0.55))
        ax2 = fig.add_axes((0.16, 0.1, 0.78, 0.22),sharex=ax1)
    jds, rvs, ervs, ins = out['rv_jds'], out['rv_data'], out['rv_err'], out['rv_ins']
    offsets, jitters = [], []
    
    lnp = tab['logpost'].values
    best_fit = np.argmax(lnp)                # Maximum lnL index
    #par_opt = tab.iloc[[best_fit]]
    nplanet = out['nplanet']
    
    orbel = []
    for i in range(nplanet):
        per = tab['Pd%d'%(i+1)].values           # day
        try:
            k1 = tab['K%d'%(i+1)].values              # m/s
        except KeyError:
            k1 = np.exp(tab['logK%d'%(i+1)].values)
        e1 = tab['e%d'%(i+1)].values
        arg1 = tab['omega%d'%(i+1)].values        # rad
        tp1 = tab['Tp%d'%(i+1)].values        # day
        orbel.append([per, tp1, e1, arg1, k1])
    
    # calculate RV offsets for different instruments if they were marginalized during fitting 
    for n, tel in enumerate(ins):   
        J = tab['J_'+tel].values[best_fit]
        try:
            b = tab['b_'+tel].values[best_fit]
        except KeyError:
            jd, rv, erv = jds[n], rvs[n], ervs[n]
            model_rv = np.zeros_like(jd, dtype=float)
            for ip in range(nplanet):
                per, tp, e, om, k = orbel[ip]
                model_rv += rv_calc(jd, [per[best_fit], tp[best_fit], e[best_fit], om[best_fit], k[best_fit]])
            jit = J
            ivar = 1./(erv**2+jit**2)  # adopt from orvara, marginalize rv offsets
            dRv = rv - model_rv        
            A = np.sum(ivar)
            B = np.sum(2*dRv*ivar)   # gamma = -B/2/A
            b = B/2/A
        offsets.append(b)
        jitters.append(J)

    plot_timeseries(ax1, jds, rvs, ervs, offsets, jitters, ins, orbel, best_fit, norbit=norbit)
    plot_residuals(ax2, jds, rvs, ervs, offsets, jitters, ins, orbel, best_fit, norbit=norbit)
    if savefig:
        target = out['target']
        plt.savefig(target+'_RV_OC.png')#,bbox_inches='tight',transparent=False)

def plot_GOST_fit(ax1, tab, out, astrometry_epoch, astrometry_kepler, iplanet=None, use_starfn=None, show=True):
    
    Np = out['nplanet']
    if use_starfn is not None:
        if os.path.exists(use_starfn):
            print('\nload start file:',use_starfn)
            init, pname = [], []
            f = open(use_starfn)
            for line in f:
                if line[0]=='#':continue
                lst = line.strip().split()
                pname.append(lst[0])
                init.append(float(lst[1]))
            f.close()
            pars = np.asarray(init)
        else:
            print('\nstar file not found:',use_starfn)
            sys.exit()
    else:
        lnp = tab['logpost'].values
        best_fit = np.argmax(lnp)                # Maximum lnL index
        par_opt = tab.iloc[[best_fit]].values[0]
    
        names = list(tab.keys())
        npar = names.index('logpost') # number of free parameters, the order should be same as MCMC pars
        pars = np.copy(par_opt[:npar])
    
    norbpar = 7  

    for i in range(Np):  # reconstruct the free parameters
        
        pday = np.e**pars[i*norbpar]  # logP->P
        pars[i*norbpar] = pday

        pars[i*norbpar+1] = np.e**pars[i*norbpar+1] # logK->K
        esino, ecoso = pars[i*norbpar+2], pars[i*norbpar+3]
        ecc = esino**2 + ecoso**2
        pars[i*norbpar+2] = ecc
        pars[i*norbpar+3] = np.arctan2(esino, ecoso)
    
    reflex_gost = astrometry_epoch(pars,tt=out['gost']['BJD'], out=out, iplanet=iplanet)   # reflex motion for gost epoch
    
    kep = astrometry_kepler(pars, out=out, iplanet=iplanet)
    
    #colnames = ['ra','dec','parallax','pmra','pmdec','radial_velocity']
    r1 = kep['barycenter'][out['astro_index']][:,0] 
    r2 = kep['barycenter'][out['astro_index']][:,1]
    r3 = kep['barycenter'][out['astro_index']][:,3]
    r4 = kep['barycenter'][out['astro_index']][:,4]
    r5 = kep['barycenter'][out['astro_index']][:,2]
    
    ###observed position and proper motion relative to the predicted barycentric position and proper motion
    #out['astrometry'].iloc[out['astro_index']]['ra']-kep['barycenter'].iloc[out['astro_index']]['ra']
    
    dra_obs = (out['astrometry'].iloc[out['astro_index']]['ra']-r1)*np.cos(out['astrometry'].iloc[out['astro_index']]['dec']/180*pi)*3.6e6  #deg2mas
    ddec_obs = (out['astrometry'].iloc[out['astro_index']]['dec']-r2)*3.6e6
    dpmra_obs = out['astrometry'].iloc[out['astro_index']]['pmra']-r3
    dpmdec_obs = out['astrometry'].iloc[out['astro_index']]['pmdec']-r4
    dplx_obs = out['astrometry'].iloc[out['astro_index']]['parallax']-r5
    era_obs = out['astrometry'].iloc[out['astro_index']]['ra_error']
    edec_obs = out['astrometry'].iloc[out['astro_index']]['dec_error']
    eplx_obs = out['astrometry'].iloc[out['astro_index']]['parallax_error']
    epmra_obs = out['astrometry'].iloc[out['astro_index']]['pmra_error']
    epmdec_obs = out['astrometry'].iloc[out['astro_index']]['pmdec_error']
    
    #plt.plot(dra_obs, ddec_obs,'ko')
    
    ###the position and proper motion reconstructed by a linear fit to gost-based observations generated by accounting for reflex motion
    dra_model = np.array(dra_obs-kep['cats'][:,0])   # <class 'pandas.core.series.Series'> -> np.array
    ddec_model = np.array(ddec_obs-kep['cats'][:,1])
    dplx_model = np.array(dplx_obs-kep['cats'][:,2])
    dpmra_model = np.array(dpmra_obs-kep['cats'][:,3])
    dpmdec_model = np.array(dpmdec_obs-kep['cats'][:,4])
    
    tbin = out['gost']['BJD'].values
    tmin = np.min(tbin)
    tsim = np.linspace(tmin-0.50*pday,tmin+0.50*pday,5000)
    reflex_sim = astrometry_epoch(pars, tt=tsim, out=out, iplanet=iplanet) # reflex motion for simulated epoch
    
    dra_sim, ddec_sim = reflex_sim[0,:], reflex_sim[1,:]
    
    ## plot dr23 simulated data (from gost sampling)
    normalize = mcolors.Normalize(vmin=0, vmax=1)
    colormap = getattr(cm, 'plasma')#'YlOrRd')
    tbin = out['gost']['BJD'].values
    print('Gost Baseline:',np.max(tbin-np.min(tbin)))
    phase = (tbin-np.min(tbin))%pday/np.max(tbin-np.min(tbin))
    dra_bin,ddec_bin = reflex_gost[0,:], reflex_gost[1,:]
    ax1.scatter(dra_bin,ddec_bin, s=60, marker='o',c=colormap(normalize(phase)), edgecolor='w',alpha=0.99,lw=0.3,zorder=3)
    
    # plot complete orbit
    ax1.plot(dra_sim, ddec_sim, 'k-',lw=2,zorder=2)#,label=labs[0]
    ax1.plot(0,0,color='k',marker='+',ms=8,zorder=99)
    
    cc = ['green','b']
    oc = ['green','b']
    for k in range(len(out['astro_index'])):
        m = out['gost']['BJD']<out['GDR{}_baseline'.format(k+2)]   # dr2_base_line
        ref = out['GDR{}_refep'.format(k+2)]
        t0, t1 = np.min(out['gost']['BJD'][m])-ref, np.max(out['gost']['BJD'][m])-ref
        ttsim = np.linspace(t0,t1,100)/365.25    
        
        con = conPM(dra_obs.values[k],ddec_obs.values[k],dpmra_obs.values[k],dpmdec_obs.values[k],era_obs.values[k],edec_obs.values[k],epmra_obs.values[k],epmdec_obs.values[k],ttsim)
        
        xl, xu = con['lower'][:,0], con['upper'][:,0]
        yl, yu = con['lower'][:,1], con['upper'][:,1]
    
        xx, yy = np.concatenate([xl, xu[::-1]]), np.concatenate([yl, yu[::-1]])
        ax1.fill(xx,yy,alpha=0.5,color=cc[k],edgecolor='w',zorder=99)
        
        # fitted DR2 & DR3
        ax1.plot(dra_model[k], ddec_model[k], color=cc[k],marker='o',ms=8.6,zorder=99)
        dras_model = dra_model[k]+dpmra_model[k]*ttsim
        ddecs_model = ddec_model[k]+dpmdec_model[k]*ttsim
        # error lines
        ax1.plot(dras_model, ddecs_model, color=cc[k], zorder=98,lw=2.5)
    
    if Np==1:
        # plot nodes & periastron
        dra_node, ddec_node = nodes_periastron(pars, plx=out['plx'], types='star', out=out)
        ax1.plot(dra_node[2:], ddec_node[2:],'k--',lw=1.2,alpha=0.8,zorder=0)
        ax1.plot([0,dra_node[1]], [0,ddec_node[1]],'grey',lw=1.8,alpha=0.8,zorder=0)
    
    #ax1.set_xlim(1.1,-3.1)
    #ax1.set_ylim(-0.4,0.4)
        
    #circle_direction(ax2, inc_deg=inc_deg, loc='lower left')
    
    if show:
        l1 = ax1.legend(handles=[plt.Line2D([0], [0], color=cc[0], lw=8, label='Catalog GDR2',alpha=0.5),
                            plt.Line2D([0], [0], color=cc[0], marker='o', ms=8,label='Fitted    GDR2'),
                              plt.Line2D([0], [0], color=cc[1], lw=8, label='Catalog GDR3',alpha=0.5),   
                        plt.Line2D([0], [0], color=cc[1], marker='o', ms=8,label='Fitted    GDR3')],
                bbox_to_anchor=(0.51,0.933),loc='center',ncol=2,fontsize=13,frameon=False,labelspacing=0.3)
          
        ax1.legend(handles=[plt.scatter([], [], color=colormap(0.01), s=30, label='0'),
                        plt.scatter([], [], color=colormap(0.2), s=30,label='0.2'),
                        plt.scatter([], [], color=colormap(0.4), s=30, label='0.4'),
                    plt.scatter([], [], color=colormap(0.6), s=30,label='0.6'),
                    plt.scatter([], [], color=colormap(0.8), s=30,label='0.8'),
                    plt.scatter([], [], color=colormap(0.99), s=30,label='1'),],
            bbox_to_anchor=(0.49,1.03),loc='center',ncol=6,fontsize=13,frameon=False,columnspacing=0.00) 
        
        ax1.add_artist(l1)
        ax1.text(0.5,1.075, 'Time', horizontalalignment='center',verticalalignment='center', transform=ax1.transAxes,fontsize=14)
    
    ax1.set_xlabel(r'$\Delta \alpha_\ast$ [mas]',fontsize=16)
    ax1.set_ylabel(r'$\Delta \delta$ [mas]',fontsize=16)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(direction='in', which='both', left=True, right=True, bottom=True, top=True,labelsize=14)
    ax1.invert_xaxis()

def plot_GDR4_fit(ax1, tab, out, astrometry_epoch, loglikelihood, types='2D'):  # types: 2D, 1D, OC
    
    Np = out['nplanet']
    lnp = tab['logpost'].values
    best_fit = np.argmax(lnp)                # Maximum lnL index
    par_opt = tab.iloc[[best_fit]].values[0] # best-fit pars
    
    names = list(tab.keys())
    npar = names.index('logpost') # number of free parameters, the order should be same as MCMC pars
    pars = np.copy(par_opt[:npar])
    norbpar = 7  # indicates the number of basic keplerian pars (p k e omega M0 Omega inc)

    for i in range(Np):  # reconstruct the free parameters
        pday = np.e**pars[i*norbpar]  # logP->P
        pars[i*norbpar] = pday
        pars[i*norbpar+1] = np.e**pars[i*norbpar+1] # logK->K
        esino, ecoso = pars[i*norbpar+2], pars[i*norbpar+3]
        ecc = esino**2 + ecoso**2
        pars[i*norbpar+2] = ecc
        pars[i*norbpar+3] = np.arctan2(esino, ecoso)
        
    if types=='2D':
        ax1.set_xlabel(r'$\Delta \alpha_\ast$ [mas]',fontsize=18)
        ax1.set_ylabel(r'$\Delta \delta$ [mas]',fontsize=18)
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax1.tick_params(direction='in', which='both', left=True, right=True, bottom=True, top=True,labelsize=14)
        ax1.invert_xaxis()
    elif types=='OC':
        ax3 = ax1
        ax3.set_ylabel('O-C [mas]',fontsize=18)
        ax3.set_xlabel('Epoch [year]',fontsize=18)
        ax3.xaxis.set_minor_locator(AutoMinorLocator())
        ax3.yaxis.set_minor_locator(AutoMinorLocator())
        ax3.tick_params(direction='in', which='both', left=True, right=True, bottom=True, top=True,labelsize=14)
        ax3.xaxis.set_major_locator(MultipleLocator(1))
    else:
        print('Wrong types!')
        sys.exit()
    
    ############################# plot Gaia abscissa fitting ###################
    # out['GDR4_abs']:
    # Time_BJD, np.sin(Scan_angle_rad), np.cos(Scan_angle_rad), Parallax_factor,AL_posistion_mas, AL_error
    tepoch = ts = out['GDR4_abs'][:,0]
    cpsi = out['GDR4_abs'][:,2]
    spsi = out['GDR4_abs'][:,1]
    absci = out['GDR4_abs'][:,4]  # original AL_abs
    print('RMS of GDR4 IAD:',np.std(absci))
    jit = pars[names.index('J_gaia')] if 'J_gaia' in names else 0
    eabsci = np.sqrt(out['GDR4_abs'][:,5]**2+jit**2)
    
    epoch = astrometry_epoch(pars,tt=tepoch, out=out) #reflex motion
    pred_res = loglikelihood(pars,prediction=True,verbose=True,out=out)
    
    absres = pred_res['res']['GDR4']
    print('RMS of fitted GDR4 IAD:',np.std(absres))
    dra = pred_res['res']['GDR4']*spsi + epoch[0,:]
    ddec = pred_res['res']['GDR4']*cpsi + epoch[1,:]

    ts_yr = (ts-2451544.5)/365.25+2000
    ts1 = np.array([ts[0]] + list(ts))
    dt = np.diff(ts1)
    index = []
    ii = 0
    for j in range(len(dt)):
        if(dt[j]>=0.01):
            ii = ii+1
        index.append(ii)
    index = np.array(index)
    
    # bin data per day
    w = 1/eabsci**2
    ts_yrbin = np.array([np.sum(ts_yr[index==j]*w[index==j])/np.sum(w[index==j]) for j in np.unique(index)])
    tbin = np.array([np.sum(ts[index==j]*w[index==j])/np.sum(w[index==j]) for j in np.unique(index)])
    dra_bin = np.array([np.sum(dra[index==j]*w[index==j])/np.sum(w[index==j]) for j in np.unique(index)])
    ddec_bin = np.array([np.sum(ddec[index==j]*w[index==j])/np.sum(w[index==j]) for j in np.unique(index)])
    eabsci_bin = np.array([np.std(absci[index==j]) for j in np.unique(index)])
    psi = np.arctan2(spsi,cpsi)
    psi_bin = np.array([np.sum(psi[index==j]*w[index==j])/np.sum(w[index==j]) for j in np.unique(index)])
    absres_bin = np.array([np.sum(absres[index==j]*w[index==j])/np.sum(w[index==j]) for j in np.unique(index)])
    
    # delete element if bin size = 1
    nn = np.array([np.sum(index==j) for j in np.unique(index)])
    ii = nn==1
    if(np.sum(ii)>0):
        tbin = tbin[~ii]
        dra_bin = dra_bin[~ii]
        ddec_bin = ddec_bin[~ii]
        eabsci_bin = eabsci_bin[~ii]
        psi_bin = psi_bin[~ii]
        ts_yrbin = ts_yrbin[~ii]
        absres_bin = absres_bin[~ii]
    
    cpsi_bin = np.cos(psi_bin)
    spsi_bin = np.sin(psi_bin)
    era_bin = eabsci_bin*spsi_bin
    edec_bin = eabsci_bin*cpsi_bin
    
    ################## plot gaia IAD residual #################
    # plot un- and binned points, errorbar
    normalize = mcolors.Normalize(vmin=0, vmax=1)
    colormap = getattr(cm, 'plasma')#'YlOrRd')
    print('GDR4 IAD tmin:',np.min(tbin),np.max(tbin-np.min(tbin)))    
    phase = (tbin-np.min(tbin))%pday/np.max(tbin-np.min(tbin))
            
    if types=='OC':
        ax3.errorbar(ts_yr,absres,yerr=eabsci, fmt='ko', capsize=0, capthick=0, elinewidth=0, ms=4.5, zorder=0, alpha=0.15)
        ax3.scatter(ts_yrbin,absres_bin,s=60, marker='o',c=colormap(normalize(phase)), edgecolor='w',alpha=0.99,lw=1,zorder=1)
        ax3.axhline(0,ls='--',lw=1,color='grey',zorder=0)
    if types=='2D':
        
        tmin = np.min(tepoch)
        tsim = np.linspace(tmin-0.50*pday,tmin+0.50*pday,5000)
        reflex_sim = astrometry_epoch(pars, tt=tsim, out=out) # reflex motion for simulated epoch
        dra_sim, ddec_sim = reflex_sim[0,:], reflex_sim[1,:]
        
        # plot complete orbit
        ax1.plot(dra_sim, ddec_sim, 'k-',lw=2,zorder=0)#,label=labs[0]
        ax1.plot(0,0,color='k',marker='+',ms=8,zorder=99)
        # plot data
        ax1.scatter(dra,ddec, s=30, marker='o',facecolor='k',edgecolor='w',alpha=0.25,lw=0.1,zorder=0)
        ax1.scatter(dra_bin,ddec_bin, s=60, marker='o',c=colormap(normalize(phase)), edgecolor='w',alpha=0.99,lw=1,zorder=1)

        if Np==1:
            # plot nodes & periastron
            dra_node, ddec_node = nodes_periastron(pars, plx=out['plx'], types='star', out=out)
            ax1.plot(dra_node[2:], ddec_node[2:],'k--',lw=1.2,alpha=0.8,zorder=0)
            ax1.plot([0,dra_node[1]], [0,ddec_node[1]],'grey',lw=1.8,alpha=0.8,zorder=0)
    
    x0, y0 = dra_bin-era_bin, ddec_bin-edec_bin
    x1, y1 = dra_bin+era_bin, ddec_bin+edec_bin
    y2, y3 = absres_bin-eabsci_bin, absres_bin+eabsci_bin
    
    # plot error bar
    for ii in range(len(dra_bin)):
        if types=='2D':
            ax1.plot([x0[ii], x1[ii]],[y0[ii], y1[ii]],color=colormap(normalize(phase[ii])),lw=1, zorder=1,)
        elif types=='OC':
            ax3.plot([ts_yrbin[ii], ts_yrbin[ii]],[y2[ii], y3[ii]],color=colormap(normalize(phase[ii])),lw=1, zorder=1,)


def plot_hip_fit(ax1, tab, out, astrometry_epoch, astrometry_kepler, loglikelihood, types='2D'):  # types: 2D, 1D, OC
    
    Np = out['nplanet']
    lnp = tab['logpost'].values
    best_fit = np.argmax(lnp)                # Maximum lnL index
    par_opt = tab.iloc[[best_fit]].values[0] # best-fit pars
    
    names = list(tab.keys())
    npar = names.index('logpost') # number of free parameters, the order should be same as MCMC pars
    pars = np.copy(par_opt[:npar])
    norbpar = 7  # indicates the number of basic keplerian pars (p k e omega M0 Omega inc)

    for i in range(Np):  # reconstruct the free parameters
        pday = np.e**pars[i*norbpar]  # logP->P
        pars[i*norbpar] = pday

        pars[i*norbpar+1] = np.e**pars[i*norbpar+1] # logK->K
        esino, ecoso = pars[i*norbpar+2], pars[i*norbpar+3]
        ecc = esino**2 + ecoso**2
        pars[i*norbpar+2] = ecc
        pars[i*norbpar+3] = np.arctan2(esino, ecoso)
        
    if types=='2D':
        pass
        # ax1.set_xlabel(r'$\Delta \alpha_\ast$ [mas]',fontsize=18)
        # ax1.set_ylabel(r'$\Delta \delta$ [mas]',fontsize=18)
        # ax1.xaxis.set_minor_locator(AutoMinorLocator())
        # ax1.yaxis.set_minor_locator(AutoMinorLocator())
        # ax1.tick_params(direction='in', which='both', left=True, right=True, bottom=True, top=True,labelsize=14)
        # ax1.invert_xaxis()
    elif types=='OC':
        ax3 = ax1
        ax3.set_ylabel('O-C [mas]',fontsize=18)
        ax3.set_xlabel('Epoch [year]',fontsize=18)
        ax3.xaxis.set_minor_locator(AutoMinorLocator())
        ax3.yaxis.set_minor_locator(AutoMinorLocator())
        ax3.tick_params(direction='in', which='both', left=True, right=True, bottom=True, top=True,labelsize=14)
        ax3.xaxis.set_major_locator(MultipleLocator(1))
    elif types=='1D':
        ra_ax, dec_ax = ax1
        ra_ax.set_ylabel(r'$\Delta \alpha_\ast$ [mas]',fontsize=18)
        dec_ax.set_ylabel(r'$\Delta \delta$ [mas]',fontsize=18)        
        dec_ax.set_xlabel('Epoch [year]',fontsize=18)
        for iax in ax1:
            iax.xaxis.set_minor_locator(AutoMinorLocator())
            iax.yaxis.set_minor_locator(AutoMinorLocator())
            iax.tick_params(direction='in', which='both', left=True, right=True, bottom=True, top=True,labelsize=14)
    else:
        print('Wrong types!')
        sys.exit()
    
    ############################# plot hip2 abscissa fitting ###################
    tepoch = ts = out['data_epoch']['BJD'].values
    cpsi = out['data_epoch']['CPSI'].values
    spsi = out['data_epoch']['SPSI'].values
    absci = out['data_epoch']['RES'].values
    print('RMS of Hip IAD:',np.std(absci))
    eabsci = out['data_epoch']['SRES'].values    
    tmin, tmax = np.min(tepoch)-4, np.max(tepoch)+4
    tsim = np.linspace(tmin,tmax,1000)  # day
    tsim_yr = (tsim-2451544.5)/365.25+2000
    
    epoch = astrometry_epoch(pars,tt=tepoch, out=out) #reflex motion
    epoch_sim = astrometry_epoch(pars,tt=tsim, out=out) #reflex motion
    pred_res = loglikelihood(pars,prediction=True,verbose=True,out=out)
    
    dra0 = absci*cpsi
    era  = eabsci*cpsi
    ddec0 = absci*spsi
    edec = eabsci*spsi
    absres = pred_res['res']['epoch_hip2']
    print('RMS of fitted Hip IAD:',np.std(absres))
    dra = pred_res['res']['epoch_hip2']*cpsi + epoch[0,:]
    ddec = pred_res['res']['epoch_hip2']*spsi + epoch[1,:]

    ts_yr = (ts-2451544.5)/365.25+2000
    ts1 = np.array([ts[0]] + list(ts))
    dt = np.diff(ts1)
    index = []
    ii = 0
    for j in range(len(dt)):
        if(dt[j]>=0.1):
            ii = ii+1
        index.append(ii)
    index = np.array(index)
    
    # bin data per day
    w = 1/eabsci**2
    ts_yrbin = np.array([np.sum(ts_yr[index==j]*w[index==j])/np.sum(w[index==j]) for j in np.unique(index)])
    tbin = np.array([np.sum(ts[index==j]*w[index==j])/np.sum(w[index==j]) for j in np.unique(index)])
    dra_bin = np.array([np.sum(dra[index==j]*w[index==j])/np.sum(w[index==j]) for j in np.unique(index)])
    ddec_bin = np.array([np.sum(ddec[index==j]*w[index==j])/np.sum(w[index==j]) for j in np.unique(index)])
    eabsci_bin = np.array([np.std(absci[index==j]) for j in np.unique(index)])
    psi = np.arctan2(spsi,cpsi)
    psi_bin = np.array([np.sum(psi[index==j]*w[index==j])/np.sum(w[index==j]) for j in np.unique(index)])
    absres_bin = np.array([np.sum(absres[index==j]*w[index==j])/np.sum(w[index==j]) for j in np.unique(index)])
    
    # delete element if bin size = 1
    nn = np.array([np.sum(index==j) for j in np.unique(index)])
    ii = nn==1
    if(np.sum(ii)>0):
        tbin = tbin[~ii]
        dra_bin = dra_bin[~ii]
        ddec_bin = ddec_bin[~ii]
        eabsci_bin = eabsci_bin[~ii]
        psi_bin = psi_bin[~ii]
        ts_yrbin = ts_yrbin[~ii]
        absres_bin = absres_bin[~ii]
    
    cpsi_bin = np.cos(psi_bin)
    spsi_bin = np.sin(psi_bin)
    era_bin = eabsci_bin*cpsi_bin
    edec_bin = eabsci_bin*spsi_bin
    
    ################## plot hip IAD residual #################
    # plot un- and binned points, errorbar
    normalize = mcolors.Normalize(vmin=0, vmax=1)
    colormap = getattr(cm, 'plasma')#'YlOrRd')
    print('Hip IAD tmin:',np.min(tbin),np.max(tbin-np.min(tbin)))    
    phase = (tbin-np.min(tbin))%pday/np.max(tbin-np.min(tbin))
    
    if types=='1D':
        ra_ax.plot(tsim_yr, epoch_sim[0,:],'k-',lw=2,zorder=2)#,label=labs[0]
        ra_ax.errorbar(ts_yr,dra,yerr=np.abs(era), fmt='ko', capsize=0, capthick=1, elinewidth=1, ms=4.5, zorder=0, alpha=0.15)
        ra_ax.scatter(ts_yrbin,dra_bin,s=30, marker='o',c=colormap(normalize(phase)), edgecolor='w',alpha=0.99,lw=0.2,zorder=1)
        #ra_ax.errorbar(ts_yr,dra0,yerr=np.abs(era), fmt='ro', capsize=0, capthick=0, elinewidth=0, ms=4.5, zorder=0, alpha=0.15)
        dec_ax.plot(tsim_yr, epoch_sim[1,:],'k-',lw=2,zorder=2)#,label=labs[0]
        dec_ax.errorbar(ts_yr,ddec,yerr=np.abs(edec), fmt='ko', capsize=0, capthick=1, elinewidth=1, ms=4.5, zorder=0, alpha=0.15)
        dec_ax.scatter(ts_yrbin,ddec_bin,s=30, marker='o',c=colormap(normalize(phase)), edgecolor='w',alpha=0.99,lw=0.2,zorder=1)
        
    if types=='OC':
        ax3.errorbar(ts_yr,absres,yerr=eabsci, fmt='ko', capsize=0, capthick=0, elinewidth=0, ms=4.5, zorder=0, alpha=0.15)
        ax3.scatter(ts_yrbin,absres_bin,s=30, marker='o',c=colormap(normalize(phase)), edgecolor='w',alpha=0.99,lw=0.1,zorder=1)
        ax3.axhline(0,ls='--',lw=1,color='grey',zorder=0)

    if types=='2D':
        ax1.scatter(dra,ddec, s=30, marker='o',facecolor='k',edgecolor='w',alpha=0.25,lw=0.1,zorder=0)
        ax1.scatter(dra_bin,ddec_bin, s=30, marker='o',c=colormap(normalize(phase)), edgecolor='w',alpha=0.99,lw=0.1,zorder=1)
    
    x0, y0 = dra_bin-era_bin, ddec_bin-edec_bin
    x1, y1 = dra_bin+era_bin, ddec_bin+edec_bin
    y2, y3 = absres_bin-eabsci_bin, absres_bin+eabsci_bin
    
    # plot error bar
    for ii in range(len(dra_bin)):
        if types=='2D':
            ax1.plot([x0[ii], x1[ii]],[y0[ii], y1[ii]],color=colormap(normalize(phase[ii])),lw=1, zorder=1,)
        elif types=='OC':
            ax3.plot([ts_yrbin[ii], ts_yrbin[ii]],[y2[ii], y3[ii]],color=colormap(normalize(phase[ii])),lw=1, zorder=1,)
        if types=='1D':
            ra_ax.plot([ts_yrbin[ii], ts_yrbin[ii]],[x0[ii], x1[ii]],color=colormap(normalize(phase[ii])),lw=1, zorder=1,)
            dec_ax.plot([ts_yrbin[ii], ts_yrbin[ii]],[y0[ii], y1[ii]],color=colormap(normalize(phase[ii])),lw=1, zorder=1,)


def plot_proper_motion(tab, out, astrometry_epoch, norbits=200, iplanet=1):
    ################################ proper motion
    fig = plt.figure(figsize=(5.8, 5.8),dpi=120)
    ax4 = fig.add_axes((0.12, 0.54, 0.8, 0.35))
    ax5 = fig.add_axes((0.12, 0.10, 0.8, 0.35),sharex=ax4)
    
    Np = out['nplanet']
    if iplanet > Np: iplanet=Np
    
    lnp = tab['logpost'].values
    names = list(tab.keys())
    npar = names.index('logpost') # number of free parameters, the order should be same as MCMC pars
    
    # 2455197.5000 J2010
    tmin = (1987-2010)*365.25 + 2455197.5000
    tmax = (2020-2010)*365.25 + 2455197.5000
    tsim = np.linspace(tmin,tmax,1000)  # day
    tyr = (tsim-2455197.5000)/365.25+2010    
    ax4.set_xlim(1987,2020)
    
    inc2 = tab['Inc%d'%(iplanet)].values*180/np.pi
    m90 = inc2>90
    
    ilab = chr(98+iplanet-1)
    ############################ inc2<90 ######################
    if np.sum(~m90)>50:
        best_fit = np.argmax(lnp[~m90])                # Maximum lnL index
        par_opt = tab[~m90].iloc[[best_fit]].values[0]
        pars = np.copy(par_opt[:npar])
        
        norbpar = 7  
        for i in range(Np):  # reconstruct the free parameters
            pday = np.e**pars[i*norbpar]  # logP->P
            pars[i*norbpar] = pday
            pars[i*norbpar+1] = np.e**pars[i*norbpar+1] # logK->K
            esino, ecoso = pars[i*norbpar+2], pars[i*norbpar+3]
            ecc = esino**2 + ecoso**2
            pars[i*norbpar+2] = ecc
            pars[i*norbpar+3] = np.arctan2(esino, ecoso)
            
        epoch_sim = astrometry_epoch(pars,tt=tsim, out=out)
        pmra0, pmdec0 = pars[-3], pars[-2]
        pmra_sim = epoch_sim[3,:]#+(tsim-2457388.5)/365.25*(out['astrometry'].iloc[out['iref']]['pmra']-par_opt['dpmra'].values)
        pmdec_sim = epoch_sim[4,:]#+(tsim-2457388.5)/365.25*(out['astrometry'].iloc[out['iref']]['pmdec']-par_opt['dpmdec'].values)
        
        #ax1.plot(dra_sim-dra0,ddec_sim-ddec0, ccs[0]+'-',lw=0.5,zorder=2,alpha=0.1)#,label=labs[0]
        #dr3pmra, dr3pmdec = out['astrometry'].iloc[out['iref']]['pmra'], out['astrometry'].iloc[out['iref']]['pmdec']
        abspmra, abspmdec = out['astrometry'].iloc[out['iref']]['pmra']-pmra0, out['astrometry'].iloc[out['iref']]['pmdec']-pmdec0
        
        ax4.plot(tyr, abspmra+pmra_sim,'-',color='purple', lw=1.2,zorder=99,label=r'$i_{}<90^\circ$'.format(ilab))
        ax5.plot(tyr, abspmdec+pmdec_sim,'-',color='purple', lw=1.2,zorder=99) 
    
    ############################## inc2>90 ###################################
    if np.sum(m90)>50:
        best_fit = np.argmax(lnp[m90])                # Maximum lnL index
        par_opt = tab[m90].iloc[[best_fit]].values[0]
        pars = np.copy(par_opt[:npar])
        
        norbpar = 7  
        for i in range(Np):  # reconstruct the free parameters
            pday = np.e**pars[i*norbpar]  # logP->P
            pars[i*norbpar] = pday
            pars[i*norbpar+1] = np.e**pars[i*norbpar+1] # logK->K
            esino, ecoso = pars[i*norbpar+2], pars[i*norbpar+3]
            ecc = esino**2 + ecoso**2
            pars[i*norbpar+2] = ecc
            pars[i*norbpar+3] = np.arctan2(esino, ecoso)
            
        epoch_sim = astrometry_epoch(pars,tt=tsim, out=out)
        pmra0, pmdec0 = pars[-3], pars[-2]
        pmra_sim = epoch_sim[3,:]#+(tsim-2457388.5)/365.25*(out['astrometry'].iloc[out['iref']]['pmra']-par_opt['dpmra'].values)
        pmdec_sim = epoch_sim[4,:]#+(tsim-2457388.5)/365.25*(out['astrometry'].iloc[out['iref']]['pmdec']-par_opt['dpmdec'].values)
        
        #ax1.plot(dra_sim-dra0,ddec_sim-ddec0, ccs[0]+'-',lw=0.5,zorder=2,alpha=0.1)#,label=labs[0]
        #dr3pmra, dr3pmdec = out['astrometry'].iloc[out['iref']]['pmra'], out['astrometry'].iloc[out['iref']]['pmdec']
        abspmra, abspmdec = out['astrometry'].iloc[out['iref']]['pmra']-pmra0, out['astrometry'].iloc[out['iref']]['pmdec']-pmdec0
        
        ax4.plot(tyr, abspmra+pmra_sim,'-',color='orange', lw=1.2,zorder=99,label=r'$i_{}>90^\circ$'.format(ilab))
        ax5.plot(tyr, abspmdec+pmdec_sim,'-',color='orange', lw=1.2,zorder=99)    
    
    ############################################################################
    
    random_index = random.sample(range(len(lnp)), norbits)
    for ind in random_index:
        par_opt = tab.iloc[[ind]].values[0]
        pars = np.copy(par_opt[:npar])
        
        norbpar = 7  
        for i in range(Np):  # reconstruct the free parameters
            pday = np.e**pars[i*norbpar]  # logP->P
            pars[i*norbpar] = pday
            pars[i*norbpar+1] = np.e**pars[i*norbpar+1] # logK->K
            esino, ecoso = pars[i*norbpar+2], pars[i*norbpar+3]
            ecc = esino**2 + ecoso**2
            pars[i*norbpar+2] = ecc
            pars[i*norbpar+3] = np.arctan2(esino, ecoso)
            
        epoch_sim = astrometry_epoch(pars,tt=tsim, out=out)
        pmra0, pmdec0 = pars[-3], pars[-2]
        pmra_sim = epoch_sim[3,:]#+(tsim-2457388.5)/365.25*(out['astrometry'].iloc[out['iref']]['pmra']-par_opt['dpmra'].values)
        pmdec_sim = epoch_sim[4,:]#+(tsim-2457388.5)/365.25*(out['astrometry'].iloc[out['iref']]['pmdec']-par_opt['dpmdec'].values)
        
        #ax1.plot(dra_sim-dra0,ddec_sim-ddec0, ccs[0]+'-',lw=0.5,zorder=2,alpha=0.1)#,label=labs[0]
        #dr3pmra, dr3pmdec = out['astrometry'].iloc[out['iref']]['pmra'], out['astrometry'].iloc[out['iref']]['pmdec']
        abspmra, abspmdec = out['astrometry'].iloc[out['iref']]['pmra']-pmra0, out['astrometry'].iloc[out['iref']]['pmdec']-pmdec0
        
        inc2 = pars[(iplanet-1)*norbpar+6]*180/np.pi
        if inc2<90:
            ax4.plot(tyr, abspmra+pmra_sim,'-',color='purple',lw=0.1,alpha=0.05)
            ax5.plot(tyr, abspmdec+pmdec_sim,'-',color='purple',lw=0.1,alpha=0.05)
        else:
            ax4.plot(tyr, abspmra+pmra_sim,'-',color='orange',lw=0.1,alpha=0.05)
            ax5.plot(tyr, abspmdec+pmdec_sim,'-',color='orange',lw=0.1,alpha=0.05)            
    
    Astins = ['Hipparcos', 'GDR2', 'GDR3']
    cc = ['r','g','b']
    for j in range(3):
        cata_pmra, cata_pmdec = out['astrometry'].iloc[j]['pmra'], out['astrometry'].iloc[j]['pmdec']
        epmra, epmdec = out['astrometry'].iloc[j]['pmra_error'], out['astrometry'].iloc[j]['pmdec_error']
        ref_epoch = out['astrometry'].iloc[j]['ref_epoch']
        print(cata_pmra, cata_pmdec, epmra, epmdec)
        if j==0:
            hipbjd = out['hip_array'][0]
            xerr = (np.max(hipbjd)-np.min(hipbjd))/365.25/2
        if j>0:
            gostbjd = out['gost_array'][0,:]
            gostbjd = gostbjd[gostbjd<out['GDR%d_baseline'%(j+1)]]
            xerr = (np.max(gostbjd)-np.min(gostbjd))/365.25/2
        
        ax4.errorbar((ref_epoch-2455197.5000)/365.25+2010, cata_pmra, xerr=xerr,yerr=epmra, zorder=99,
                          fmt=cc[j]+'o', ecolor=cc[j],capsize=3, capthick=1.35, elinewidth=1.35,ms=6,mec=cc[j])
        ax5.errorbar((ref_epoch-2455197.5000)/365.25+2010, cata_pmdec, xerr=xerr,yerr=epmdec, zorder=99,
                          fmt=cc[j]+'o', ecolor=cc[j],capsize=3, capthick=1.35, elinewidth=1.35,ms=6,mec=cc[j], label=Astins[j])
    
    plt.setp(ax4.get_xticklabels(), visible=False)

    
    for ax in [ax4, ax5]:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(direction='in', which='both', left=True, right=True, bottom=True, top=True,labelsize=14)
    
    ax4.text(0.5,1.075, 'Acceleration', horizontalalignment='center',verticalalignment='center', transform=ax4.transAxes,fontsize=14)
    #ax2.text(0.88,0.90, 'Zoom in', horizontalalignment='center',verticalalignment='center', transform=ax2.transAxes,fontsize=14)
    
    ax5.set_xlabel('Epoch [year]',fontsize=16)
    ax4.set_ylabel(r'$\mu_{\alpha*}$ [mas/yr]',fontsize=16)
    ax5.set_ylabel(r'$\mu_{\delta}$ [mas/yr]',fontsize=16)
    ax5.legend(fontsize=13,bbox_to_anchor=(0.49,1.11),loc='center',ncol=6,frameon=False,columnspacing=1)
    ax4.legend(loc='upper center',ncol=2,frameon=False,fontsize=13,)

######################### plot direct Imaging ######################
########################### first create fig #######################
def plot_DI(tab, out, astrometry_epoch, iplanet=1, norbit=100, ax=None):
    
    if ax is None:
        fig = plt.figure(figsize=(12.8, 4.2),dpi=150)
        # position
        ax1 = fig.add_axes((0.08, 0.10, 0.25, 0.59+0.2))
    else:
        ax1 = ax
    
    ax1.set_xlabel(r'$\Delta \alpha*$ [arcsec]',fontsize=18)
    ax1.set_ylabel(r'$\Delta \delta$ [arcsec]',fontsize=18)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(direction='in', which='both', left=True, right=True, bottom=True, top=True,labelsize=14)

    ax = ax1
    fcol = ['k','r','b',]
    tab = tab
    Np = out['nplanet']
    lnp = tab['logpost'].values
    best_fit = np.argmax(lnp)                # Maximum lnL index
    par_opt = tab.iloc[[best_fit]].values[0]
    
    names = list(tab.keys())
    npar = names.index('logpost') # number of free parameters, the order should be same as MCMC pars
    pars = np.copy(par_opt[:npar])
    norbpar = 7  
    
    for i in range(Np):  # reconstruct the free parameters
        pday = np.e**pars[i*norbpar]  # logP->P
        pars[i*norbpar] = pday
    
        pars[i*norbpar+1] = np.e**pars[i*norbpar+1] # logK->K
        esino, ecoso = pars[i*norbpar+2], pars[i*norbpar+3]
        ecc = esino**2 + ecoso**2
        pars[i*norbpar+2] = ecc
        pars[i*norbpar+3] = np.arctan2(esino, ecoso)
    
    iplanet = iplanet  # used to prediction of planet i
    mstar = par_opt[names.index('Mstar')]   # Msun
    mp = par_opt[names.index('Mc%d'%iplanet)]        # MJ -> Msun
    itmin = 2445000 #out['tmin']
    tsim = np.linspace(itmin,itmin+2*pday,1000)
    lenDI = len(out['relAst']['rel_JD'])
    tsim = np.concatenate([tsim, out['relAst']['rel_JD']])
    reflex_sim = astrometry_epoch(pars,tt=tsim,iplanet=iplanet,out=out)                # pd.DataFrame
    xis = -mp/(mstar+mp)
    dra_sim, ddec_sim = reflex_sim[0,:][:-lenDI]/xis*1e-3, reflex_sim[1,:][:-lenDI]/xis*1e-3
    # plot best-fit complete orbit (best-fit)
    ax.plot(dra_sim,ddec_sim, 'k-',lw=1.5,zorder=2)
    # plot nodes & periastron
    dra_node, ddec_node = nodes_periastron(pars, plx=out['plx'], types='planet',iplanet=iplanet,out=out)
    ax.plot(dra_node[2:]*1e-3, ddec_node[2:]*1e-3,'k--',lw=1,alpha=0.7)
    ax.plot([0,dra_node[1]*1e-3], [0,ddec_node[1]*1e-3],fcol[0],lw=1.1,alpha=0.7)    
    ax.plot(0,0,color='orange',marker='*',ms=12,zorder=3)
    
    tsim = np.linspace(itmin,itmin+2*pday,1000)
    for i in range(norbit):
        ind = random.randint(0,len(lnp))
        par_opt = tab.iloc[ind].values
        pars = np.copy(par_opt[:npar])
        for i in range(Np):  # reconstruct the free parameters
            pday = np.e**pars[i*norbpar]  # logP->P
            pars[i*norbpar] = pday
            pars[i*norbpar+1] = np.e**pars[i*norbpar+1] # logK->K
            esino, ecoso = pars[i*norbpar+2], pars[i*norbpar+3]
            ecc = esino**2 + ecoso**2
            pars[i*norbpar+2] = ecc
            pars[i*norbpar+3] = np.arctan2(esino, ecoso)       
        mstar = par_opt[names.index('Mstar')]   # Msun
        mp = par_opt[names.index('Mc%d'%iplanet)]        # MJ -> Msun 
        reflex_sim = astrometry_epoch(pars,tt=tsim,iplanet=iplanet,out=out)
        xis = -mp/(mstar+mp)
        dra_sim, ddec_sim = reflex_sim[0,:][:-lenDI]/xis*1e-3, reflex_sim[1,:][:-lenDI]/xis*1e-3
        # plot best-fit complete orbit (best-fit)
        ax.plot(dra_sim,ddec_sim, 'k-',lw=0.6,zorder=0, alpha=0.2)        
    # plot direct imaging data is available
    ccs = ['r','r','r']

    if 'relAst' in out.keys():
        for i in range(len(out['relAst']['rel_sep'])):
            cc = 'r'#ccs[i]
            reljd =  out['relAst']['rel_JD'][i]
            epoch = (reljd-2451544.5)/365.25+2000  # JD to yr
            #out['relAst_type'] = 'Dra_Ddec'   # 'Dra_Ddec', units:"(arcsec);  'Sep_PA', units:"(arcsec) and deg
            
            if out['relAst_type']=='Sep_PA':
                sp, esp = out['relAst']['rel_sep'][i], out['relAst']['rel_sep_err'][i]
                pa, epa = out['relAst']['rel_PA'][i], out['relAst']['rel_PA_err'][i]
                sps = (np.random.randn(10000)*esp + sp)        # arcsec -. mas
                pas = (np.random.randn(10000)*epa + pa)*np.pi/180   # rad
                y2 = sps**2/(1+(np.tan(pas))**2)
                x2 = sps**2-y2
                xs, ys = np.sqrt(x2), np.sqrt(y2)
                x, ex, y, ey = np.mean(xs), np.std(xs), np.mean(ys), np.std(ys)
                print('Sep_PA to Dra_Ddec:',x, ex, y, ey)
                if pa<0: pa = 360+pa
                if pa>180: x=-x
                if (pa>90) and (pa<270): y=-y                
                
            elif out['relAst_type']=='Dra_Ddec':
                x, ex = out['relAst']['rel_sep'][i], out['relAst']['rel_sep_err'][i]
                y, ey = out['relAst']['rel_PA'][i], out['relAst']['rel_PA_err'][i]
                xs = (np.random.randn(10000)*ex + x)
                ys = (np.random.randn(10000)*ey + y)
                sps = np.sqrt(xs**2+ys**2)
                pas = np.arctan2(xs, ys)*180/pi
                sp, esp, pa, epa = np.mean(sps), np.std(sps), np.mean(pas), np.std(pas)
                if pa<0: pa = 360+pa
            ax.errorbar(x,y, xerr=ex,yerr=ey, fmt=cc+'o', ecolor='black', capsize=4, capthick=1, elinewidth=1.2,ms=8.5,mec='k', zorder = 299,label='JWST F444W')
    ax.invert_xaxis()
    ax.legend(fontsize=12,)

def plot_DI_sep_theta(tab, out, astrometry_epoch, iplanet=1):

    # plot companion position & sep & PA fitting
    fig = plt.figure(figsize=(12.8, 4.2),dpi=150)
    # position
    ax1 = fig.add_axes((0.08, 0.10, 0.25, 0.59+0.2))
    
    # Sep 
    ax3 = fig.add_axes((0.41, 0.30, 0.25, 0.59))
    ax4 = fig.add_axes((0.411, 0.10, 0.25, 0.18))
    
    # PA
    ax5 = fig.add_axes((0.734, 0.30, 0.25, 0.59))
    ax6 = fig.add_axes((0.734, 0.10, 0.25, 0.18))
    
    ax1.set_xlabel(r'$\Delta \alpha*$ [arcsec]',fontsize=18)
    ax1.set_ylabel(r'$\Delta \delta$ [arcsec]',fontsize=18)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(direction='in', which='both', left=True, right=True, bottom=True, top=True,labelsize=14)
    
    ax3.set_ylabel(r'$\rho\,$[arcsec]', fontsize = 18)
    ax5.set_ylabel(r'$\theta\,$[$^\circ$]', fontsize = 18)
    for ax in [ax3, ax5]:
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(direction='in', which='both', left=True, right=True, bottom=True, top=True,labelsize=14)
        
    for ax in [ax4,ax6]:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(direction='in', which='both', left=True, right=True, bottom=True, top=True,labelsize=14)
        ax.set_xlabel('Epoch [year]', fontsize = 18)
        ax.set_ylabel('O-C', fontsize = 18)
    
    for iax in [ax1, ax3, ax4, ax5, ax6]:
        plt.setp(iax.spines.values(), linewidth=1.5)
    
    ax3.get_shared_x_axes().join(ax3, ax4)
    ax5.get_shared_x_axes().join(ax5, ax6)
    
    ax = ax1
    fcol = ['k','r','b',]
    tab = tab
    Np = out['nplanet']
    lnp = tab['logpost'].values
    best_fit = np.argmax(lnp)                # Maximum lnL index
    par_opt = tab.iloc[[best_fit]].values[0]
    
    names = list(tab.keys())
    npar = names.index('logpost') # number of free parameters, the order should be same as MCMC pars
    pars = np.copy(par_opt[:npar])
    norbpar = 7  
    
    for i in range(Np):  # reconstruct the free parameters
        pday = np.e**pars[i*norbpar]  # logP->P
        pars[i*norbpar] = pday
    
        pars[i*norbpar+1] = np.e**pars[i*norbpar+1] # logK->K
        esino, ecoso = pars[i*norbpar+2], pars[i*norbpar+3]
        ecc = esino**2 + ecoso**2
        pars[i*norbpar+2] = ecc
        pars[i*norbpar+3] = np.arctan2(esino, ecoso)
    
    iplanet = iplanet  # used to prediction of planet i
    
    #inc_deg = par_opt[names.index('Inc%d'%iplanet)]*180/np.pi
    
    mstar = par_opt[names.index('Mstar')]   # Msun
    mp = par_opt[names.index('Mc%d'%iplanet)]        # MJ -> Msun
    
    
    itmin = 2455000 #out['tmin']
    tsim = np.linspace(itmin,itmin+2*pday,1000)
    tsim_yr = (tsim-2451544.5)/365.25+2000  # JD to yr
    lenDI = len(out['relAst']['rel_JD'])
    tsim = np.concatenate([tsim, out['relAst']['rel_JD']])
    reflex_sim = astrometry_epoch(pars,tt=tsim,iplanet=iplanet,out=out)          # pd.DataFrame
    #print(reflex_sim)
    xis = -mp/(mstar+mp)
    dra_sim, ddec_sim = reflex_sim[0,:][:-lenDI]/xis*1e-3, reflex_sim[1,:][:-lenDI]/xis*1e-3
    pa_sim = np.arctan2(dra_sim,ddec_sim)*180/pi#deg
    #print(np.min(pa_sim),np.max(pa_sim))
    pa_sim[pa_sim<0] += 360
    sep_sim = np.sqrt(dra_sim**2+ddec_sim**2) #arcsec
    
    # plot complete orbit (best-fit)
    ax.plot(dra_sim,ddec_sim, 'k-',lw=1.5,zorder=2)
    
    # plot nodes & periastron
    dra_node, ddec_node = nodes_periastron(pars, plx=out['plx'], types='planet',iplanet=iplanet,out=out)
    ax.plot(dra_node[2:]*1e-3, ddec_node[2:]*1e-3,'k--',lw=1,alpha=0.7)
    ax.plot([0,dra_node[1]*1e-3], [0,ddec_node[1]*1e-3],fcol[0],lw=1.1,alpha=0.7)
    
    ax3.plot(tsim_yr, sep_sim,'k-',lw=1.5,zorder=2)
    ax5.plot(tsim_yr, pa_sim,'k-',lw=1.5,zorder=2)
    ax.plot(0,0,color='orange',marker='*',ms=12,zorder=3)
    
    # plot direct imaging data is available
    ccs = ['r','r','r']

    if 'relAst' in out.keys():
        for i in range(len(out['relAst']['rel_sep'])):
            cc = 'r'#ccs[i]
            reljd =  out['relAst']['rel_JD'][i]
            epoch = (reljd-2451544.5)/365.25+2000  # JD to yr
            #out['relAst_type'] = 'Dra_Ddec'   # 'Dra_Ddec', units:"(arcsec);  'Sep_PA', units:"(arcsec) and deg
            
            if out['relAst_type']=='Sep_PA':
                sp, esp = out['relAst']['rel_sep'][i], out['relAst']['rel_sep_err'][i]
                pa, epa = out['relAst']['rel_PA'][i], out['relAst']['rel_PA_err'][i]
                sps = (np.random.randn(10000)*esp + sp)        # arcsec -. mas
                pas = (np.random.randn(10000)*epa + pa)*np.pi/180   # rad
                y2 = sps**2/(1+(np.tan(pas))**2)
                x2 = sps**2-y2
                xs, ys = np.sqrt(x2), np.sqrt(y2)
                x, ex, y, ey = np.mean(xs), np.std(xs), np.mean(ys), np.std(ys)
                print('Sep_PA to Dra_Ddec:',x, ex, y, ey)
                if pa<0: pa = 360+pa
                if pa>180: x=-x
                if (pa>90) and (pa<270): y=-y                
                
            elif out['relAst_type']=='Dra_Ddec':
                x, ex = out['relAst']['rel_sep'][i], out['relAst']['rel_sep_err'][i]
                y, ey = out['relAst']['rel_PA'][i], out['relAst']['rel_PA_err'][i]
                xs = (np.random.randn(10000)*ex + x)
                ys = (np.random.randn(10000)*ey + y)
                sps = np.sqrt(xs**2+ys**2)
                pas = np.arctan2(xs, ys)*180/pi
                sp, esp, pa, epa = np.mean(sps), np.std(sps), np.mean(pas), np.std(pas)
                if pa<0: pa = 360+pa
                
            ax.errorbar(x,y, xerr=ex,yerr=ey, fmt=cc+'o', ecolor='black', capsize=4, capthick=1, elinewidth=1.2,ms=8.5,mec='k', zorder = 299,)
            
                          # pd.DataFrame
            dra_sim, ddec_sim = reflex_sim[0,:][-lenDI:][i]/xis*1e-3, reflex_sim[1,:][-lenDI:][i]/xis*1e-3
            pa_sim = np.arctan2(dra_sim,ddec_sim)*180/pi#deg
            pa_sim = pa_sim%360 
            sep_sim = np.sqrt(dra_sim**2+ddec_sim**2)#as
            
            ax3.errorbar(epoch,sp, yerr=esp, fmt=cc+'o', ecolor='black', alpha = 0.8, zorder = 299,ms=1,capsize=4,)
            ax3.scatter(epoch,sp, s=70, facecolors=cc, edgecolors='k', alpha = 0.99, zorder=300)
            
            ax4.errorbar(epoch,sp-sep_sim, yerr=esp, fmt=cc+'o', ecolor='black', alpha = 0.8, zorder = 299,ms=1,capsize=4,)
            ax4.scatter(epoch,sp-sep_sim, s=70, facecolors=cc, edgecolors='k', alpha = 0.99, zorder=300)
            
            ax5.errorbar(epoch,pa, yerr=epa, fmt=cc+'o', ecolor='black', alpha = 0.8, zorder = 299,ms=1,capsize=4,)
            ax5.scatter(epoch,pa, s=70, facecolors=cc, edgecolors='k', alpha = 0.99, zorder=300)
    
            ax6.errorbar(epoch,pa-pa_sim, yerr=epa, fmt=cc+'o', ecolor='black', alpha = 0.8, zorder = 299,ms=1,capsize=4,)
            ax6.scatter(epoch,pa-pa_sim, s=70, facecolors=cc, edgecolors='k', alpha = 0.99, zorder=300)
            print('lallalallala------')
    
    ax4.axhline(0, color='0.5', linestyle='--',lw=1.2)
    ax6.axhline(0, color='0.5', linestyle='--',lw=1.2)    
    ax.invert_xaxis()
    # for iax in [ax3, ax5]:
    #     iax.set_xlim(1995,2024)

    # ax6.set_ylim(-3.6,3.6)
    # ax4.set_ylim(-0.38,0.38)
    # ax5.set_ylim(29,56)
    # ax3.set_ylim(3.4,5.6)
    
    align_ylabels(fig, [ax3,ax4], pad=0.01)
    align_ylabels(fig, [ax5,ax6], pad=0.01)
    
    for i, iax in enumerate([ax,ax3,ax5]):
        xoff = off = 0
        if (i==0):
            xoff,off = 0.00, -0.03
        iax.text(0.07-xoff,0.92-off, '({})'.format(chr(i+97)), horizontalalignment='center',verticalalignment='center', transform=iax.transAxes,fontsize=14,fontweight='bold')
        
    # ax.legend(handles=[plt.scatter([], [], color=ccs[0], s=50, edgecolors='k',label='NEAR'),
    #                 plt.scatter([], [], color=ccs[1], s=50,edgecolors='k',label='MIRI')],
    #     bbox_to_anchor=(0.15,0.08),loc='center',ncol=1,fontsize=11,frameon=False,labelspacing=0.15,columnspacing=0.00) 
    
    
    #plt.savefig(target+'_comp_position.pdf',bbox_inches='tight',transparent=False)
    
    plt.show()

def myboxplot(ax, data, xpos, width, c):
    q03, q16, q50, q84, q99 = np.percentile(data, [0.3, 16, 50, 84, 99.7])
    ax.plot([xpos-width/2, xpos+width/2],[q50,q50],lw=2.2,color=c)
    ax.plot([xpos-width/2, xpos-width/2, xpos+width/2, xpos+width/2, xpos-width/2,],
            [q16,q84,q84,q16,q16],lw=1.2,color=c)
    
    ax.plot([xpos-width/4, xpos+width/4],[q03,q03],lw=1.2,color=c)
    ax.plot([xpos-width/4, xpos+width/4],[q99,q99],lw=1.2,color=c)

    ax.plot([xpos, xpos],[q99,q84],lw=1.2,color=c)
    ax.plot([xpos, xpos],[q03,q16],lw=1.2,color=c)


def plot_model_rv(tab, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(6.8, 4.2),dpi=150)
        # position
        ax1 = fig.add_axes((0.17, 0.12, 0.72, 0.59+0.2))
    else:
        ax1 = ax
    ax1.set_xlabel(r'JD-2450000',fontsize=18)
    ax1.set_ylabel(r'RV [m/s]',fontsize=18)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(direction='in', which='both', left=True, right=True, bottom=True, top=True,labelsize=14)

    names = list(tab.keys())
    Np = np.sum(['logP' in i for i in names])
    lnp = tab['logpost'].values
    best_fit = np.argmax(lnp)                # Maximum lnL index   
    key = ['Pd', 'Tp', 'e', 'omega', 'logK']
    jd = np.linspace(1945, 2030, 1000)
    jds = (jd - 2000)*365.25 + 2451544.5
    model_rv = np.zeros_like(jd)
    for i in range(Np):
        orbpar= [tab[k+'%d'%(i+1)].values[best_fit] for k in key]
        orbpar[-1] = np.e**orbpar[-1]
        model_rv += rv_calc(jds, orbpar)
    ax1.axhline(0,ls='--')
    ax1.plot(jds-2450000, model_rv)
    
    ############################### for HIP38957 #############################
    if True:
        #ax1.axvline(2444653.24395-2450000)
        #ax1.axvline(2458924.5105-2450000)
        #ax1.axvline(2447160.2655-2450000)
        x = np.array([2444653.24395, 2449800, 2458924.5105])-2450000
        y = np.array([13.2,19.4,20])*1e3
        y = y-(y[-1]-3560)
        yerr = np.array([2.5,1.9,1.1])*1e3
        ax1.errorbar(x,y, yerr=yerr, fmt='s', mec='none',alpha=0.99, zorder=99,ms=8,capsize=4,) 
        #ax1.axvline((1983 - 2000)*365.25 + 2451544.5-2450000)
        fns = glob.glob('HIP38957/binary_rv/VPup_*.csv')
        fns = sorted(fns)
        for i, fn in enumerate(fns):
            tab = pd.read_csv(fn)
            hjd_col = tab.keys()[0]
            tmp = float(hjd_col.split('_')[-1])
            if i==0:
                hjd = tab[hjd_col].values + tmp
                RV1 = tab.RV1.values
                RV2 = tab.RV2.values
            else:
                hjd = np.concatenate([hjd, tab[hjd_col].values + tmp])
                RV1 = np.concatenate([RV1, tab.RV1.values])
                RV2 = np.concatenate([RV2, tab.RV2.values])
        phase = (hjd-2445367.60633)%1.4544859/1.4544859
        from scipy.interpolate import interp1d
        tab = pd.read_csv('HIP38957/binary_rv/Model_RV1.csv')
        S_func1 = interp1d(tab.phase.values, tab.RV.values, kind='linear',bounds_error=False)
        tab = pd.read_csv('HIP38957/binary_rv/Model_RV2.csv')
        S_func2 = interp1d(tab.phase.values, tab.RV.values, kind='linear',bounds_error=False)
        ax1.scatter(hjd-2450000, (RV1-S_func1(phase))*1e3)
        ax1.scatter(hjd-2450000, (RV2-S_func2(phase))*1e3)
        
        if False:
            jd, rv  = hjd, (RV1-S_func1(phase))*1e3
            ts1 = np.array([jd[0]] + list(jd))
            dt  = np.diff(ts1)
            index, ii = [], 0
            for j in range(len(dt)):
                if(dt[j]>=2): ii = ii+1
                index.append(ii)
            index   = np.array(index)
            w = 1/np.repeat(1, len(jd))**2
            t_bin   = np.array([np.sum(jd[index==j]*w[index==j])/np.sum(w[index==j]) for j in np.unique(index)])
            rv_bin  = np.array([np.sum(rv[index==j]*w[index==j])/np.sum(w[index==j]) for j in np.unique(index)])        
            
            ax1.scatter(t_bin-2450000, rv_bin)
            
            jd, rv  = hjd, (RV2-S_func2(phase))*1e3
            ts1 = np.array([jd[0]] + list(jd))
            dt  = np.diff(ts1)
            index, ii = [], 0
            for j in range(len(dt)):
                if(dt[j]>=2): ii = ii+1
                index.append(ii)
            index   = np.array(index)
            w = 1/np.repeat(1, len(jd))**2
            t_bin   = np.array([np.sum(jd[index==j]*w[index==j])/np.sum(w[index==j]) for j in np.unique(index)])
            rv_bin  = np.array([np.sum(rv[index==j]*w[index==j])/np.sum(w[index==j]) for j in np.unique(index)])        
            
            ax1.scatter(t_bin-2450000, rv_bin)
    ##########################################################################
    
    ax = ax1
    # years on upper axis
    axyrs = ax.twiny()
    xl = np.array(list(ax.get_xlim())) + 2450000
    decimalyear = Time(xl, format='jd', scale='utc').decimalyear
    axyrs.get_xaxis().get_major_formatter().set_useOffset(False)
    axyrs.set_xlim(*decimalyear)
    axyrs.set_xlabel('Year', fontsize=14)#fontweight='bold',
    axyrs.xaxis.set_minor_locator(AutoMinorLocator())
    axyrs.tick_params(direction='in', which='both',labelsize=13)
    plt.locator_params(axis='x', nbins=8)

####################### return 5-p astrometry residuals #####################
def return_ast_res(out, astrometry_epoch, astrometry_kepler, tab=None, iplanet=None, use_starfn=None, zero_planet=False):
    
    if zero_planet:
        pars = np.ones(len(out['new_index']))*0.5
        pars[-6:-1] = 0   # dra = ddec = dplx = dpmra = dpmdec = 0
        kep = astrometry_kepler(pars, out=out, iplanet=iplanet, state=True)    
        return kep['cats'], kep['bary']        
    
    Np = out['nplanet']
    if use_starfn is not None:
        if os.path.exists(use_starfn):
            print('\nload start file:',use_starfn)
            init, pname = [], []
            f = open(use_starfn)
            for line in f:
                if line[0]=='#':continue
                lst = line.strip().split()
                pname.append(lst[0])
                init.append(float(lst[1]))
            f.close()
            names, pars = pname, np.asarray(init)
        else:
            print('\nstar file not found:',use_starfn)
            sys.exit()
    else:
        lnp = tab['logpost'].values
        best_fit = np.argmax(lnp)                # Maximum lnL index
        par_opt = tab.iloc[[best_fit]].values[0]
    
        names = list(tab.keys())
        npar = names.index('logpost') # number of free parameters, the order should be same as MCMC pars
        pars = np.copy(par_opt[:npar])
    
    norbpar = 7     
    
    for i in range(Np):  # reconstruct the free parameters
        pday = np.e**pars[i*norbpar]  # logP->P
        pars[i*norbpar] = pday
    
        pars[i*norbpar+1] = np.e**pars[i*norbpar+1] # logK->K
        esino, ecoso = pars[i*norbpar+2], pars[i*norbpar+3]
        ecc = esino**2 + ecoso**2
        pars[i*norbpar+2] = ecc
        pars[i*norbpar+3] = np.arctan2(esino, ecoso)
    
    iplanet = iplanet  # used to prediction of planet i

    kep = astrometry_kepler(pars, out=out, iplanet=iplanet)    

    return kep['cats'], kep['bary']

###################### add boxplot to compare different solutions ##############
def plot_boxplot(tab, out, astrometry_epoch, astrometry_kepler, nsamp=100, iplanet=None, use_starfn=None):
    
    fig, ax = plt.subplots(1,5,figsize=(12.5,6),dpi=120)
    fig.subplots_adjust(hspace=0.45, wspace=0.50)
    
    Np = out['nplanet']
    if use_starfn is not None:
        if os.path.exists(use_starfn):
            print('\nload start file:',use_starfn)
            init, pname = [], []
            f = open(use_starfn)
            for line in f:
                if line[0]=='#':continue
                lst = line.strip().split()
                pname.append(lst[0])
                init.append(float(lst[1]))
            f.close()
            names, pars = pname, np.asarray(init)
        else:
            print('\nstar file not found:',use_starfn)
            sys.exit()
    else:
        lnp = tab['logpost'].values
        best_fit = np.argmax(lnp)                # Maximum lnL index
        par_opt = tab.iloc[[best_fit]].values[0]
    
        names = list(tab.keys())
        npar = names.index('logpost') # number of free parameters, the order should be same as MCMC pars
        pars = np.copy(par_opt[:npar])
    
    norbpar = 7     
    
    for i in range(Np):  # reconstruct the free parameters
        pday = np.e**pars[i*norbpar]  # logP->P
        pars[i*norbpar] = pday
    
        pars[i*norbpar+1] = np.e**pars[i*norbpar+1] # logK->K
        esino, ecoso = pars[i*norbpar+2], pars[i*norbpar+3]
        ecc = esino**2 + ecoso**2
        pars[i*norbpar+2] = ecc
        pars[i*norbpar+3] = np.arctan2(esino, ecoso)
    
    iplanet = iplanet  # used to prediction of planet i

    try:
        J_gaia = pars[names.index('J_gaia')]
    except:
        J_gaia = 1
    
    kep = astrometry_kepler(pars, out=out, iplanet=iplanet, Zero_planet=False)
    ###observed position and proper motion relative to the predicted barycentric position and proper motion

    dra_obs = (out['astrometry'].iloc[out['astro_index']]['ra'].values-kep['barycenter'][out['astro_index'], 0])*np.cos(out['astrometry'].iloc[out['astro_index']]['dec'].values/180*pi)*3.6e6  #deg2mas
    ddec_obs = (out['astrometry'].iloc[out['astro_index']]['dec'].values-kep['barycenter'][out['astro_index'], 1])*3.6e6
    dpmra_obs = out['astrometry'].iloc[out['astro_index']]['pmra'].values-kep['barycenter'][out['astro_index'], 3]
    dpmdec_obs = out['astrometry'].iloc[out['astro_index']]['pmdec'].values-kep['barycenter'][out['astro_index'], 4]
    dplx_obs = out['astrometry'].iloc[out['astro_index']]['parallax'].values-kep['barycenter'][out['astro_index'], 2]
    era_obs = out['astrometry'].iloc[out['astro_index']]['ra_error'].values
    edec_obs = out['astrometry'].iloc[out['astro_index']]['dec_error'].values
    eplx_obs = out['astrometry'].iloc[out['astro_index']]['parallax_error'].values
    epmra_obs = out['astrometry'].iloc[out['astro_index']]['pmra_error'].values
    epmdec_obs = out['astrometry'].iloc[out['astro_index']]['pmdec_error'].values
    
    ###the position and proper motion reconstructed by a linear fit to gost-based observations generated by accounting for reflex motion
    a, b, c, d, e = [],[],[],[],[]  # get fitted ast posterior at DR2,3
    inds = random.sample(list(range(len(tab))), nsamp)
    for j in inds:
        par_opt = tab.iloc[[j]].values[0]
        
        names = list(tab.keys())
        npar = names.index('logpost') # number of free parameters, the order should be same as MCMC pars
        pars = np.copy(par_opt[:npar])
        
        for i in range(Np):  # reconstruct the free parameters
            pday = np.e**pars[i*norbpar]  # logP->P
            pars[i*norbpar] = pday
        
            pars[i*norbpar+1] = np.e**pars[i*norbpar+1] # logK->K
            esino, ecoso = pars[i*norbpar+2], pars[i*norbpar+3]
            ecc = esino**2 + ecoso**2
            pars[i*norbpar+2] = ecc
            pars[i*norbpar+3] = np.arctan2(esino, ecoso)
        #pars[-6:-1] = 0   # dra = ddec = dplx = dpmra = dpmdec = 0
        kep = astrometry_kepler(pars, out=out, iplanet=iplanet, Zero_planet=False)

        a.append(np.array(dra_obs-kep['cats'][:,0]))
        b.append(np.array(ddec_obs-kep['cats'][:,1]))
        c.append(np.array(dpmra_obs-kep['cats'][:,3]))
        d.append(np.array(dpmdec_obs-kep['cats'][:,4]))
        e.append(np.array(dplx_obs-kep['cats'][:,2]))

    a, b, c = np.array(a), np.array(b), np.array(c)
    d, e = np.array(d), np.array(e)
    tsmps = np.hstack([a,b,c,d,e])
    
    for iax in range(5):
        ax2 = ax[iax]
        ## plot dr23 simulated data (from gost sampling)        
        oc = ['green','blue']
        for k in range(len(out['astro_index'])):
            # observed DR2 & DR3        
            tasts = np.array([dra_obs[k],ddec_obs[k],dpmra_obs[k],dpmdec_obs[k], dplx_obs[k]])
            tasterr = [era_obs[k],edec_obs[k],epmra_obs[k],epmdec_obs[k], eplx_obs[k]]

            g1, g2 = iax*1, (iax+1)*1
            if g2>len(tasts):g2==len(tasts)
            asts, asterr = tasts[g1:g2], tasterr[g1:g2]
            for j, _ in enumerate(asts):
                #print(iax, g1, g2)
                ax2.scatter(k,asts[j],marker='s',color=oc[k],label=out['cats'][k],alpha=0.6)
                ax2.errorbar(k,asts[j], yerr=asterr[j]*J_gaia, fmt='s', mfc=oc[k], mec='none',alpha=0.6, ecolor=oc[k], zorder = 0,ms=8,capsize=0,)       
            
        s1, s2 = iax*2, (iax+1)*2
        if s2>tsmps.shape[1]:
            s2 = tsmps.shape[1]
        smps = tsmps[:,s1:s2]

        for h in range(smps.shape[1]):
            myboxplot(ax2, smps[:,h], h, 0.8, 'k')   # boxplot
        
        if True:
            # ymin, ymax = ax2.get_ylim()
            # if iax==0:
            #     fact = ymax-ymin
            # else:
            #     diff = ymax-ymin
            #     ax2.set_ylim(ymin-(fact-diff)/2, ymax+(fact-diff)/2)
            if iax==0:
                ylabs = [r'Astrometric offset [mas or mas/yr]', r'Proper motion offset [mas/yr]',
                         r'Parallactic offset [mas]']
                ax2.set_ylabel(ylabs[iax],fontsize=18)
            ax2.tick_params( which='both', left='off', right=False, bottom=True, top=False,labelsize=14)
            ax2.tick_params(axis='x', pad=0, labelsize=18, rotation=0,)
            if iax==4:
                ax2.legend(fontsize=14)
                
        if True:
            labs = [r'${\Delta}\alpha_{\ast 2}$', r'$\Delta \alpha_{\ast 3}$', 
                    r'$\Delta\delta_{2}$', r'$\Delta \delta_{3}$',
                    r'$\mu_{\alpha 2}$', r'$\mu_{\alpha 3}$', 
                    r'$\mu_{\delta2}$', r'$\mu_{\delta3}$',
                    r'$\varpi_{2}$', r'$\varpi_{3}$']
            ax2.set_xticks(list(range(smps.shape[1])))
            ax2.set_xticklabels(labs[s1:s2])        
    plt.show()
    #plt.savefig('HD222237_boxplot.pdf',bbox_inches='tight',transparent=False)


def plot_ETV(tab, out, calculate_tauT, use_starfn=None):

    fig = plt.figure(figsize=(8.8, 7.8),dpi=120)
    ax1 = fig.add_axes((0.15, 0.63, 0.8, 0.32))
    ax2 = fig.add_axes((0.15, 0.3, 0.8, 0.32),sharex=ax1)
    ax3 = fig.add_axes((0.15, 0.1, 0.8, 0.19),sharex=ax2)
        
    #sys.exit()
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(direction='in', which='both', left=True, right=True, bottom=True, top=False,labelsize=14)
        if ax != ax1:
            ax.axhline(0,ls='--',lw=1,color='grey',zorder=0)
    
    ax3.set_xlabel('BJD-2450000',fontsize=18)#, weight='bold'
    ax3.set_ylabel('Res. [min]', fontsize=18)
    ax2.set_ylabel(r'O-C$_1$ [min]', fontsize=18)    
    ax1.set_ylabel('O-C [min]', fontsize=18)  
    
    Np = out['nplanet']
    
    if use_starfn is not None:
        if os.path.exists(use_starfn):
            print('\nload start file:',use_starfn)
            init, pname = [], []
            f = open(use_starfn)
            for line in f:
                if line[0]=='#':continue
                lst = line.strip().split()
                pname.append(lst[0])
                init.append(float(lst[1]))
            f.close()
            names, pars = pname, np.asarray(init)
        else:
            print('\nstar file not found:',use_starfn)
            sys.exit()
    else:
        lnp = tab['logpost'].values
        best_fit = np.argmax(lnp)                # Maximum lnL index
        par_opt = tab.iloc[[best_fit]].values[0]
    
        names = list(tab.keys())
        npar = names.index('logpost') # number of free parameters, the order should be same as MCMC pars
        pars = np.copy(par_opt[:npar])

    
    norbpar = out['norbpar']
    pps = np.copy(pars)
    new_index = out['new_index']
    #print(pps)
    
    for i in range(Np):  # reconstruct the free parameters
        pday = np.e**pps[i*norbpar]  # logP->P
        pps[i*norbpar] = pday
        pps[i*norbpar+1] = np.e**pps[i*norbpar+1] # logK->K
        esino, ecoso = pps[i*norbpar+2], pps[i*norbpar+3]
        ecc = esino**2 + ecoso**2
        pps[i*norbpar+2] = ecc
        pps[i*norbpar+3] = atan2(esino, ecoso)#pi2+w if w<0 else w
    
    # # fake ETV data
    # bjd = np.array([ii for jj in out['rv_jds'] for ii in jj])
    # t_ttv = np.linspace(bjd.min()-100,bjd.max()-1000,2000)
    # m = np.int32(np.random.uniform(1,len(t_ttv),50))
    # t_ttv = t_ttv[m]
    # t_ttv = np.sort(t_ttv)
    # tauT0 = calculate_tauT(pps,tt=t_ttv,out=out)
    # edt = np.random.randn(len(t_ttv))*0.001
    # dt0 = (t_ttv-tauT0) + edt #+ (t_ttv-t_ttv[0])/365.25*0.002 + ((t_ttv-t_ttv[0])/365.25)**2*-0.002 +0.008#day 
    # plt.plot(t_ttv, dt0, 'o')
    # flat_samples = np.array([t_ttv, dt0, np.abs(edt)]).T
    # df0 = pd.DataFrame(flat_samples, columns=['BJD','dt','edt'])
    # df0.to_csv('fake_ETV.dat',sep=' ',mode='w',index=False)
    # sys.exit()
    
    for timetype in out['timing'].keys():
        bjd = out['timing'][timetype]['data'][:,0]
        t_ttv = np.linspace(bjd.min()-100,bjd.max()+100,1000)  # simulated time
        tauT = calculate_tauT(pps,tt=t_ttv,out=out)
        tauT0 = calculate_tauT(pps,tt=bjd,out=out)
        for ip in out['timing'][timetype]['planet_index']: # 1, 2, 3 ...
            ip = int(ip)
            index = ip==out['timing'][timetype]['data'][:,-1] # planet id
            if re.search('pulsation|eb',timetype):
                dt = (t_ttv-tauT)#day
                dt0 = (bjd-tauT0)#day
                ## baseline delay model is a parabola: dti=ti+gi*dt.yr+hi*dt.yr^2
                if ('pulsation' in timetype) or ('eb' in timetype):
                    dt_yr = (t_ttv-out['timing_T0'])/365.25
                    dt_yr0 = (bjd-out['timing_T0'])/365.25
                    it = new_index.index('ti')
                    ti, gi, hi, ci = pps[it], 0, 0, 0
                    if out['timing_model']=='linear': gi = pps[it+1]
                    elif out['timing_model']=='quadric': gi, hi = pps[it+1], pps[it+2]
                    elif out['timing_model']=='cubic': gi, hi, ci = pps[it+1], pps[it+2], pps[it+3]
                    dt_model = dt*3600*24#+ti+gi*dt_yr+hi*dt_yr**2 # sec
                    dt_model2 = dt*3600*24+ti+gi*dt_yr+hi*dt_yr**2+ ci*dt_yr**3# sec
                    obs_model = dt0*3600*24+ti+gi*dt_yr0+hi*dt_yr0**2 + ci*dt_yr0**3   # sec
                    obs_mod = ti+gi*dt_yr0+hi*dt_yr0**2 + ci*dt_yr0**3
                else:  # not use
                    t0 = pps[new_index.index('t0')]
                    dt_model = dt+t0
                ddt, edt = out['timing'][timetype]['data'][:,1],out['timing'][timetype]['data'][:,2]
                ax1.errorbar(bjd[index]-2450000, (ddt[index])/60, yerr=edt[index]/60,fmt='o', ecolor='black',capsize=3, 
                         capthick=1, elinewidth=1.2,ms=10,mec='k',)
                ax1.plot(t_ttv-2450000, dt_model2/60, 'k-', rasterized=False, lw=2.5, zorder=99, )
                ax1.plot(t_ttv-2450000, (ti+1+gi*dt_yr+hi*dt_yr**2+ ci*dt_yr**3)/60, 'k--', rasterized=False, lw=2, zorder=99, alpha=0.7) 
                
                ax2.errorbar(bjd[index]-2450000, (ddt[index]-obs_mod)/60, yerr=edt[index]/60,fmt='o', ecolor='black',capsize=3, 
                         capthick=1, elinewidth=1.2,ms=10,mec='k',)
                ax2.plot(t_ttv-2450000, dt_model/60, 'k-', rasterized=False, lw=2.5, zorder=99, )
                res = ddt[index]/60-obs_model/60 # min
                ax3.errorbar(bjd[index]-2450000, res, yerr=edt[index]/60,fmt='o', ecolor='black',capsize=3, 
                         capthick=1, elinewidth=1.2,ms=10,mec='k',)
                print('RMS:',np.std(res))
                #ax1.legend()
                #plt.savefig(savefix+'{}_{}_{}_{:03d}.png'.format(target,timetype,timing_model,v+1))    
            else:
                ts = bjd[index] #rvp['tt'][index]
                lte  =  (ts-tauT0[index])*24*60#min
                lte_sim  =  (t_ttv-tauT)*24*60#min
                p = pps[0+(ip-1)*norbpar] #day
                dt = ts-out['ttv0']  # min(t_rv & t_ttv)
                if re.search('occult',timetype):#(grepl('occult',n2)) dt <- dt+p/2
                    dt = dt+p/2
                ttv = (dt-np.round(dt/p)*p)*24*60#min
                ettv = out['timing'][timetype]['data'][:,1][index]*24*60#min
                t0 = pps[new_index.index('t0')]
                #ll_time += np.sum(dnorm(ttv, lte+t0, ettv, log=True))

                ax2.plot(ts, ttv, 'o', label='{} planet {}'.format(timetype, ip))
                ax2.plot(t_ttv, (lte_sim+t0), '-')
                ax2.legend()
                #plt.savefig(savefix+'{}_{}_fitting_{:03d}.png'.format(target,timetype,v+1))  
    # years on upper axis
    axyrs = ax1.twiny()
    xl = np.array(list(ax1.get_xlim())) + 2450000
    decimalyear = Time(xl, format='jd', scale='utc').decimalyear
    axyrs.get_xaxis().get_major_formatter().set_useOffset(False)
    axyrs.set_xlim(*decimalyear)
    axyrs.set_xlabel('Year', fontsize=14)#fontweight='bold',
    axyrs.xaxis.set_minor_locator(AutoMinorLocator())
    axyrs.tick_params(direction='in', which='both',labelsize=13)
    plt.locator_params(axis='x', nbins=8)
    align_ylabels(fig, [ax1, ax3], pad=0.001)
    align_ylabels(fig, [ax2, ax3], pad=0.001)
    
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)


import scipy.stats as st

def chi2_to_sigma(chi2, dof):
    """
    把 χ² 值转换成等效的高斯 σ 显著性
    
    参数
    ----
    chi2 : float
        计算得到的 χ² 值
    dof  : int
        自由度（proper motion 二维异常就填 2）
    
    返回
    ----
    float
        等效的高斯 σ（如 3.0 表示 3σ）
    """
    # 右尾概率
    p_right = st.chi2.sf(chi2, dof)          # survival function ≡ 1 - CDF
    # 转成双侧正态尾概率（因为 χ² 检验天生是双侧）
    p_two_tail = 2.0 * p_right
    # 单侧正态分位数
    return st.norm.isf(p_right)              # 也可以写 

def calc_hip_gaia_PMa_SNR(Hipid, HGCAfile='/home/xiaogy/orvara/HGCA_vEDR3.fits'):
    #Hipid = 88414
    
    fn = HGCAfile#'/home/xiaogy/orvara/HGCA_vEDR3.fits'
    hdu = fits.open(fn)
    
    hip_id = hdu[1].data['hip_id'] 
    ind = np.argmax(hip_id==Hipid)
    
    gaia_source_id = hdu[1].data['gaia_source_id'][ind] 
    gaia_ra = hdu[1].data['gaia_ra'][ind]
    gaia_dec = hdu[1].data['gaia_dec'][ind] 
    radial_velocity = hdu[1].data['radial_velocity'][ind]
    radial_velocity_error = hdu[1].data['radial_velocity_error'][ind]
    radial_velocity_source = hdu[1].data['radial_velocity_source'][ind]
    parallax_gaia = hdu[1].data['parallax_gaia'][ind]
    parallax_gaia_error = hdu[1].data['parallax_gaia_error'][ind]
    pmra_gaia = hdu[1].data['pmra_gaia'][ind]
    pmdec_gaia = hdu[1].data['pmdec_gaia'][ind]
    pmra_gaia_error = hdu[1].data['pmra_gaia_error'][ind]
    pmdec_gaia_error = hdu[1].data['pmdec_gaia_error'][ind]
    pmra_pmdec_gaia = hdu[1].data['pmra_pmdec_gaia'][ind]
    pmra_hg = hdu[1].data['pmra_hg'][ind]
    pmdec_hg = hdu[1].data['pmdec_hg'][ind]
    pmra_hg_error = hdu[1].data['pmra_hg_error'][ind]
    pmdec_hg_error = hdu[1].data['pmdec_hg_error'][ind]
    pmra_pmdec_hg = hdu[1].data['pmra_pmdec_hg'][ind]
    pmra_hip = hdu[1].data['pmra_hip'][ind]
    pmdec_hip = hdu[1].data['pmdec_hip'][ind]
    pmra_hip_error = hdu[1].data['pmra_hip_error'][ind]
    pmdec_hip_error = hdu[1].data['pmdec_hip_error'][ind]
    pmra_pmdec_hip = hdu[1].data['pmra_pmdec_hip'][ind]
    epoch_ra_gaia = hdu[1].data['epoch_ra_gaia'][ind]
    epoch_dec_gaia = hdu[1].data['epoch_dec_gaia'][ind] 
    epoch_ra_hip = hdu[1].data['epoch_ra_hip'][ind] 
    epoch_dec_hip = hdu[1].data['epoch_dec_hip'][ind] 
    chisq = hdu[1].data['chisq'][ind]
    
    ############################## gaia #####
    diff_pmra_gaia = (pmra_gaia - pmra_hg)
    diff_pmra_gaia_err = np.sqrt(pmra_gaia_error**2 + pmra_hg_error**2)
    
    diff_pmdec_gaia = (pmdec_gaia - pmdec_hg)
    diff_pmdec_gaia_err = np.sqrt(pmdec_gaia_error**2 + pmdec_hg_error**2)
    
    cov_hg = np.array([[pmra_hg_error**2, pmra_pmdec_hg*pmra_hg_error*pmdec_hg_error],
                     [pmra_pmdec_hg*pmra_hg_error*pmdec_hg_error,pmdec_hg_error**2]])
    
    cov_gaia = np.array([[pmra_gaia_error**2, pmra_pmdec_gaia*pmra_gaia_error*pmdec_gaia_error],
                     [pmra_pmdec_gaia*pmra_gaia_error*pmdec_gaia_error,pmdec_gaia_error**2]])
    
    cov_diff = cov_hg + cov_gaia
    
    vector_diff = np.array([diff_pmra_gaia, diff_pmdec_gaia])
    inv_cov = np.linalg.inv(cov_diff).astype(float)
    SNR_gaia = np.sqrt(vector_diff@inv_cov@vector_diff.T)
    
    vra, vdec = diff_pmra_gaia/parallax_gaia*4740.47, diff_pmdec_gaia/parallax_gaia*4740.47
    vtot = np.sqrt(vra**2+vdec**2)
    print('Gaia PMa (mas/yr):',diff_pmra_gaia,diff_pmra_gaia_err,diff_pmdec_gaia,diff_pmdec_gaia_err)
    print('Gaia delta_v (m/s):',vra, vdec, np.sqrt(vra**2+vdec**2))
    print('Gaia PMa SNR:', SNR_gaia,'chi2:',chisq, 'chi2->sigma:',chi2_to_sigma(chisq, 2))
    
    msun = 1.9885e30*1 # kg
    au = 149597870e3  # m
    sau = np.linspace(1, 100, 10000)
    r = 5 * au
    G = 6.67e-11
    m2 = np.sqrt(msun/G*r*vtot**2)/1.9885e30
    
    print('m2:',m2)
    # plt.plot(sau, m2)
    # plt.xscale('log')
    # plt.yscale('log')
    ############################## hip #####
    diff_pmra_hip = (pmra_hip - pmra_hg)
    diff_pmra_hip_err = np.sqrt(pmra_hip_error**2 + pmra_hg_error**2)
    
    diff_pmdec_hip = (pmdec_hip - pmdec_hg)
    diff_pmdec_hip_err = np.sqrt(pmdec_hip_error**2 + pmdec_hg_error**2)
    
    cov_hg = np.array([[pmra_hg_error**2, pmra_pmdec_hg*pmra_hg_error*pmdec_hg_error],
                     [pmra_pmdec_hg*pmra_hg_error*pmdec_hg_error,pmdec_hg_error**2]])
    
    cov_hip = np.array([[pmra_hip_error**2, pmra_pmdec_hip*pmra_hip_error*pmdec_hip_error],
                     [pmra_pmdec_hip*pmra_hip_error*pmdec_hip_error,pmdec_hip_error**2]])
    
    cov_diff = cov_hg + cov_hip
    
    vector_diff = np.array([diff_pmra_hip, diff_pmdec_hip])
    inv_cov = np.linalg.inv(cov_diff).astype(float)
    SNR_hip = np.sqrt(vector_diff@inv_cov@vector_diff.T)
    vra, vdec = diff_pmra_hip/parallax_gaia*4740.47, diff_pmdec_hip/parallax_gaia*4740.47
    vtot = np.sqrt(vra**2+vdec**2)
    
    print('-'*30)
    print('hip PMa (mas/yr):',diff_pmra_hip,diff_pmra_hip_err,diff_pmdec_hip,diff_pmdec_hip_err)
    print('Gaia delta_v (m/s):',vra, vdec, np.sqrt(vra**2+vdec**2))
    print('hip PMa SNR:', SNR_hip)
    m2 = np.sqrt(msun/G*r*vtot**2)/1.9885e30
    print('m2:',m2)
