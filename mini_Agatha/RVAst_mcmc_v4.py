#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 22:05:25 2024

add ptmcmc to sampling the posterior
python version of agatha of Feng
not include ARMA model for RV 
add direct imaging model
没有边缘化 RV offsets
Gaia jitter set to 0
use to fit WD1202-232
@author: xiaogy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import corner
from matplotlib import rcParams, gridspec
from matplotlib.ticker import MaxNLocator
#from astropy.time import Time
import copy
import random
import glob
import seaborn as sns
from configparser import ConfigParser
from math import sin, cos, tan, sqrt
#from numpy import sin, cos, tan
from scipy.stats import multivariate_normal
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator
#from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
from scipy import stats, signal
from ptemcee import Sampler as PTSampler
import time
import re
import os
import scipy
#from test_orbit import kep_mt2 as agatha_kep

start_time = time.time()

config = {
    "font.family":'serif', # sans-serif/serif/cursive/fantasy/monospace
    #"font.size": 14, # medium/large/small
    'font.style':'normal', # normal/italic/oblique
    'font.weight':'normal', # bold
    "mathtext.fontset":'cm',# 'cm' (Computer Modern)
    "font.serif": ['STIXGeneral'], # 'Simsun'宋体  STIXGeneral  cmr10
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

def kep_mt2(m, e):  #solve for keplerian (not expensive)
    tol = 1e-8
    E0 = m.copy()
    Ntt = 1000
    for k in range(Ntt):
        E1 = E0-(E0-e*np.sin(E0)-m)/(np.sqrt((1-e*np.cos(E0))**2-(E0-e*np.sin(E0)-m)*(e*np.sin(E0))))
        if(np.all(np.abs(E1-E0)<tol)): 
            break
        E0 = E1
    if(k==Ntt):
        print('Keplerian solver does not converge!\n')
        print('length(which(abs(E1-E0)>tol))=',np.sum(np.abs(E1-E0)>tol),'\n')
    return E1

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
    #e1 = kep_mt2(m, e)
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


def construct_cov(catas): # catalog astrometry: hip, GDR2, GDR3
    covs = []
    keys = ['ra','dec','parallax','pmra','pmdec']
    for i in range(len(catas)):
        icov = np.zeros((5,5),dtype=float)
        for j in range(5):
            for k in range(j,5):
                if keys[j]==keys[k]:
                    key = keys[j]+'_'+'error'
                    value = catas[key].iloc[i]**2
                    icov[j, k] = value
                else:
                    key = keys[j]+'_'+keys[k]+'_'+'cov'
                    value = catas[key].iloc[i]
                    icov[j, k] = icov[k, j] = value
                    
        covs.append(icov)
        #
    return np.array(covs,dtype=float)

def extract_par(pars_kep,out=None):
    keys = ['Pd','K','e','omega','Mo','Inc','Omega']    
    #names = pars_kep.columns.to_list()
    nplanet = out['nplanet']#np.sum(['omega' in i for i in names])
    pp = dict()
    for key in keys:
        pp[key] = []
        for i in range(nplanet):
            if key=='per':
                pp[key].append(np.exp(pars_kep[key+'%d'%(i+1)].values[0]))
            else:
                pp[key].append(pars_kep[key+'%d'%(i+1)].values[0])
    pp = pd.DataFrame(pp)
    return pp

####Note that bl2xyz return coordinates in the heliocentric frame and the x axis point to the GC, yaxis point to the rotation of the Galaxy
def bl2xyz(b_rad,l_rad):  #dec,ra
    x = cos(b_rad)*cos(l_rad)
    y = cos(b_rad)*sin(l_rad)
    z = sin(b_rad)
    return np.array([x,y,z])

def xyz2bl_vec(x,y,z):
  b = np.arctan2(z,np.sqrt(x**2+y**2))
  ind = b>pi/2
  if(np.sum(ind)>0):
      b[ind] = b[ind]-pi
  l = np.arctan2(y,x)%(2*pi)
  return b,l

def obs_lin_prop(obs,t,PA=True):
    ##whether to consider perspective acceleration
    ##t is in unit of day
    ##obs: ra(deg), dec(deg),plx(mas),pmra(mas/yr),pmdec(mas/yr)
    #kpcmyr2auyr = 1e3*206265/1e6
    
    pc2au = 206265
    #kpcmyr2kms = kpcmyr2auyr*auyr2kms
    ra = (obs['ra'])/180*pi     # deg to rad
    dec = (obs['dec'])/180*pi   # deg to rad
    plx = (obs['parallax'])
    pmra, pmdec = obs['pmra'], obs['pmdec']
    rv = obs['radial_velocity']
    
    if(PA):
        #### obs-> initial state: propagation observables to states
        d = 1/plx#kpc
        x, y, z = bl2xyz(dec,ra)*d*1e3  #pc

        vde = pmdec*d
        vra = pmra*d
        #vp = np.sqrt(vra**2+vde**2)
        vr = rv/auyr2kms#au/yr
        vx_equ = vr*cos(dec)*cos(ra)-vde*sin(dec)*cos(ra)-vra*sin(ra)##note: vr is positive if the star is moving away from the Sun
        vy_equ = vr*cos(dec)*sin(ra)-vde*sin(dec)*sin(ra)+vra*cos(ra)
        vz_equ = vr*sin(dec)+vde*cos(dec)
        x1 = x+vx_equ*t/365.25/pc2au
        y1 = y+vy_equ*t/365.25/pc2au
        z1 = z+vz_equ*t/365.25/pc2au
        
        ### propagation: convert time-varying states back to observables
        dec1_rad, ra1_rad = xyz2bl_vec(x1,y1,z1)  # rad
        d1 = np.sqrt(x1**2+y1**2+z1**2)*1e-3#kpc
        
        ra1 = ra1_rad*180/pi#deg
        dec1 = dec1_rad*180/pi
        ###state -> obs: velocity to pm
        #vequ = array(NA,dim=c(length(t),3))
        vequ = np.repeat(np.nan,len(t)*3).reshape(len(t),3)
        for j in range(len(t)):
          rotz = np.array([cos(ra1_rad[j]),sin(ra1_rad[j]),0.0,-sin(ra1_rad[j]),cos(ra1_rad[j]),0.0,0.0,0.0,1.0]).reshape(3,3)
          roty = np.array([cos(dec1_rad[j]),0.0,sin(dec1_rad[j]),0.0,1.0,0.0,-sin(dec1_rad[j]),0.0,cos(dec1_rad[j])]).reshape(3,3)
          ##rotz = matrix(data=c(cos(ra1.rad[j]),sin(ra1.rad[j]),0.0,-sin(ra1.rad[j]),cos(ra1.rad[j]),0.0,0.0,0.0,1.0),nrow=3,ncol=3,byrow=TRUE)#o-xyz -> o-x'y'z'
          #roty = matrix(data=c(cos(dec1.rad[j]),0.0,sin(dec1.rad[j]),0.0,1.0,0.0,-sin(dec1.rad[j]),0.0,cos(dec1.rad[j])),nrow=3,ncol=3,byrow=TRUE)
          vv = np.array([vx_equ,vy_equ,vz_equ]).reshape(3,1)
          #vequ[j,] = roty%*%rotz%*%as.numeric(c(vx.equ,vy.equ,vz.equ))
          vequ[j,:] = (roty@rotz@vv)[:,0]
        
        pmra1 = vequ[:,1]/d1#mas/yr
        pmdec1 = vequ[:,2]/d1#mas/yr
        rv1 = vequ[:,0]*auyr2kms
        out0 = np.array([ra1,dec1,1/d1,pmra1,pmdec1,rv1]).T
    else:
        decs = dec+pmdec*t/365.25/206265e3#rad
        ras = ra+pmra*t/365.25/np.cos(decs)/206265e3#rad
        out0 = np.array([ras*180/pi,decs*180/pi,np.repeat(plx,len(t)),np.repeat(pmra,len(t)),np.repeat(pmdec,len(t)),np.repeat(rv,len(t))]).T
        
    colnames = ['ra','dec','parallax','pmra','pmdec','radial_velocity']
    df0 = pd.DataFrame(out0, columns=colnames)
    return df0


def astrometry_bary(pars_kep,tt=None,Ein=None,pa=False,out=None):
    ### This is to give both absolute and relative astrometry if the absolute astrometry data is given
    data_astrometry = out['astrometry']    # catalog astrometry
    if tt is None:
        tt = out['astrometry']['ref_epoch'].values           # catalog 1astrometryst column: 'ref_epoch' hip2 dr2 dr3

    #dt = data_astrometry.iloc[out['iref'],0]-tmin#epoch offset between astrometry reference point and the RV reference point
    DT = tt - data_astrometry.iloc[out['iref'],0]   #relative to astrometry reference point (DR3)

    dra = ddec = dpmra = dpmdec = 0
    dplx = drv = 0
    names = pars_kep.columns.to_list()
    if('dra' in names): dra = dra+pars_kep['dra'].values[0]#as; note that this dra0 is ra
    if('ddec' in names): ddec = ddec+pars_kep['ddec'].values[0]
    if('dplx' in names): dplx = dplx+pars_kep['dplx'].values[0]
    if('dpmra' in names): dpmra = dpmra+pars_kep['dpmra'].values[0]#mas/yr
    if('dpmdec' in names): dpmdec = dpmdec+pars_kep['dpmdec'].values[0]
    if('drv' in names): drv = drv+pars_kep['drv'].values[0]
    
    ### model parallax and mean proper motion; only assumption: constant heliocentric velocity; reference Gaia epoch
    obs = data_astrometry[['ra','dec','parallax','pmra','pmdec','radial_velocity']].iloc[out['iref']]
    ### subtract the offset position and PM to get the initial condition for the barycentric motion (tDR3)
    obs['ra'] = obs['ra']-dra/3.6e6/np.cos(obs['dec']/180*pi)#mas to deg
    obs['dec'] = obs['dec']-ddec/3.6e6   # mas to deg 
    obs['parallax'] = obs['parallax']-dplx#mas
    obs['pmra'] = obs['pmra']-dpmra  #mas/yr
    obs['pmdec'] = obs['pmdec']-dpmdec
    obs['radial_velocity'] = obs['radial_velocity']-drv#km/s
    
    ##propagation barycentric observables
    return obs_lin_prop(obs,DT)


def k2m(K,P,e,Ms,Inc=None,Niter=100,tol=1e-6,more=False): 
###If Inc is given, k2m will determine absolute mass                                                                                
###If Inc is not given, k2m will approximately give msini if m is small                                                             
    Me2s = 3.003e-6#Earth mass in solar unit                                                                                       
    Mj2s = 1/1048
    if Inc is None:
        sinI = 1
    else:
        sinI = np.sin(Inc)
    K = K/1e3/4.74047#from m/s to au/yr                                                                                            
    P = P/365.25#yr                                                                                                                

    a1 = (K/sinI)**2/(4*pi**2)*(1-e**2)*P**(2/3)
    mp0 = mp = np.sqrt(a1*Ms**(4/3))
    for j in range(Niter):
        mp = np.sqrt(a1*(Ms+mp)**(4/3))
        if np.all(np.abs(mp-mp0)/mp0<tol):
            break
        mp0 = mp
    Mpj = mp/Mj2s
    Mpe = Mpj*Mj2s/Me2s
    if more:
        a = (P**2*(Ms+mp))**(1/3)#au                                                                                          as = \
        as_ = a*mp/(Ms+mp)                                                                                                                        
        ap = a*Ms/(Ms+mp)
        return {'mj':Mpj,'me':Mpe,'ms':mp,'ap':ap,'a':a,'as':as_}
    else:
        return {'mj':Mpj,'me':Mpe,'ms':mp}  # units

def calc_astro(K,P,e,inc,omega,Omega,E,plx,Mstar,pars,eta=np.nan,state=False,comp=1):
    #### calculate orbital motion of 1 planets
    alpha0 = K/sin(inc)/1e3/4.74047#au/yr
    beta0 = P/365.25*(K*1e-3/4.74047)*sqrt(1-e**2)/(2*pi)/sin(inc)#au

    T = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))

    alpha = alpha0*plx#proper motion in mas/yr
    ##semi-major axis is the astrometric signature in micro-arcsec
    A = cos(Omega)*cos(omega)-sin(Omega)*sin(omega)*cos(inc)
    B = sin(Omega)*cos(omega)+cos(Omega)*sin(omega)*cos(inc)
    F = -cos(Omega)*sin(omega)-sin(Omega)*cos(omega)*cos(inc)
    G = -sin(Omega)*sin(omega)+cos(Omega)*cos(omega)*cos(inc)
    C = sin(omega)*sin(inc)
    H = cos(omega)*sin(inc)

    Vx = -np.sin(T)
    Vy = np.cos(T)+e
    ###calculate POS
    X = np.cos(E)-e
    Y = np.sqrt(1-e**2)*np.sin(E)

    beta = beta0*plx#mas
    ##    beta = (P*3600*24)*K*sqrt(1-e**2)/(2*pi)*plx/sin(inc)*6.68459e-12#mas
    raP = beta*(B*X+G*Y)
    decP = beta*(A*X+F*Y)
    plxP = -beta*(C*X+H*Y)*plx/206265e3#parallax change
    pmraP = alpha*(B*Vx+G*Vy)
    pmdecP = alpha*(A*Vx+F*Vy)
    ##    rvP.epoch = alpha*(C*Vx+H*Vy)
    mp = k2m(K,P,e,Mstar,Inc=inc)['ms']  # in unit of solar mass
    if np.isnan(eta):
        names = pars.columns.to_list()
        if 'eta' in names:
            eta = pars['eta'].values[0]
        else:
            eta = 0#calc_eta(Mstar,mp)  xiao
    xi = 1/(eta+1)
    if(comp==2): 
        xi = -Mstar/mp
    raP = raP*xi      # photocentric motion; xiao
    decP = decP*xi
    plxP = plxP*xi
    pmraP = pmraP*xi
    pmdecP = pmdecP*xi
    if False:
        print('eta=',eta,';xi=',xi,';mp=',mp,';Mstar=',Mstar,'\n')
    rv = alpha0*(C*Vx+H*Vy)#km/s
    if state:
        BT = np.array([raP/plx,decP/plx,beta0*(C*X+H*Y),pmraP/plx,pmdecP/plx,rv])#au,au/yr
        return BT 
    else:
        return np.array([raP,decP,plxP,pmraP,pmdecP,rv*4.74047])  #mas mas/yr


def astrometry_epoch(pars_kep,tt=None,pp=None,eta=0,barycenter=None,band='G',comp=1,iplanet=None,out=None):
    # reflex motion
    if pp is None:
        pp = extract_par(pars_kep,out=out)
    #ra.planet = dec.planet = pmra.planet = pmdec.planet = 0
    names = pars_kep.columns.to_list()
    Np_kep = out['nplanet']#np.sum(['omega' in i for i in names])
    
    if 'Mstar' in names:
        Mstar = pars_kep['Mstar'].values[0]
    else:
        Mstar = out['Mstar']
    plx = out['plx']
    if 'dplx' in names: 
        plx = plx-pars_kep['dplx'].values[0]
    drv_reflex = dpmdec_reflex = dpmra_reflex = dplx_reflex = ddec_reflex = dra_reflex = 0
    if Np_kep>0:
        for j in range(Np_kep):
            if iplanet is not None:
                if j != (iplanet-1):continue
            ms = (pp['Mo'][j]+2*pi*(tt-out['tmin'])/pp['Pd'][j])%(2*pi)  # Mean anomaly
            #E = agatha_kep(ms,pp['e'][j])
            #E = kepler(ms, np.repeat(pp['e'][j],len(ms)))
            E = kep_mt2(ms,pp['e'][j])                             # eccentric anomaly
            tmp = calc_astro(pp['K'][j],pp['Pd'][j],pp['e'][j],pp['Inc'][j],pp['omega'][j],pp['Omega'][j],E,plx=plx,Mstar=Mstar,pars=pars_kep,eta=eta,comp=comp)
            dra_reflex = dra_reflex+tmp[0]
            ddec_reflex = ddec_reflex+tmp[1]
            dplx_reflex = dplx_reflex+tmp[2]
            dpmra_reflex = dpmra_reflex+tmp[3]
            dpmdec_reflex = dpmdec_reflex+tmp[4]
            drv_reflex = drv_reflex+tmp[5]#km/s
    out0 = np.array([dra_reflex,ddec_reflex,dplx_reflex,dpmra_reflex,dpmdec_reflex,drv_reflex]).T
    colnames = ['dra','ddec','dplx','dpmra','dpmdec','drv']
    df0 = pd.DataFrame(out0, columns=colnames)
    #list(epoch=data.frame(dra=dra.reflex,ddec=ddec.reflex,dplx=dplx.reflex,dpmra=dpmra.reflex,dpmdec=dpmdec.reflex,drv=drv.reflex))
    return {'epoch':df0}

def astrometry_rel(pars_kep,pp=None,out=None):
    # i-th planet motion relative to host star
    if pp is None:
        pp = extract_par(pars_kep,out=out)
    names = pars_kep.columns.to_list()
    
    if 'Mstar' in names:
        Mstar = pars_kep['Mstar'].values[0]  
    else:
        Mstar = out['Mstar'] # if RV only, then do'not sample Mstar using MCMC
    plx = out['plx']
    if 'dplx' in names: 
        plx = plx-pars_kep['dplx'].values[0]
    tmp = dict()
    rel_dra = np.zeros_like(out['relAst']['rel_JD'])
    rel_ddec = np.zeros_like(out['relAst']['rel_JD'])
    tt = out['relAst']['rel_JD']
    for ip in list(set(out['relAst']['rel_iplanet'])):
        mm = ip==out['relAst']['rel_iplanet']
        j = ip-1
        Omega, omega, inc, e = pp['Omega'][j], pp['omega'][j], pp['Inc'][j], pp['e'][j]
        K, P = pp['K'][j], pp['Pd'][j]
        ms = (pp['Mo'][j]+2*pi*(tt[mm]-out['tmin'])/P)%(2*pi)
        #E = kepler(ms, np.repeat(e,len(ms)))
        E = kep_mt2(ms,pp['e'][j]) 
        beta0 = P/365.25*(K*1e-3/4.74047)*np.sqrt(1-e**2)/(2*pi)/sin(inc)
        A = cos(Omega)*cos(omega)-sin(Omega)*sin(omega)*cos(inc)
        B = sin(Omega)*cos(omega)+cos(Omega)*sin(omega)*cos(inc)
        F = -cos(Omega)*sin(omega)-sin(Omega)*cos(omega)*cos(inc)
        G = -sin(Omega)*sin(omega)+cos(Omega)*cos(omega)*cos(inc)
        beta = beta0*plx
        X = np.cos(E)-e
        Y = np.sqrt(1-e**2)*np.sin(E)
        mp = k2m(K,P,e,Mstar,Inc=inc)['ms']
        xis = -mp/(Mstar+mp)
        rel_dra[mm] = beta*(B*X+G*Y)/xis#mas
        rel_ddec[mm] = beta*(A*X+F*Y)/xis#mas
    if out['relAst_type'] == 'Sep_PA':
        pa_mc = np.arctan2(rel_dra,rel_ddec)*180/pi#deg
        sep_mc = np.sqrt(rel_dra**2+rel_ddec**2)*1e-3#arcsec
    elif out['relAst_type'] == 'Dra_Ddec':
        sep_mc, pa_mc = rel_dra/1000, rel_ddec/1000 #mas->"
    else:
        print('\nErr: Unkonw relAst type!')
        sys.exit()
    tmp['res_sep'] = sep_mc-out['relAst']['rel_sep']
    tmp['res_PA'] = pa_mc-out['relAst']['rel_PA']

    tmp['cov'] = []
    for i in range(len(tt)):
        esep, epa, corr = out['relAst']['rel_sep_err'][i], out['relAst']['rel_PA_err'][i], out['relAst']['rel_corr'][i]
        cov = np.array([esep**2, corr*esep*epa, corr*esep*epa, epa**2]).reshape(2,2)
        tmp['cov'].append(cov)

    return tmp

#### model of astrometry
def astrometry_kepler(pars_kep,tt=None,Pmin=0,out=None):
    #'barycenter'    'epoch':hip2     'cats':residual dr2 dr3
    pp = extract_par(pars_kep,out=out)
    if(tt is None):
        sim = False
    else:
        sim = True
    tmp = dict()
    
    tmp['barycenter'] = None
    if(len(out['astrometry'])>0):  # 3 barycenter at hip2 dr2 dr3
        tmp['barycenter'] = astrometry_bary(pars_kep=pars_kep,tt=tt,out=out)       
        
    if 'relAst' in out.keys():     # direct imaging
        tmp['relAst'] = astrometry_rel(pars_kep,pp=pp,out=out)

    tmp['epoch'] = dict()
    if(len(out['data_epoch'])>0):
        i =  out['ins_epoch']   # 'hip2'
        if(not sim):
            tt = out['data_epoch']['BJD'].values     # hip2 abscissa
        if('hip' in i):band = 'Hp'
        else:band = 'G'
        tmp['epoch'][i] = astrometry_epoch(pars_kep,pp=pp,tt=tt,band=band,out=out)['epoch']
        
    if(len(out['gost'])>0):
        reflex = astrometry_epoch(pars_kep,pp=pp,tt=out['gost']['BJD'].values,band='G',out=out)['epoch']
        
        obs0 = tmp['barycenter'].iloc[out['iref']]  # barycenter parameters
        bary = obs_lin_prop(obs0,t=out['gost']['BJD'].values-out['astrometry'].iloc[out['iref']]['ref_epoch'],PA=False)

        dec = bary['dec'].values+reflex['ddec'].values/3.6e6#deg
        dra = (bary['ra'].values-out['astrometry'].iloc[out['iref']]['ra'])*np.cos(dec/180*pi)*3.6e6+reflex['dra'].values#mas
        ddec = (dec-out['astrometry'].iloc[out['iref']]['dec'])*3.6e6#mas
        gabs = dra*np.sin(out['gost']['psi'].values)+ddec*np.cos(out['gost']['psi'].values)+(bary['parallax'].values+reflex['dplx'].values)*out['gost']['parf'].values#mas
        cats = []
        for k in range(len(out['cats'])):  # out['cats'] = ["GDR2","GDR3"]     # used Gaia catalogs
            m = out['gost']['BJD'].values<out['{}_baseline'.format(out['cats'][k])]
            m &= out['{}_valid_gost'.format(out['cats'][k])]
            solution_vector = out['Gaia_solution_vector'][k]
            yy = np.array(gabs[m]).T
            
            # 5 parameters fitting
            theta = solution_vector@yy
            ast = theta.flatten()
            res = out['astro_gost'].iloc[k]-ast  # DR2, DR3 catalogs - fitted ast
            #print(out['astro_gost'].iloc[k])
            # type(res)-> <class 'pandas.core.series.Series'>
            cats.append(np.array(res))
        tmp['cats'] = np.array(cats)

    return tmp

###calculate the astrometric difference between two epochs
def AstroDiff(obs1,obs2):
###obs1, obs2: ra[deg], dec[deg], parallax [mas], pmra [mas/yr], pmdec [mas/yr], rv [km/s]
    astro_name = ['ra','dec','parallax','pmra','pmdec','radial_velocity']
    o1 = obs1[astro_name].values
    o2 = obs2[astro_name].values
    dobs = o2-o1
    dobs[:2] = dobs[:2]*3.6e6#deg to mas
    dobs[0] = dobs[0]*cos(np.mean([o1[1],o2[1]])*pi/180)

    return pd.Series(dobs,index=astro_name)   # <class 'pandas.core.series.Series'>

def dnorm(x, mean, sd, log=False):
    mu, sigma = mean, sd
    if log:
        return np.log(1./(np.sqrt(2*np.pi)*sigma))-(x-mu)**2/(2*sigma**2)
    else:
        return 1./(np.sqrt(2*np.pi)*sigma)*np.exp(-((x-mu)**2)/(2*sigma**2))

def check_bound(pars, ins, Np):
    names = pars.columns.to_list()
    #Np = np.sum(['omega' in i for i in names])
    cols = ['K','e','Omega','Inc','Mo']#'omega',
    low_bound = [1e-5,1e-5, 1e-5, 1e-5, 1e-5]
    hig_bound = [np.inf,1, 2*pi, pi,   2*pi]
    for i in range(Np):
        for j in range(len(cols)):
            key = cols[j] + '%d'%(i+1)
            if key in names:
                if (pars[key].values[0] >= low_bound[j]) and (pars[key].values[0] <= hig_bound[j]):
                    pass
                else:
                    #print('lalalal',key,pars[key].values[0])
                    return False
    for i in ins:
        key = 'J_' + i
        if pars[key].values[0]<0:
            return False
    
    if 'J_hip2' in names:
        if pars['J_hip2'].values[0]<0:
            return False
        
    if 'Mstar' in names:
        if pars['Mstar'].values[0]<0:
            return False
    if 'logJ_gaia' in names:
        if (pars['logJ_gaia'].values[0]<-12) or (pars['logJ_gaia'].values[0]>12):
            return False

    if 'J_gaia' in names:
        if (pars['J_gaia'].values[0]<0) or (pars['J_gaia'].values[0]>10):
            return False    
    
    return True
        
def logpost(par, RVonly=False, marginalize=False, out=None):
    # MCMC par: logP, k, esino, ecoso, Mo, ... , here I add Pd, e, omega to match that of Feng 
    pars = pd.DataFrame(par.reshape(1, len(par)),columns=new_index)
    Np = out['nplanet']#np.sum(['logP' in i for i in new_index])
    for i in range(Np):
        
        if (pars['logP%d'%(i+1)].values[0]<-10) or (pars['logP%d'%(i+1)].values[0]>15):
            return -np.inf
        
        if 'logK%d'%(i+1) in pars.keys():
            if (pars['logK%d'%(i+1)].values[0]<-10) or (pars['logK%d'%(i+1)].values[0]>15):
                return -np.inf
            pars['K%d'%(i+1)] = np.exp(pars['logK%d'%(i+1)].values[0])
            
        pars['Pd%d'%(i+1)] = np.exp(pars['logP%d'%(i+1)].values[0])
        esino, ecoso = pars['esino%d'%(i+1)].values[0], pars['ecoso%d'%(i+1)].values[0]
        pars['e%d'%(i+1)] = esino**2 + ecoso**2
        ome = np.arctan(esino/ecoso)
        #if ome < 0: ome = 2*np.pi+ome
        pars['omega%d'%(i+1)] = ome

    if not check_bound(pars, out['rv_ins'], Np):
        #print('check bound...')
        return -np.inf
    # Mstar prior (Guassian default)
    logprior = 0
    if 'eMstar' in out.keys() and np.isfinite(out['eMstar']) and 'Mstar' in pars.keys():
        x, mu, sigma = pars['Mstar'].values[0], out['Mstar'], out['eMstar']
        logprior += -0.5 * ((x - mu) / sigma)**2 - 0.5*np.log((sigma**2)*2.*np.pi)

    if 'J_gaia' in pars.keys():
        x, mu, sigma = pars['J_gaia'].values[0], 1, 0.1
        logprior += -0.5 * ((x - mu) / sigma)**2 - 0.5*np.log((sigma**2)*2.*np.pi)

    # if 'J_UVES' in pars.keys():  # new add for HD209100 
    #     x, mu, sigma = pars['J_UVES'].values[0], 2, 0.5
    #     logprior += -0.5 * ((x - mu) / sigma)**2 - 0.5*np.log((sigma**2)*2.*np.pi)

    ll = loglikelihood(pars, RVonly=RVonly, marginalize=marginalize, verbose=False, out=out)
    #print(ll)
    #sys.exit()
    #print('lalal:',pars['dra'].values)

    return logprior+ll
    
# likelihood, not include RV & relative astrometry terms (direct imaging)
def loglikelihood(pars,RVonly=False,marginalize=False,prediction=False,indrel=None,verbose=False,out=None):
        
    res_all = dict()
    names = pars.columns.to_list()
    Np = out['nplanet']#np.sum(['omega' in i for i in names])
    logLike = 0
    
    ####### RV
    jds, rvs, ervs, ins = out['rv_jds'], out['rv_data'], out['rv_err'], out['rv_ins']
    # ins -> instruments
    
    for n in range(len(ins)):
        jd, rv, erv = jds[n], rvs[n], ervs[n]
        model_rv = np.zeros_like(jd, dtype=float)
        for ip in range(Np):
            per = pars['Pd%d'%(ip+1)].values[0]
            e = pars['e%d'%(ip+1)].values[0]
            w = pars['omega%d'%(ip+1)].values[0]
            M0 = pars['Mo%d'%(ip+1)].values[0]
            k = pars['K%d'%(ip+1)].values[0]
            tp = out['tmin']-(M0%(2*pi))*per/(2*pi)
            model_rv += rv_calc(jd, [per, tp, e, w, k])
        jit = pars['J_'+ins[n]].values[0]
        if marginalize:
            ivar = 1./(erv**2+jit**2)  # adopt from orvara, marginalize rv offsets
            dRv = rv - model_rv        
            A = np.sum(ivar)
            B = np.sum(2*dRv*ivar)
            C = np.sum(dRv**2*ivar)    # gamma = -B/2/A
            chi2 = -B**2/4/A + C + np.log(A) - np.sum(np.log(ivar))
            ll = -0.5*chi2
        else:
            gamma = pars['b_'+ins[n]].values[0]
            residuals = rv - gamma - model_rv
            sigz = erv**2 + jit**2
            chi2 = np.sum(residuals**2/sigz + np.log(2*np.pi*sigz))
            #ll2 = np.sum(dnorm(residuals, mean=0, sd=np.sqrt(sigz), log=True)) # note: ll2=ll
            ll = -0.5*chi2
            
        logLike =  logLike +ll
    if verbose:
        print('\nll for RV:', logLike, '\n')
   
   
    if not np.isfinite(logLike):
        return -np.inf
    if RVonly:
        return logLike

    epoch = barycenter = None
    if('relAst' in out.keys() or len(out['data_epoch'])>0 or len(out['gost'])>0):
        astro = astrometry_kepler(pars_kep=pars,out=out)
        barycenter = astro['barycenter']
        epoch = astro['epoch']    # astro['epoch']['hip2'] reflex motion
    
    # direct imaging
    if 'relAst' in out.keys():
        ll = 0
        rel = astro['relAst']
        for i in range(len(rel['res_sep'])):
            x = np.array([rel['res_sep'][i], rel['res_PA'][i]])
            mean = np.repeat(0, len(x))
            cov = rel['cov'][i]
            ll += multivariate_normal.logpdf(x, mean=mean, cov=cov)
        logLike = logLike +ll
        if verbose:
            print('ll for direct image:', ll, '\n')
    
    # hip2 fit
    if len(out['data_epoch'])>0:
        ###reflex motion induced position change
        i  = out['ins_epoch']    # 'hip2'

        dpmdec = dpmra = dplx = 0
        n1 = 'J_{}'.format(i)
        n2 = 'logJ_{}'.format(i)
        s = 0
        if(n1 in names):
            s = pars[n1].values[0]
        if(n2 in names):
            s = np.exp(pars[n2].values[0])
        
        data_epoch = out['data_epoch'] # default to use 'hip2' IAD data
        ##contribution of reflex motion to target astrometry
        dra = epoch[i]['dra'].values
        ddec = epoch[i]['ddec'].values
        ###since plx, pmra and pmdec are parameters *fixed* at the reference epoch, we should not consider the "time-varying" contribution from reflex motion
        ll = 0
        if('hip' in i):
            dastro = AstroDiff(out['astrometry'].iloc[out['ihip']], barycenter.iloc[out['ihip']])

            ##contribution of barycenter-offset to target astrometry
            dra = dra+dastro['ra']
            ddec = ddec+dastro['dec']
            dplx = dplx+dastro['parallax']
            dpmra = dpmra+dastro['pmra']
            dpmdec = dpmdec+dastro['pmdec']
            
            if(i=='hip2'):
                dabs_new = data_epoch['CPSI'].values*(dra+dpmra*data_epoch['EPOCH'].values)+data_epoch['SPSI'].values*(ddec+dpmdec*data_epoch['EPOCH'].values)+data_epoch['PARF'].values*dplx
                res = data_epoch['RES'].values-dabs_new
                ll = ll+np.sum(dnorm(res, mean=0, sd=np.sqrt(data_epoch['SRES'].values**2+s**2), log=True))

                # sigz = data_epoch['SRES'].values**2+s**2
                # chi2 = np.sum(res**2/sigz + np.log(sigz*2*pi))
                # ll += -0.5*chi2
            ####https://aas.aanda.org/articles/aas/pdf/1998/10/ds1401.pdf
            ####https://aas.aanda.org/articles/aas/pdf/2000/13/ds1810.pdf
            ####https://www.cosmos.esa.int/documents/532822/552851/vol3_all.pdf/dca04df4-dc6f-4755-95f2-b1217e539926
        if(prediction):
            res_all['epoch_'+i] = np.array(res)
        
        logLike = logLike+ll
        if(verbose):
            print('ll for',i,'epoch astrometry=',ll,'\n')
            
    if not np.isfinite(logLike):
        return -np.inf

    ####Gaia GOST fit
    if len(out['gost'])>0:
        ll_gost = 0
        nast = len(out['astro_index'])    # out['astro_index']=np.array([1, 2])
        for k in range(nast):
            j = out['astro_index'][k]
            s = 1  # xiao 0623, s = 0
            if 'logJ_gaia' in names:
                s = np.exp(pars['logJ_gaia'].values[0])  # #inflated error
            if 'J_gaia' in names:
                s = pars['J_gaia'].values[0]
            x = np.array(astro['cats'][k])
            mean = np.repeat(0., 5)
            cov = out['cov_astro'][j]*(s**2)  # xiao 0623  *(1+s)
            
            #cov = cov.reshape(5,5)
            #print(cov)
            ll = multivariate_normal.logpdf(x, mean=mean, cov=cov)
            #ll = np.log(pdf_val)
            #print(x,cov,ll,s)                     # test logJ_gaia
            ll_gost = ll_gost+ll
            if(verbose):
                print('ll for gost:',out['cats'][k],ll,'\n')
        logLike = logLike+ll_gost
    
    if not np.isfinite(logLike):
        return -np.inf    

    if(not prediction):
        return(logLike)              # return(list(ll=logLike,llastro=ll.astro))
    else:
        return {'loglike':logLike,'res':res_all}

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
    if pos1=='upper':
        yc = ymax-0.08*dy
    elif pos1=='lower':
        yc = ymin+0.08*dy
    else:
        print('error location!')
        sys.exit()
        
    if pos2=='left':
        xc = xmin-0.08*dx
    elif pos2=='right':
        xc = xmax+0.08*dx
    else:
        print('error location!')
        sys.exit()        
    a = 0.04*dx
    b = a
    theta = np.arange(0,2*pi,0.01)
    xs = xc+a*cos(theta)
    ys = yc+b*sin(theta)
    ax.plot(xs,ys,'grey',alpha=0.6,lw=1)
    w, h = a, b
    if inc_deg<90:
        w, h = -w, -h
    ax.arrow(xc,yc+b,-w,0,head_width=0.3*a, head_length=0.5*b, color='grey',alpha=0.5,lw=1)
    ax.arrow(xc,yc-b,w,0,head_width=0.3*a, head_length=0.5*b, color='grey',alpha=0.5,lw=1)
    ax.arrow(xc+a,yc,0,h,head_width=0.3*a, head_length=0.5*b, color='grey',alpha=0.5,lw=1)
    ax.arrow(xc-a,yc,0,-h,head_width=0.3*a, head_length=0.5*b, color='grey',alpha=0.5,lw=1)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

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

def predict_positon(MCMCtab, pre_epoch='2024-01-01T0:0:0', iplanet=1, tmin=None, par_opt=None, plx=None):
    i = iplanet
    jd_ref = 0#Time(pre_epoch, format='isot', scale='utc').jd
    mstars = MCMCtab['Mstar'].values
    mps = MCMCtab['Mc%d'%i].values/1047.0
    M0s = MCMCtab['Mo%d'%i].values
    ps = MCMCtab['Pd%d'%i].values
    MA = (M0s+2*pi*((jd_ref-tmin)%ps)/ps)%(2*pi)
    es = MCMCtab['e%d'%i].values
    EA = kep_mt2(MA,es)
    incs = MCMCtab['Inc%d'%i].values
    omegas = MCMCtab['omega%d'%i].values
    Omegas = MCMCtab['Omega%d'%i].values
    Ks = MCMCtab['K%d'%i].values
    plx = plx-par_opt['dplx'].values
    astro = calc_astro(K=Ks,P=ps,e=es,inc=incs,omega=omegas,Omega=Omegas,E=EA,plx=plx,Mstar=mstars,pars=par_opt,eta=0)
    xis = mps/(mstars+mps)
    dra_mc = -astro[0]/xis
    ddec_mc = -astro[1]/xis
    pa_mc = np.arctan2(dra_mc,ddec_mc)*180/pi#deg
    sep_mc = np.sqrt(dra_mc**2+ddec_mc**2)*1e-3#as
    print('sep=',np.mean(sep_mc),'+-',np.std(sep_mc),'arcsec;pa=',np.mean(pa_mc),'+-',np.std(pa_mc),'deg\n')
    return dra_mc, ddec_mc, pa_mc, sep_mc


def nodes_periastron(par_opt, iplanet=1, plx=None, types='star'):
    i = iplanet
    es = par_opt['e%d'%i].values[0]
    Ks = par_opt['K%d'%i].values[0]
    ps = par_opt['Pd%d'%i].values[0]
    incs = par_opt['Inc%d'%i].values[0]
    omegas = par_opt['omega%d'%i].values[0]
    Omegas = par_opt['Omega%d'%i].values[0]
    eccterm = np.sqrt((1 - es)/(1 + es))
    EA = [-1e-3, 0, 2*np.arctan(eccterm*np.tan((np.pi - omegas)/2.)),
          2*np.arctan(eccterm*np.tan(-omegas/2.))]
    EA = np.array(EA)
    plx = plx-par_opt['dplx'].values[0]
    mstars = par_opt['Mstar'].values[0]
    astro = calc_astro(K=Ks,P=ps,e=es,inc=incs,omega=omegas,Omega=Omegas,E=EA,plx=plx,Mstar=mstars,pars=par_opt,eta=0)
    xis = 1
    if types=='planet':
        mps = par_opt['Mc%d'%i].values[0]/1047.0
        xis = -mps/(mstars+mps)
    return astro[0]/xis, astro[1]/xis

def set_init_params(nwalkers, nsteps, ndim, ntemps, MC_index, nplanet, startfn=None):
    use_startfn = False
    if startfn is not None:  # use start file as init params
        if os.path.exists(startfn):
            print('\nload start file:',startfn)
            use_startfn = True
            init, pname = [], []
            f = open(startfn)
            for line in f:
                if line[0]=='#':continue
                lst = line.strip().split()
                pname.append(lst[0])
                init.append(float(lst[1]))
            f.close()
            
            if len(init) != len(MC_index):
                print('Warn: start file can not match MCMC index! (length)')
                use_startfn = False
            for i in range(len(MC_index)):  # check par names
                if MC_index[i] != pname[i]:
                    print('Warn: start file can not match MCMC index! (name)')
                    use_startfn = False
                    break
        else:
            print('\nstar file not found:',startfn)
            
    if use_startfn:
        print('\nInitial ll:',logpost(np.asarray(init),RVonly=RVonly,marginalize=marginalize,out=out)) 
    

    if not use_startfn:  # default values
        init = np.ones(ndim)
        init = pd.DataFrame(init.reshape(1, len(init)),columns=MC_index)
        p0 = {'logP':9.1, 'logK':5, 'K':47, 'esino':-0.2, 'ecoso':0.1, 'dra':0.1, 'ddec':0.1, 
              'dplx':0.1, 'dpmra':0.1, 'dpmdec':0.1,
              'logJ_gaia':5, 'J_gaia':0.1}
        for key in p0.keys():
            if key in init.keys():
                init[key] *=p0[key]            
            for i in range(nplanet):
                if key+str(i+1) in init.keys():
                    init[key+str(i+1)] *=(p0[key]*(i+1))
        init = init.iloc[0].values

        # customer can define any initial pars, but their order should match MC_index
        #p1 = ['logP1', 'K1', 'esino1', 'ecoso1', 'Mo1', 'Omega1', 'Inc1',]
        #p2 = ['logP2', 'K2', 'esino2', 'ecoso2', 'Mo2', 'Omega2', 'Inc2',]
        pall = []
        if len(pall)>0:
            for i in range(len(pall)):
                init[i] = pall[i]
    
    if ntemps != 0:
        par0 = np.ones((ntemps, nwalkers, ndim))
    else:
        par0 = np.ones((nwalkers, ndim))

    par0 *= np.asarray(init)
    np.random.seed(2)
    scatter = 0.05*np.random.randn(np.prod(par0.shape)).reshape(par0.shape)
    par0 += scatter

    all_jit = ['J_'+i for i in out['rv_ins']] + ['J_hip2']
    
    # check the bound of initial parameters since some of them might be unreasonable 
    for j, key in enumerate(MC_index):
        if key in (['Mstar'] + all_jit):
            par0[..., j][par0[..., j] < 1e-5] = 1e-5  # low bound
            par0[..., j][par0[..., j] > 1e6] = 1e6    # high bound
        if key in (['J_gaia']):
            par0[..., j][par0[..., j] < 1e-5] = 1e-5  # low bound
            par0[..., j][par0[..., j] > 10] = 10    # high bound
        if re.sub(r'\d+$', '', key) in ['Inc']:
            par0[..., j][par0[..., j] < 1e-5] = 1e-5
            par0[..., j][par0[..., j] > pi] = pi
        if re.sub(r'\d+$', '', key) in ['K']:
            par0[..., j][par0[..., j] < 1e-5] = 1e-5
        if re.sub(r'\d+$', '', key) in ['logP','logK']:
            par0[..., j][par0[..., j] < -1] = -1  # low bound
            par0[..., j][par0[..., j] > 15] = 15    # high bound  
        if re.sub(r'\d+$', '', key) in ['logJ_gaia']:
            par0[..., j][par0[..., j] < -12] = -12  # low bound
            par0[..., j][par0[..., j] > 12] = 12    # high bound
        if re.sub(r'\d+$', '', key) in ['Mo','Omega']:
            par0[..., j][par0[..., j] < 1e-5] = 1e-5    # low bound
            par0[..., j][par0[..., j] > 2*pi] = 2*pi    # high bound    
 
    for i in range(nplanet):  # adopt from orvara
        if ('esino%d'%(i+1) in MC_index) and ('ecoso%d'%(i+1) in MC_index):
            js = [j for j, key in enumerate(MC_index) if key=='esino%d'%(i+1)][0]
            jc = [j for j, key in enumerate(MC_index) if key=='ecoso%d'%(i+1)][0]
            ecc = par0[..., js]**2 + par0[..., jc]**2      # from orvara, keep e within [0,1]
            fac = np.ones(ecc.shape)
            fac[ecc > 0.99] = np.sqrt(0.99)/np.sqrt(ecc[ecc > 0.99])
            par0[..., js] *= fac
            par0[..., jc] *= fac

    return par0


############################## load RV data ############################
target = 'HD221420'#'HD30219'##'HIP97657'#'WASP-107'#'HD259440'#'WD1202-232'#'HD29021'##'HD222237'#'HD68475'
RVonly = False
nplanet = 1
print('*'*15,target,'*'*15,'\n')

prefix = '.'
rvdir = prefix +'{}/'.format(target) #+ '{}bin_old/'.format(target)
#'/home/xiaogy/exoplanet/Test_agatha/data/combined/HD222237_test/*'
RVfiles = glob.glob(rvdir+'*')
jds, rvs, ervs, offsets, jitters, ins = [], [], [], [], [], []
RVfiles = sorted(RVfiles)
print('load RV file from:',rvdir)
for fn in RVfiles:
    if fn.endswith((".vels")):
        tel = fn.replace('.vels','').split('_')[-1]
        print('\nInstrument:',tel)
    elif fn.endswith((".dat")) and 'photo' not in fn:
        tel = fn.replace('.dat','').split('_')[-1]
        print('\nInstrument:',tel)
    else:
        continue
    ins.append(tel)
    tab = pd.read_table(fn, sep='\s+', header=0, encoding='utf-8')
    if 'MJD' in tab.columns:
        jds.append(tab['MJD'].values+2400000)
    elif (tab['BJD'].values[0]<10000) and (target == 'HD259440'):
        jds.append(tab['BJD'].values+2450000)
    else:
        jds.append(tab['BJD'].values)
    try:
        tmprv = tab['RV'].values
    except KeyError:
        tmprv = tab['RV_mlc'].values
    if target in ['HD221420'] and 'HARPS' in tel:
        tmprv *= 1000        #Km/s -> m/s
    rvs.append(tmprv-np.mean(tmprv))
    try:
        tmperv = tab['eRV'].values
    except KeyError:
        tmperv = tab['e_RV_mlc'].values
    ervs.append(tmperv)
    print(fn, len(tab))

if len(jds) == 0:
    tmin = 2457206
    print('RV file not found! Use default tmin:',tmin)
else:
    for i in range(len(jds)):
        inds = np.argsort(jds[i])
        jds[i], rvs[i], ervs[i] = jds[i][inds], rvs[i][inds], ervs[i][inds]
    tmin, tmax = min_max_jd(jds)

out = dict()
out['rv_jds'], out['rv_data'], out['rv_err'], out['rv_ins']= jds, rvs, ervs, ins
out['tmin'] = tmin
out['nplanet'] = nplanet
############################## load imaging data ##############################
relAstfn = prefix + '{}/{}_Image.rel1'.format(target, target)  
# default 
# Date (UTC or BJD); Sep("); Err_sep; PA(deg); Err_PA; Corr_Sep_PA; PlanetID (1-th,2-th ...) 
if os.path.exists(relAstfn) and True:
    out['relAst'] = dict()
    out['relAst_type'] = 'Sep_PA'   # 'Dra_Ddec', units:";  'Sep_PA', units:" and deg
    rel_data = np.loadtxt(relAstfn)
    try:
        tmp = rel_data[:,0]
    except IndexError:
        rel_data = rel_data.reshape(1,len(rel_data))
    rel_epoch = rel_data[:,0]
    if np.median(rel_epoch) < 3000: # if utc -> BJD
        rel_epoch = (rel_epoch - 2000)*365.25 + 2451544.5
    out['relAst']['rel_JD'] = rel_epoch
    out['relAst']['rel_sep'] = rel_data[:,1]
    out['relAst']['rel_sep_err'] = rel_data[:,2]
    out['relAst']['rel_PA'] = rel_data[:,3]
    out['relAst']['rel_PA_err'] = rel_data[:,4]
    out['relAst']['rel_corr'] = rel_data[:,5]
    out['relAst']['rel_iplanet'] = np.int32(rel_data[:,6])
    print('\nload imaging file:',relAstfn, len(rel_epoch))

############################## load astrometry data ######################
gaia_dr3_refep = 2457388.5
dr3_base_line = 2457902
gaia_dr2_refep = 2457206
dr2_base_line = 2457532

out['GDR2_baseline'] = dr2_base_line
out['GDR3_baseline'] = dr3_base_line
out['gaia_dr2_refep'] = gaia_dr2_refep
out['gaia_dr3_refep'] = gaia_dr3_refep

############## catalog astrometry
fn = prefix + '{}/{}_hipgaia.hg123'.format(target, target)
#'/home/xiaogy/exoplanet/Test_agatha/data/combined/HD222237_test/HD222237_hipgaia.hg123'
if os.path.exists(fn):
    cata_astrometry = pd.read_table(fn, sep='\s+', header=0, encoding='utf-8')
    print('\nload catalog file:',fn, len(cata_astrometry))
    if len(cata_astrometry)<3:
        print('Warning: maybe lack of Hip or Gaia data!')
    if len(cata_astrometry)>3:
        cata_astrometry = cata_astrometry.drop(1)   # delete DR1 catalog data
        cata_astrometry = cata_astrometry.reset_index(drop=True)
        print('\ndelete DR1 catalog astrometry!')
    out['astrometry'] = cata_astrometry
    df = cata_astrometry[['ra','dec','parallax','pmra','pmdec']][-2:]
    df['ra'] = (df['ra']-df['ra'].iloc[-1])*np.cos(df['dec'].iloc[-2]*np.pi/180)*3.6e6
    df['dec'] = (df['dec']-df['dec'].iloc[-1])*3.6e6
    
    df.rename(columns={"ra": "dra", 'dec':'ddec'}, inplace=True)
    out['astro_gost'] = df

    tt = cata_astrometry[['ra','dec','parallax','pmra','pmdec']].iloc[2]
    #print(out['astrometry'].iloc[[0,1]]['ra'])
    cov_astro = construct_cov(cata_astrometry)
    out['cov_astro'] = cov_astro
    out['ihip'] = 0
    out['iref'] = 2  # 参考epoch所在索引Gaia catalogs，DR3->2第3行数据   hip-0, dr2-1, dr3-2
    out['cats'] = ["GDR3"]          # used Gaia catalogs, "GDR2","GDR3"
    out['astro_index'] = np.array([2])  # "GDR2","GDR3" index, 1, 2

    if 'GDR2' not in out['cats']:out['astro_gost'] = out['astro_gost'].drop(1)
    print('\nUsing Gaia:',out['cats'])
    print('Ref epoch (DR3):',out['astrometry'].iloc[out['iref'],0])
    # print(out['astrometry'])
else:
    print('\nCatalog astrometry not found!')
    out['astrometry'] = []


### check whether PM-induced RV trend is subtracted; perspective acceleration
for iins in ['VLC','LC']:
    if (iins in ins) and (len(out['astrometry'])>0) and False: # if False, do not correct PA effect
        ind = [j for j in range(len(ins)) if ins[j]==iins][0]
        t3 = jds[ind]-out['astrometry'].iloc[out['iref'],0]
        obs0 = out['astrometry'][['ra','dec','parallax','pmra','pmdec','radial_velocity']].iloc[out['iref']]
        tmp = obs_lin_prop(obs0,t3)
        rv_pm = (tmp['radial_velocity'].values)*1e3#m/s
        rv_pm -= rv_pm[0]
        rvs[ind] -= rv_pm
        print('\nCorrect perspective acceleration:',iins)
        #print(rv_pm)

########### Hip2 abs data
hpfn = prefix + '{}/{}_hip2.abs'.format(target, target)
#'/home/xiaogy/exoplanet/Test_agatha/data/combined/HD222237_test/HD222237_hip2.abs'
if os.path.exists(hpfn) and True:        # if false, then not use hip2
    print('\nload hip2 abs file:',hpfn)
    hip2 = pd.read_table(hpfn, sep='\s+', header=0, encoding='utf-8')
    out['data_epoch'] = hip2
    out['ins_epoch'] = 'hip2'
else:
    print('\nhip2 abs file not found!')
    out['data_epoch'] = []

########### Gaia Gost data
gostfn = prefix + '{}/{}_gost.csv'.format(target, target)
#'/home/xiaogy/exoplanet/Test_agatha/data/combined/HD222237_test/HD222237_gost.csv'
if os.path.exists(gostfn):
    print('\nload Gost file:', gostfn)
    tb = pd.read_csv(gostfn,comment='#')
    goname = ['BJD', 'psi', 'parf', 'parx']
    colname = ['ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]',
               'scanAngle[rad]','parallaxFactorAlongScan','parallaxFactorAcrossScan']
    gost = []
    for key in colname:
        gost.append(tb[key].values)
    gost = np.array(gost)
    m = gost[0,:] < dr3_base_line
    gost = pd.DataFrame(gost[:,:m.sum()].T, columns=goname)
    out['gost'] = gost
    
    # check dead time of DR2 and DR3
    epochs = out['gost']['BJD']
    valid2 = np.ones(len(epochs), dtype=bool)
    valid3 = np.ones(len(epochs), dtype=bool)
    if False:
        DR2_dead_time = prefix + 'astrometric_gaps_gaiadr2_08252020.csv'
        tb2 = pd.read_csv(DR2_dead_time,comment='#')
        tb2['start'] = 2457023.75 + (tb2['start'] - 1717.6256)/(1461)*365.25
        tb2['end']   = 2457023.75 + (tb2['end'] - 1717.6256)/(1461)*365.25
        for dead in tb2.values:
            valid2[np.logical_and(epochs >= dead[0], epochs <= dead[1])] = 0
        print('\nDead time for DR2:',np.sum(~valid2),'/',np.sum(out['gost']['BJD']<dr2_base_line))
        DR3_dead_time = prefix + 'astrometric_gaps_gaiaedr3_12232020.csv'
        tb3 = pd.read_csv(DR3_dead_time,comment='#')
        tb3['start'] = 2457023.75 + (tb3['start'] - 1717.6256)/(1461)*365.25
        tb3['end']   = 2457023.75 + (tb3['end'] - 1717.6256)/(1461)*365.25
        for dead in tb3.values:
            valid3[np.logical_and(epochs >= dead[0], epochs <= dead[1])] = 0
        print('\nDead time for DR3:',np.sum(~valid3),'/',np.sum(out['gost']['BJD']<dr3_base_line))
    out['GDR2_valid_gost'] = valid2
    out['GDR3_valid_gost'] = valid3 
    
    # coefficients of 5-p model of Gaia astrometry
    out['a1'],out['a2'],out['a3'],out['a4'],out['a5'] = [],[],[],[],[]
    if 'GDR2' in out['cats']:
        m = (out['gost']['BJD']<dr2_base_line) & out['GDR2_valid_gost']
        out['a1'].append(np.sin(out['gost']['psi'][m].values))
        out['a2'].append(np.cos(out['gost']['psi'][m].values))
        out['a3'].append(out['gost']['parf'][m].values)
        out['a4'].append(((out['gost']['BJD']-gaia_dr2_refep)/365.25*np.sin(out['gost']['psi']))[m].values)
        out['a5'].append(((out['gost']['BJD']-gaia_dr2_refep)/365.25*np.cos(out['gost']['psi']))[m].values)
    if 'GDR3' in out['cats']:
        m = (out['gost']['BJD']<dr3_base_line) & out['GDR3_valid_gost']
        out['a1'].append(np.sin(out['gost']['psi'][m].values))
        out['a2'].append(np.cos(out['gost']['psi'][m].values))
        out['a3'].append(out['gost']['parf'][m].values)
        out['a4'].append(((out['gost']['BJD']-gaia_dr3_refep)/365.25*np.sin(out['gost']['psi']))[m].values)
        out['a5'].append(((out['gost']['BJD']-gaia_dr3_refep)/365.25*np.cos(out['gost']['psi']))[m].values)
    
    out['Gaia_solution_vector'] = []
    for k in range(len(out['a1'])):
        df = {'a1':out['a1'][k],'a2':out['a2'][k],'a3':out['a3'][k],'a4':out['a4'][k],'a5':out['a5'][k]}
        data = pd.DataFrame(df)
        XX_dr = np.array([data['a1'].values, data['a2'].values, data['a3'].values, data['a4'].values,data['a5'].values]).T
        solution_vector = np.linalg.inv(XX_dr.T@XX_dr).astype(float)@XX_dr.T
        out['Gaia_solution_vector'].append(solution_vector)
else:
    print('\nGost file not found!')
    out['gost'] = []


out['Mstar'] = out['astrometry'].iloc[-1]['mass']
mlower, mupper = out['astrometry'].iloc[-1]['mass.lower'], out['astrometry'].iloc[-1]['mass.upper']
if mupper < out['Mstar']:
    out['eMstar'] = (mlower + mupper)/2
else:
    out['eMstar'] = (abs(mlower-out['Mstar']) + abs(mupper-out['Mstar']))/2
#if out['eMstar'] = np.inf, then the prior of stellar mass will be uniform. Stellar mass will be treated as free parameter. relative astrometry (direct image)
out['plx'] = out['astrometry'].iloc[-1]['parallax']  #87.3724 HD222237 Dr3
print('\nstellar mass from catalog:',out['Mstar'], out['eMstar'], 'plx:',out['plx'],'\n')

############################## Feng MCMC index ######################
# Index(['Mc1', 'Tp1', 'Mstar', 'Pd1', 'K1', 'e1', 'omega1', 'Mo1', 'Inc1',
#        'Omega1', 'b_AAT', 'J_AAT', 'b_HARPSpost', 'J_HARPSpost',
#        'c_HARPSpost1', 'c_HARPSpost2', 'c_HARPSpost3', 'c_HARPSpost4',
#        'c_HARPSpost5', 'b_HARPSpre', 'J_HARPSpre', 'w1_HARPSpre',
#        'beta_HARPSpre', 'b_PFS', 'J_PFS', 'w1_PFS', 'beta_PFS', 'logJ_hip2',
#        'logJ_gaia', 'dra', 'ddec', 'dplx', 'dpmra', 'dpmdec', 'logpost',
#        'loglike'],
#       dtype='object')
if len(out['data_epoch'])==0:  # if no hip2 abs data
    MCMC_pars_base = ['logP','logK','esino','ecoso','Mo','Omega','Inc',
                      'dra', 'ddec', 'dplx', 'dpmra', 'dpmdec','Mstar']  # default MCMC parameters'J_gaia',
else:
    MCMC_pars_base = ['logP','K','esino','ecoso','Mo','Omega','Inc','J_hip2','J_gaia',
                      'dra', 'ddec', 'dplx', 'dpmra', 'dpmdec','Mstar']  # default MCMC parameters'J_gaia',
    
# note: the meaning of J_hip2 and J_gaia is different, J_gaia->s, see model
# derived par: Mc, Tp
new_index = []
for i in range(nplanet):
    if RVonly:
        for j in MCMC_pars_base[:5]:   # 5*nplanet parameters
            new_index += [j+'%d'%(i+1)]
    else:
        for j in MCMC_pars_base[:7]:
            new_index += [j+'%d'%(i+1)]

marginalize = False
for i in ins:
    if marginalize:
        new_index += ['J_'+i] # marginalize rv offset, 'b_'+i, 
    else:
        new_index += ['b_'+i,'J_'+i]
if not RVonly:
    new_index += MCMC_pars_base[7:]   # 7*nplanet parameters
ndim = len(new_index)
### plot RVs
# plt.figure()
# for n in range(len(jds)):
#     plt.plot(jds[n], rvs[n], 'o', label=ins[n])
# plt.legend()
# sys.exit()
############################## PTmcmc ######################################
if True:
    def return_one(theta):
        return 1.
    # The number of walkers must be greater than ``2*dimension``
    ntemps = 10
    nwalkers = max(50, 2*(ndim+1))
    nsteps = 40000
    ndim = ndim
    thin = 100     # final saved steps -> nsteps/thin
    nthreads = 1
    buin = min(200, int(nsteps/thin/2))    # should < nsteps/thin
    print('MCMC pars:',new_index, 'ndim:',ndim,'buin:',buin,'\n')
    
    par0 = set_init_params(nwalkers=nwalkers, nsteps=nsteps, ndim=ndim, ntemps=ntemps, 
                           MC_index=new_index, nplanet=nplanet, startfn='{}_pars.dat'.format(target))
    start_time = time.time()
    
    # check initial parameters
    for i in range(par0.shape[0]):
        for j in range(par0.shape[1]):
            ll = logpost(par0[i,j],RVonly=RVonly,marginalize=marginalize,out=out)
            if not np.isfinite(ll):
                print(i,j,ll)
                print(par0[i,j])
                sys.exit()
    #ll = logpost(par0[0,0],RVonly=RVonly,out=out) 
    print('Check time: %.5f second' % ((time.time() - start_time)))
    #sys.exit()
    sample0 = PTSampler(ntemps=ntemps, nwalkers=nwalkers, dim=ndim,
                    logl=logpost, loglargs=[RVonly, marginalize, out],
                    logp=return_one,
                    threads=nthreads)
    # sample0.run_mcmc(par0, nsteps, thin=thin)
    # print('Time: %.5f second' % ((time.time() - start_time)))
    # sys.exit()    
    
    # add a progress bar, adapt from orvara
    width = 30
    N = min(100, nsteps//thin)
    n_taken = 0
    sys.stdout.write("[{0}]  {1}%".format(' ' * width, 0))
    for ipct in range(N):
        dn = (((nsteps*(ipct + 1))//N - n_taken)//thin)*thin
        n_taken += dn
        if ipct == 0:
            sample0.run_mcmc(par0, dn, thin=thin)
        else:
            # Continue from last step
            sample0.run_mcmc(sample0.chain[..., -1, :], dn, thin=thin)
        n = int((width+1) * float(ipct + 1) / N)
        sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
        sys.stdout.write("%3d%%" % (int(100*(ipct + 1)/N)))
    sys.stdout.write("\n")
    
    print("Mean acceptance fraction (cold chain): {0:.6f}".format(np.mean(sample0.acceptance_fraction[0, :])))
    print('Total Time: %.0f mins' % ((time.time() - start_time)/60.)) 
    
    ############################# plot chain ####################################
    # plot chain and save max lnp params
    plot_file = '{}_pars.dat'.format(target)
    lnp = sample0.logprobability[0] 
    index = np.where(lnp==np.max(lnp))
    row, col = index[0][0],index[1][0]
    if True:           # Save maximum prob parameters
        f = open(plot_file, 'w')
        for i in range(ndim):
            val = sample0.chain[0, row, col,i]
            message = new_index[i] + ' ' + '{:.4f}'.format(val)
            print(message)
            f.write(message + os.linesep)
        f.close() 
        print('Save maximum prob parameters:',plot_file,np.max(lnp[:,buin:].flatten()))
    
    ################# plot chain to check convergence visually ##################
    for v in range(99):
        if not RVonly:
            savefn = target + '_RVast_corner_{:03d}.pdf'.format(v+1)
        else:
            savefn = target + '_RVonly_corner_{:03d}.pdf'.format(v+1)
        if not os.path.exists(savefn):
            break
        
    plt.figure()
    for i in range(nwalkers):
        plt.plot(range(lnp.shape[1]),lnp[i,:],lw=0.4,c='r')
    plt.savefig('{}_check_converge_{:03d}.png'.format(target,v+1))
    ############################### plot RV fitting ##############################
    if len(jds) != 0:
        plt.figure(figsize=(8,6),dpi=150)
        ax = plt.gca()
        best_par = sample0.chain[0, row, col,:]
        pars = pd.DataFrame(best_par.reshape(1, len(best_par)),columns=new_index)
        
        tsim = np.linspace(tmin, tmax, 1000)
        mol_rvs = np.zeros_like(tsim, dtype=float)
        for n in range(len(ins)):
            jd, rv, erv = jds[n], rvs[n], ervs[n]
            model_rv = np.zeros_like(jd, dtype=float)
            for i in range(out['nplanet']):
                per = np.exp(pars['logP%d'%(i+1)].values[0])
                esino, ecoso = pars['esino%d'%(i+1)].values[0], pars['ecoso%d'%(i+1)].values[0]
                e = esino**2 + ecoso**2
                w = np.arctan(esino/ecoso)
                M0 = pars['Mo%d'%(i+1)].values[0]
                if 'logK%d'%(i+1) in pars.keys():
                    k = np.exp(pars['logK%d'%(i+1)].values[0])
                else:
                    k = pars['K%d'%(i+1)].values[0]
                tp = out['tmin']-(M0%(2*pi))*per/(2*pi)
                model_rv += rv_calc(jd, [per, tp, e, w, k])
                if n==0:
                    mol_rvs += rv_calc(tsim, [per, tp, e, w, k])
            jit = pars['J_'+ins[n]].values[0]
            if marginalize:
                ivar = 1./(erv**2+jit**2)  # adopt from orvara, marginalize rv offsets
                dRv = rv - model_rv        
                A = np.sum(ivar)
                B = np.sum(2*dRv*ivar)   
                gamma = -B/2/A
            else:
                gamma = pars['b_'+ins[n]].values[0]
            ax.errorbar(jd-2450000, rv-gamma, yerr=np.sqrt(erv**2 + jit**2), 
                             fmt='o', ecolor='black',capsize=3, 
                             capthick=1, elinewidth=1.2,ms=10,mec='k', label=ins[n])
        ax.plot(tsim-2450000, mol_rvs, 'k-', rasterized=False, lw=2.5, zorder=99)
        
        ax.set_xlabel('JD-2450000', weight='bold',fontsize=18)
        ax.set_ylabel('RV [m/s]', weight='bold',fontsize=18)
        ax.xaxis.grid(False)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(direction='in', which='both', left=True, right=True, bottom=True, top=False,labelsize=18)
    
        #ax.grid(False)
        ax.legend(fontsize=14)
        plt.savefig('{}_RV_fitting_{:03d}.png'.format(target,v+1))
    
    #sys.exit()
    ################################# plot corner ################################ 
    flat_samples = []
    for i in range(ndim):
        flat_samples.append(sample0.chain[0,:,buin:,i].flatten())
    
    flat_samples = np.asarray(flat_samples).T
    
    fig = corner.corner(flat_samples, labels=new_index, quantiles=[0.16, 0.5, 0.84],
                        range=[0.999 for l in new_index], verbose=False, show_titles=True, 
                        title_kwargs={"fontsize": 12}, hist_kwargs={"lw":1.}, title_fmt='.2f',
                        label_kwargs={"fontsize":15}, xlabcord=(0.5,-0.45), ylabcord=(-0.45,0.5))

    print('\nSave corner plot to: ',savefn)
    fig.savefig(savefn)

    # save posterior
    df0 = pd.DataFrame(flat_samples, columns=new_index)

    for i in range(nplanet):
        df0['Pd%d'%(i+1)] = np.exp(df0['logP%d'%(i+1)].values)
        esino, ecoso = df0['esino%d'%(i+1)].values, df0['ecoso%d'%(i+1)].values
        df0['e%d'%(i+1)] = esino**2 + ecoso**2
        df0['omega%d'%(i+1)] = np.arctan(esino/ecoso)
        M0 = df0['Mo%d'%(i+1)].values
        P = df0['Pd%d'%(i+1)].values
        tp = tmin-(M0%(2*pi))*P/(2*pi)
        df0['Tp%d'%(i+1)] = tp 
        if not RVonly:
            if 'logK%d'%(i+1) in df0.keys():
                df0['K%d'%(i+1)] = np.exp(df0['logK%d'%(i+1)].values)
            K = df0['K%d'%(i+1)].values
            e = df0['e%d'%(i+1)].values
            Ms = df0['Mstar'].values
            Inc = df0['Inc%d'%(i+1)].values
            Mc = k2m(K,P,e,Ms,Inc=Inc)['ms']  # unit: msun
            df0['Mc%d'%(i+1)] = Mc
            print('MJ:',np.median(Mc)*1048, np.std(Mc)*1048,'e:',np.median(e),'Pd:',np.median(P))
    df0['logpost'] = lnp[:,buin:].flatten()
        
    if not RVonly:
        savefn = '{}_RVAst_posterior_{:03d}.txt'.format(target,v+1)   
    else:
        savefn = '{}_RVonly_posterior_{:03d}.txt'.format(target,v+1)
    df0.to_csv(savefn,sep=' ',mode='w',index=False)
    print('\nSave posterior to: ',savefn)
    
    sys.exit()


