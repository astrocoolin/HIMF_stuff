#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from scipy.optimize import curve_fit
from scipy import integrate
from Input_output import rothead

def func(x,a,b,c,d):
    #exponential fit for V0, rPE
    #return d*x**3. + a*x**2. + b*x + c
    #return d*x**3. + a*x**2. + b*x + c
    #return a*x**2. + b*x + c
    return a*np.exp(-x*b-c) +d*x

def func2(x,b,c):
    #Linear fit for A, Rt
    return b*x + c

def make_vrot(radi,Mag,hr,v):
    A=np.array([0.008,0.002,0.003,0.002,0.011,0.022,0.010,0.020,0.029,0.019])
    dA = np.array([0.003,0.001,0.001,0.001,0.001,0.002,0.003,0.005,0.005,0.015])

    V0 = np.array([275.,255.,225.,200.,170.,148.,141.,122.,103.,85.])
    dV0 = np.array([6.,2.,1.,1.,1.,2.,2.,2.,2.,5.])

    rPE = np.array([0.126,0.132,0.149,0.164,0.178,0.201,0.244,0.261,0.260,0.301])
    drPE = np.array([0.007,0.003,0.003,0.002,0.003,0.004,0.005,0.008,0.008,0.002])

    m=np.array([-23.76,-23.37,-22.98,-22.60,-22.19,-21.80,-21.41,-21.02,-20.48,-19.38])

    V0, foo = curve_fit(func, m, V0,sigma=dV0)
    rPE,foo = curve_fit(func2, m, rPE,sigma=drPE)
    A, foo  = curve_fit(func2, m, A,sigma=dA)

    #Mag = np.arange(-24,10.,0.1)

    vt=func(Mag,*V0)
    rt=hr*func2(Mag,*rPE)
    a=np.max([0.,func2(Mag,*A)])

    #plt.figure(1)
    #plt.plot(Mag,vt)
    #plt.scatter(m,V0)
    #plt.figure(2)
    #plt.plot(Mag,rt)
    #plt.scatter(m,hr*rPE)
    #plt.figure(3)
    #plt.scatter(m,A)
    #plt.plot(Mag,a)
    #plt.show()

    print('vt[km/s]:',round(vt,2))
    print('rt[arcsec]:',round(rt,2))
    print('a:',round(a,2))

    return vt*(1.-np.exp(-radi/rt))*(1.+a*radi/rt)

def sbr_calc(radi,RHI,x,dx,vt,Rs):
    sbr = np.zeros_like(radi)
    for j, r in enumerate(radi):
        sig1 = np.exp(-((r-0.4*RHI)/(np.sqrt(2)*(x+dx)*RHI))**2.)
        sig2 = (np.sqrt(vt/120.)-1.)*np.exp(-r/Rs)
        sbr[j] = sig1-sig2
        if sbr[j] < 0:
            sbr[j] = 0
    R_HI= np.argmin(abs(radi**2. - RHI**2.))
    sbr = sbr/sbr[R_HI]
    return sbr


def make_sbr(radi,Rs,DHI,vt,mass):
    RHI=DHI/2.
    x=0.36
    delta = np.arange(-0.045,-0.03,0.001)
    Mass_guess = np.zeros_like(delta)

    for i, dx in enumerate(delta):
        sbr = sbr_calc(radi,RHI,x,dx,vt,Rs)
        Mass_guess[i] = (integrate.simps(sbr*2*np.pi*radi,radi)*1000.**2.)

    Mj = np.argmin(abs(Mass_guess**2.- mass**2.))
    dx = delta[Mj]

    sbr = sbr_calc(radi,RHI,x,dx,vt,Rs)
    Mass_guess = (integrate.simps(sbr*2*np.pi*radi,radi)*1000.**2.)
    #print('Mass %:',100*Mass_guess/mass)
    
    return sbr

def make_z(radi,vrot,sigma):
    z = np.zeros_like(radi)
    for i, r in enumerate(radi):
        if i != 0:
            z[i] = sigma / ( np.sqrt(2./3.) * vrot[i]/r)
        if i == 0:
            z[i] = sigma / ( np.sqrt(2./3.) * vrot[1]/radi[1])
    return  z

def Magcalc(vrot,hr,Rmax):
    A=np.array([0.008,0.002,0.003,0.002,0.011,0.022,0.010,0.020,0.029,0.019])
    dA = np.array([0.003,0.001,0.001,0.001,0.001,0.002,0.003,0.005,0.005,0.015])
    V0 = np.array([275.,255.,225.,200.,170.,148.,141.,122.,103.,85.])
    dV0 = np.array([6.,2.,1.,1.,1.,2.,2.,2.,2.,5.])
    rPE = np.array([0.126,0.132,0.149,0.164,0.178,0.201,0.244,0.261,0.260,0.301])
    drPE = np.array([0.007,0.003,0.003,0.002,0.003,0.004,0.005,0.008,0.008,0.002])
    m=np.array([-23.76,-23.37,-22.98,-22.60,-22.19,-21.80,-21.41,-21.02,-20.48,-19.38])

    V0, foo = curve_fit(func, m, V0,sigma=dV0)
    rPE,foo = curve_fit(func, m, rPE,sigma=drPE)
    A, foo  = curve_fit(func2, m, A,sigma=dA)
    Mag = np.arange(-24.,0.,0.1)
    vt=func(Mag,*V0)
    Mag = round(Mag[np.argmin(abs(vt-vrot))],2)
    rt=hr*func(Mag,*rPE)
    a=func2(Mag,*A)
    vt=func(Mag,*V0)
    vRmax = np.max(vt*(1.-np.exp(-Rmax/rt))*(1.+a*Rmax/rt))
    return Mag

def phi(MHI, Mstar, alpha, phi_0):
    #Mass Function
    return np.log(10.) *phi_0* (MHI/Mstar)**(alpha+1.) * np.exp(-MHI/Mstar)

def DHI_calc(MHI, slope,intercept):
    #Diameter-Mass relation
    return (10.**(slope*np.log10(MHI)+intercept))

def Mstar_calc(Mgas,slope,const,split):
    #Stellar Mass calculator
    mass = np.log10(Mgas)
    if mass < split:
        Mstar = mass * 1./slope[0,0] - const[0,0]/slope[0,0]
    else:
        Mstar = mass * 1./slope[1,0] - const[1,0]/slope[1,0]
    return 10.** Mstar       

def BTFR(Mbar,slope,const):
    #Baryonic Tully Fisher
    logv = np.log10(Mbar) * 1./slope[0] - const[0]/slope[0]
    return 10.**(logv)

def TFR(slope,const,vflat):
    Mag = slope[0] * (np.log10(vflat) -2.3)+const[0]
    return Mag

def expdisk(a,b,g,M0,Mass):
    #scale length for the polyex fit
    return g * (Mass)**a * (1. + Mass/M0)**(b-a)
    

def setup_relations(mass,beams,thicc):
    ####################################
    MHI = np.round(10.**(np.arange(6.,11.1,.1)),1)
    mass=10.**float(mass)
    i = np.array(np.argmin(abs(float(mass)-MHI)))
    MHI = MHI[i]
    ####################################
    # Martin et al 2010 
    # https://arxiv.org/abs/1008.5107
    phi_0           = 0.0048 #\pm 0.3E-3
    Mstar           = 10.**9.96 #\pm 0.02 dex
    alpha           = -1.33 #\pm 0.02
    HIMF = phi(MHI,Mstar,alpha,phi_0)
    # MPC^-3 dex^-1
    ####################################
    # Jing Wang, Koribalski, et al 2016
    # https://arxiv.org/abs/1605.01489
    slope       =   0.506
    const       =  -3.293
    
    DHI = DHI_calc(MHI,slope, const)
    print('HI Radius:','{:.4}'.format(DHI/2.*u.kpc))
    ####################################
    # Bradford et al 2015, right after eq 4
    # https://arxiv.org/abs/1505.04819
    split           = 9.2832
    Mgas            = MHI * 1.4
    slope = np.array([[1.052,0.058],[0.461,0.011]])
    const = np.array([[0.236,0.476],[5.329,0.112]])
    Mstar = Mstar_calc(Mgas,slope,const,split)
    Mbar = Mstar + Mgas
    ####################################
    # Lelli et al 2015
    # https://arxiv.org/abs/1512.04543
    #slope = np.array([3.71,0.08])
    #const = np.array([2.27,0.18])
    slope = np.array([3.95,0.34])
    const = np.array([1.85,0.60])
    v_flat = BTFR(Mbar,slope,const)
    print('Vflat:','{:.4}'.format(v_flat*u.km/u.s))
    slope = np.array([0.277,0.004])
    const = np.array([-0.672,0.041])
    v_test = np.log10(Mbar) * slope[0] + const[0]
    print('Vflat:','{:.4}'.format((10**v_test)*u.km/u.s))
    ####################################
    # Jing et al 2014
    # https://arxiv.org/abs/1401.8164
    # HI scale length = 0.18 RHI
    Rs = (DHI/2.)*0.18
    ####################################
    # Mosleh and Franx 2013
    # https://arxiv.org/abs/1302.6240
    # 'Sersic n < 2.5'
    alpha   = np.array([0.124,0.081])
    beta    = np.array([0.278,0.161])
    gamma   = 10.**np.array([-0.874,0.756])
    M0      = 10.**np.array([10.227,0.230])
    
    Rd = 3.31*(expdisk(alpha[0],beta[0],gamma[0],M0[0],Mstar))
    ###################################
    # Approximate stuff
    # This is from Papastergis & Shankar 2012
    # Based on where the Vmax,HI is measured.
    # Smaller galaxies have Rmax at a higher
    # fraction of DHI than more massive galaxies
    Rmax = (DHI/2.) / (np.log10(MHI) -5.)
    ###################################
    # Velocity dispersion to range from 8km/s
    # for most massive, to 20km/s for least
    # massive
    Vdisp = 8.#np.round(17.714 - 0.952*np.log10(MHI),2)
    print('Vdisp=',Vdisp*u.km/u.s)
    ###################################
    # Calculating Magnitude from vmax
    # Catinella et al 2005
    # https://arxiv.org/abs/astro-ph/0512051
    Mag = Magcalc(v_flat,Rd,Rmax)
    print('I-Band Magnitude (guess):','{:.4}'.format(Mag))
    # Alternatively, TFR:
    # Ouellette et al 2017
    # https://arxiv.org/abs/1705.10794
    slope = np.array([-7.68,0.58])
    const = np.array([-22.38,0.33])
    scatter = 0.222

    Mag = TFR(slope,const,v_flat)
    print('I-Band Magnitude (TFR):','{:.4}'.format(Mag))


    dist = DHI / (4.*beams * (np.pi/162000.))

    dist = dist * u.kpc
    dist = dist.to_value(u.kpc)

    print('distance [kpc]', round(dist,2))
    delta = ((thicc*u.arcsec).to_value(u.rad)*(dist))* u.kpc
    print('ringsize',np.round(delta/(1*u.kpc),4))
    print('rings',np.round(DHI*u.kpc /delta,2))
    radi = np.arange(0.,DHI,delta/u.kpc)

    vrot = make_vrot(radi,Mag,Rd,v_flat)
    sbr  = (1./dist)*(1/0.236)*make_sbr(radi,Rs,DHI,v_flat,mass)
    print('Sigma0',round(np.max(sbr),6))
    z    = make_z(radi,vrot,Vdisp)

    #plt.semilogy(radi,sbr)
    #plt.show()
    plt.plot(radi,vrot)
    #plt.xlabel('R [kpc]')
    #plt.ylabel('Vc [km/s]')
    #plt.axvline(DHI/2.)
    plt.show()
    #plt.savefig('rot.png')

    rothead(MHI,Mag,Vdisp,Mbar,Mstar,DHI,v_flat,Rs,dist)
    return radi, sbr, vrot, Vdisp, z, MHI, DHI, Mag, dist
