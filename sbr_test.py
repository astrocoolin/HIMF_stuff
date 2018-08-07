#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from scipy.optimize import curve_fit
from scipy import integrate

def func(x,a,b,c,d):
    #exponential fit for V0, rPE
    return a*np.exp(-x*b+c) +d
    #return a*x + b


def func2(x,a,b,c,d):
    #quadratic fit for A
    #return a*x + b
    #return a*np.exp(-x*b+c) +d
    return a*x**2. + b*x + c + d*x**3.

def make_vrot(radi,Mag,hr):
    A=np.array([0.007,0.006,0.005,0.012,0.021,0.028,0.028,0.033,0.042,0.087])
    V0 = np.array([270.,248.,221.,188.,161.,143.,131.,116.,97.,64.])
    rPE = np.array([0.37,0.40,0.48,0.48,0.52,0.64,0.73,0.81,0.80,0.72])
    m=np.array([-23.8,-23.4,-23.,-22.6,-22.2,-21.8,-21.4,-21.,-20.4,-19])

    V0, foo = curve_fit(func, m, V0)
    rPE,foo = curve_fit(func, m, rPE)
    A, foo  = curve_fit(func2, m, A)

    vt=func(Mag,*V0)
    rt=hr*func(Mag,*rPE)
    a=0.#func2(Mag,*A)

    #print('vt[km/s]:',round(vt,2))
    #print('rt[arcsec]:',round(rt,2))
    #print('a:',round(a,2))

    return vt*(1.-np.exp(-radi/rt))*(1.+a*radi/rt)


def make_sbr_old(radi,Sig0,Rs,Rb,DHI):
    sbr = np.zeros_like(radi)
    for j, r in enumerate(radi):
        #if r < 0.1*DHI/2.:
        #    sbr[j] = 0.
        if r < Rb:
            sbr[j] = Sig0
        else:
            sbr[j] = (Sig0/np.exp(-Rb/Rs)) * np.exp(-r/Rs)
    return sbr


def make_z(radi,vrot,sigma):
    return sigma / ( np.sqrt(2./3.) * vrot/radi)
    

def Magcalc(vrot,hr,Rmax):
    A=np.array([0.007,0.006,0.005,0.012,0.021,0.028,0.028,0.033,0.042,0.087])
    V0 = np.array([270.,248.,221.,188.,161.,143.,131.,116.,97.,64.])
    rPE = np.array([0.37,0.40,0.48,0.48,0.52,0.64,0.73,0.81,0.80,0.72])
    m=np.array([-23.8,-23.4,-23.,-22.6,-22.2,-21.8,-21.4,-21.,-20.4,-19])
    
   # plt.scatter(m,A)

    V0, foo = curve_fit(func, m, V0)
    rPE,foo = curve_fit(func, m, rPE)
    A, foo  = curve_fit(func2, m, A)
    Mag = np.arange(-30.,0.,0.1)
    vt=func(Mag,*V0)
    Mag = round(Mag[np.argmin(abs(vt-vrot))],2)
    rt=hr*func(Mag,*rPE)
    a=func2(Mag,*A)
    
    #mee = np.arange(-30,0,0.5)
    #plt.plot(mee,func2(mee,*A))
    #plt.show()

    vt=func(Mag,*V0)
    vRmax = np.max(vt*(1.-np.exp(-Rmax/rt))*(1.+a*Rmax/rt))
    return Mag, vRmax, vt

def HI_profile(R1,Mass):
    #Have to find Sigma_centre
    #Integrated Flux should give you the mass of the galaxy
    Rb = 0.75 * R1
    Rs = R1 * 0.18

    out = 2.0*R1
    ins = 0.1*R1

    inside  = 2.*np.pi*1000.**2. *np.exp((R1-Rb )/Rs) *  (Rb**2.  - ins**2.)
    exp     = 2.*np.pi*1000.**2. *np.exp((R1-Rb )/Rs) * (Rs*(Rb+Rs ))
    edge    = 2.*np.pi*1000.**2. *np.exp((R1-out)/Rs) * (Rs*(out+Rs))

    MGuess =  (inside + exp - edge)
    #print('inside,exp',inside/MGuess,(exp+edge)/MGuess)
    Fluxc = np.exp((R1 - Rb)/Rs) 
    #print(str(round(MGuess/Mass,2)*100.)+'% of the mass')
    return Rs, Fluxc, Rb

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

def expdisk(a,b,g,M0,Mass):
    #scale length for the polyex fit
    return g * (Mass)**a * (1. + Mass/M0)**(b-a)

for mass in (5.5,6.,6.5,7.,7.5,8.,8.5,9.,9.5,10.,10.5):
#for mass in (7.5,):
    #mass=10.0
    dist = 12.E6
    thicc=0.05
    ####################################
    MHI = np.round(10.**(np.arange(5.5,11.1,.1)),1)
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
    print('Stellar Mass:','{:.4}'.format(Mstar*u.Msun))
    print('Baryonic Mass:','{:.4}'.format(Mbar*u.Msun))
    ####################################
    # Lelli et al 2015
    # https://arxiv.org/abs/1512.04543
    #slope = np.array([3.71,0.08])
    #const = np.array([2.27,0.18])
    slope = np.array([3.95,0.34])
    const = np.array([1.85,0.60]) 
    v = BTFR(Mbar,slope,const)
    #print('Vflat:','{:.4}'.format(v*u.km/u.s))
    ####################################
    # Jing et al 2014
    # https://arxiv.org/abs/1401.8164
    # HI scale length = 0.18 RHI
    Rs = (DHI/2.)*0.18
    #print('HI Scale length:','{:.4}'.format(Rs*u.kpc))
    #print('Central density:','{:.4}'.format(Sig0*u.Msun/u.pc**2.))
    ####################################
    # Mosleh and Franx 2013
    # https://arxiv.org/abs/1302.6240
    # 'Sersic n < 2.5'
    alpha   = np.array([0.124,0.081])
    beta    = np.array([0.278,0.161])
    gamma   = 10.**np.array([-0.874,0.756])
    M0      = 10.**np.array([10.227,0.230])
    
    Rd = (expdisk(alpha[0],beta[0],gamma[0],M0[0],Mstar))
    #print('Optical Scale length:','{:.4}'.format(Rd*u.kpc))
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
    Vdisp = np.round(17.714 - 0.952*np.log10(MHI),2)
    #print('Vdisp=',Vdisp*u.km/u.s)
    ###################################
    # Calculating Magnitude from vmax
    # https://arxiv.org/abs0512051
    Mag, vRmax, vt = Magcalc(v,Rd,Rmax)
    #print("Approx measured V:",round(vRmax,2)*u.km/u.s,', @ R=',round(Rmax,2)*u.kpc)
    #print('B-Band Magnitude:','{:.4}'.format(Mag))
    
    dist = dist * u.pc
    #print('distance [kpc]', dist.to_value(u.kpc))
    delta = ((thicc*u.arcsec).to_value(u.rad)*(dist.to_value(u.kpc)))* u.kpc
    #print('ringsize',delta)
    #print('rings',DHI*u.kpc /delta)
    print('vt=',vt,'DHI=',DHI,'Mass=',np.log10(mass))
    print('Mag=',Mag)
    radi = np.arange(0.,DHI,delta/u.kpc)
    
    vrot = make_vrot(radi,Mag,Rd)
    
    def make_sbr(radi,Rs,DHI,vt,mass):
        RHI=DHI/2.
        x=0.36
        delta = np.arange(-0.045,-0.03,0.001)
        Mass = np.zeros_like(delta)
        for i, dx in enumerate(delta):
            sbr = np.zeros_like(radi)
            for j, r in enumerate(radi):
                sig1 = np.exp(-((r-0.4*RHI)/(np.sqrt(2)*(x+dx)*RHI))**2.)
                sig2 = (np.sqrt(vt/120.)-1.)*np.exp(-r/Rs)
                sbr[j] = sig1-sig2
            R_HI= np.argmin(abs(radi**2. - RHI**2.))
            sbr = sbr/sbr[R_HI]
            Mass[i] = (integrate.simps(sbr*2*np.pi*radi,radi)*1000.**2.)
        Mj = np.argmin(abs(Mass**2.- mass**2.))
        dx = delta[Mj]
        for j, r in enumerate(radi):
            sig1 = np.exp(-((r-0.4*RHI)/(np.sqrt(2)*(x+dx)*RHI))**2.)
            sig2 = (np.sqrt(vt/120.)-1.)*np.exp(-r/Rs)
            sbr[j] = sig1-sig2
        R_HI= np.argmin(abs(radi**2. - RHI**2.))
        sbr = sbr/sbr[R_HI]

        return sbr
    
    sbr  = make_sbr(radi,Rs,DHI,vt,mass)
    z    = make_z(radi,vrot,Vdisp)
    plt.figure(1)
    plt.semilogy(radi/(DHI/2.),sbr,label=np.log10(mass))
    plt.axhline(1)
    plt.axvline(1)
    plt.figure(2)
    plt.plot(radi,vrot,label=np.log10(mass))

plt.figure(1)
plt.xlabel('R/R$_{HI}$')
plt.ylabel('$\Sigma$ [Msun / pc^2]')
plt.legend()
plt.figure(2)
plt.xlabel('Radi [kpc]')
plt.ylabel('Vrot [km/s]')
plt.legend()
plt.show()

