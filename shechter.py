#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

def func(x,a,b,c,d):
    #exponential fit for V0, rPE
    return a*np.exp(-x*b+c) +d

def func2(x,a,b,c):
    #quadratic fit for A
    return a*x**2. + b*x + c

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
    a=func2(Mag,*A)

    print('vt[km/s]:',round(vt,2))
    print('rt[arcsec]:',round(rt,2))
    print('a:',round(a,2))

    return vt*(1.-np.exp(-radi/rt))*(1.+a*radi/rt)

def HI_profile(R1,Mass,i):
    #Have to find Sigma_centre
    #Integrated Flux should give you the mass of the galaxy
    b = np.zeros_like(R1)
    bco = np.arange(.65,.75,0.001)
    for j, R in enumerate(R1):
        Rb = bco * R
        Rs = R * 0.18

        out = 2.0*R
        ins = 0.1*R

        inside  = 2.*np.pi*1000.**2. *np.exp((R-Rb )/Rs) *  (Rb**2.  - ins**2.)
        exp     = 2.*np.pi*1000.**2. *np.exp((R-Rb )/Rs) * (Rs*(Rb+Rs ))
        edge    = 2.*np.pi*1000.**2. *np.exp((R-out)/Rs) * (Rs*(out+Rs))

        MGuess =  (inside + exp - edge)
        k =np.argmin(abs(Mass[j]-MGuess))
        b[j] = bco[k]*R

        if j==i:
            Rb = b[j]* R
            inside_i=inside[k]
            exp_i=exp[k]
            edge_i=edge
            MGuess_i=MGuess[k] 

    Rs   = (R1)*0.18
    Fluxc = np.exp((R1 - b)/Rs) 
    print(str(round(MGuess_i/Mass[i],2)*100.)+'% of the mass')
    return Rs, Fluxc,b

#def expdisk(slope,const,Mass):
#    #expdisk for the polyex fit
#    return 10.**(slope * (np.log10(Mass)-10.) + const)


####################################
MHI = np.round(10.**(np.arange(4.,12.1,.1)),1)
i=input('Input HI Mass (dex):\n')
i=10.**float(i)
i = np.array(np.argmin(abs(float(i)-MHI)))
print('HI Mass:','{:.4}'.format(MHI[i]*u.Msun))
####################################
# Martin et al 2010 
# https://arxiv.org/abs/1008.5107
phi_0           = 0.0048 #\pm 0.3E-3
Mstar           = 10.**9.96 #\pm 0.02 dex
alpha           = -1.33 #\pm 0.02

def phi(MHI, Mstar, alpha, phi_0):
    #Mass Function
    return np.log(10.) *phi_0* (MHI/Mstar)**(alpha+1.) * np.exp(-MHI/Mstar)

HIMF = phi(MHI,Mstar,alpha,phi_0)
# MPC^-3 dex^-1
# print(10.**HIMF)
####################################
# Jing Wang, Koribalski, et al 2016
# https://arxiv.org/abs/1605.01489
slope       =   0.506
const       =  -3.293

def DHI(MHI, slope,intercept):
    #Diameter-Mass relation
    return (10.**(slope*np.log10(MHI)+intercept))

DHI = DHI(MHI,slope, const)
print('HI Radius:','{:.4}'.format(DHI[i]*u.kpc))
####################################
# Bradford et al 2015, right after eq 4
# https://arxiv.org/abs/1505.04819
split           = 9.2832
Mgas            = MHI * 1.4

def MBar_calc(Mgas,slope,const,split):
    #Baryonic Mass calculator
    MBar = np.zeros_like(MHI)
    for i, mass in enumerate(np.log10(MHI)):
        if mass < split:
            MBar[i] = mass * 1./slope[0,0] - const[0,0]/slope[0,0]
        else:
            MBar[i] = mass * 1./slope[1,0] - const[1,0]/slope[1,0]
    return 10.** MBar       

slope = np.array([[1.052,0.058],[0.461,0.011]])
const = np.array([[0.236,0.476],[5.329,0.112]])
Mbar = MBar_calc(Mgas,slope,const,split) + Mgas
print('Baryonic Mass:','{:.4}'.format(Mbar[i]*u.Msun))
####################################
# Lelli et al 2015
# https://arxiv.org/abs/1512.04543

def BTFR(Mbar,slope,const):
    #Baryonic Tully Fisher
    logv = np.log10(Mbar) * 1./slope[0] - const[0]/slope[0]
    return 10.**(logv)

slope = np.array([3.71,0.08])
const = np.array([2.27,0.18])
v = BTFR(Mbar,slope,const)
print('Vflat:','{:.4}'.format(v[i]*u.km/u.s))
####################################
# Jing et al 2014
# https://arxiv.org/abs/1401.8164
# scale length = 0.18 RHI
Rs, Sig0, b = HI_profile(DHI/2.,MHI,i)
print('HI Scale length:','{:.4}'.format(Rs[i]*u.kpc))
print('Central density:','{:.4}'.format(Sig0[i]*u.Msun/u.pc**2.))
####################################
# Wu 2017
# https://arxiv.org/abs/1710.06440
slope = np.array([0.385,0.013])
const = np.array([0.281,0.010])

def expdisk(slope,const,Mass):
    #expdisk for the polyex fit
    return 10.**(slope * (np.log10(Mass)-10.) + const)

Rd = expdisk(slope[0],const[0],MHI)
print('Optical Scale length:','{:.4}'.format(Rd[i]*u.kpc))

###################################
radi = np.arange(0.,DHI[i],0.001)
sbr = np.zeros_like(radi)
for j, r in enumerate(radi):
    if r < 0.1*DHI[i]/2.:
        sbr[j] = 0.
    elif r < b[i]:
        sbr[j] = Sig0[i]
    else:
        sbr[j] = (Sig0[i]/np.exp(-b[i]/Rs[i])) * np.exp(-r/Rs[i])

