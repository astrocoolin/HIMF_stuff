#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

def func(x,a,b,c,d):
    return a*np.exp(-x*b+c) +d

def func2(x,a,b,c):
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

def phi(MHI, Mstar, alpha, phi_0):
    return np.log(10.) *phi_0* (MHI/Mstar)**(alpha+1.) * np.exp(-MHI/Mstar)

def DHI(MHI, slope,intercept):
    return (10.**(slope*np.log10(MHI)+intercept))

def MBar_calc(Mgas,slope,const,split):
    MBar = np.zeros_like(MHI)
    for i, mass in enumerate(np.log10(MHI)):
        if mass < split:
            MBar[i] = mass * 1./slope[0,0] - const[0,0]/slope[0,0]
        else:
            MBar[i] = mass * 1./slope[1,0] - const[1,0]/slope[1,0]
    return 10.** MBar       


def BTFR(Mbar,slope,const):
    logv = np.log10(Mbar) * 1./slope[0] - const[0]/slope[0]
    #print(np.log10(Mbar))
    #print(10.**logv)
    return 10.**(logv)

def expdisk(DHI):
    Rs   = DHI*0.18*0.5
    Sig0 = 1./np.exp(-(DHI/2.)/Rs)
    return Rs, Sig0

####################################
MHI = np.round(10.**(np.arange(5.,11.,.1)),1)
i=input('Input HI Mass (dex):\n')
i=np.where(MHI==round(10.**float(i),1))
print('HI Mass:',MHI[i]*u.Msun)
####################################
# Martin et al 2010 
# https://arxiv.org/abs/1008.5107
phi_0           = 0.0048 #\pm 0.3E-3
Mstar           = 10.**9.96 #\pm 0.02 dex
alpha           = -1.33 #\pm 0.02
HIMF = phi(MHI,Mstar,alpha,phi_0)
# MPC^-3 dex^-1
# print(10.**HIMF)
####################################
# Jing Wang, Koribalski, et al 2016
# https://arxiv.org/abs/1605.01489
DHI_slope       =   0.51
DHI_intercept   =  -3.32
DHI = DHI(MHI,DHI_slope, DHI_intercept)
print('HI Radius:',DHI[i]*u.kpc)
####################################
# Bradford et al 2015
# https://arxiv.org/abs/1505.04819
split           = 9.2832
Mgas            = MHI * 1.4
slope = np.array([[1.052,0.058],[0.461,0.011]])
const = np.array([[0.236,0.476],[5.329,0.112]])
Mbar = MBar_calc(Mgas,slope,const,split) + MHI
print('Baryonic Mass:',Mbar[i]*u.Msun)
####################################
# Ouellette et al 2017
# https://arxiv.org/abs/1705.10794
# slope = np.array([3.57,0.16])
# const = np.array([2.44,0.35])
# This one seems to fit better:
# Lelli et al 2015
# https://arxiv.org/abs/1512.04543
slope = np.array([3.71,0.08])
const = np.array([2.27,0.18])
v = BTFR(Mbar,slope,const)
print('Vflat:',v[i]*u.km/u.s)
####################################
# Jing et al 2014
# https://arxiv.org/abs/1401.8164
# scale length = 0.18 RHI
Rs, Sig0 = expdisk(DHI)
print('HI Scale length:',Rs[i]*u.kpc)
print('Central density:',Sig0[i]*u.Msun/u.pc**2.)





plt.plot(np.log10(MHI),np.log10(HIMF))
plt.xlim(6,11)
plt.ylim(-6,0)
#plt.show()

plt.plot(np.log10(MHI),np.log10(DHI))
#plt.show()

plt.plot(np.log10(MHI),np.log10(Mbar))
#plt.show()

plt.plot(np.log10(Mbar),np.log10(v))
#plt.show()

#print(np.log10(HIMF))
#print(np.log10(MHI))
#print(np.log10(v))
