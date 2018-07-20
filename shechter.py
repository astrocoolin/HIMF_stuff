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

def expdisk(R1,Mass,i):
    #Have to find Sigma_centre
    #b = 0.8 * (R1)
    b = np.zeros_like(R1)
    bco = np.arange(0.5,1.01,0.01)
    #b = (R1) np.arange(0.5,1.,0.01)
    #Flux should give you the mass of the galaxy
    for j, R in enumerate(R1):
        for k, b_temp in enumerate(bco):
            Rb = b_temp * R
            Rs = R * 0.18
            out = 1.5*R
            minus = 1000.**2.*Rs*(out+Rs)*2.*np.pi*np.exp((R-out)/R)
            MGuess=2.*np.pi*np.exp((R-Rb)/Rs)*((Rb*1000.)**2.+1000.**2.*Rs*(Rb+Rs)) - minus
            if j==i:
                MGuess_i=MGuess
                minus_i = minus
            if abs(100*((MGuess-Mass[j])/Mass[j]) ) < 10.:
                b[j] = Rb
                #if j==i:
                    #print(100*((MGuess_i-Mass[j])/Mass[j]),MGuess_i/Mass[i]*100,np.array([MGuess_i]),Mass[i],R,b_temp)
                break
    Rs   = (R1)*0.18
    Fluxc = np.exp((R1 - b)/Rs) 
    print(Fluxc,b/R1)
    print(MGuess_i/Mass[i]*100,'% subtract part',minus_i/Mass[i]*100.,'%')

    return Rs, Fluxc,b
#def expdisk(R1,Mass,i):
#    #Have to find Sigma_centre
#    #b = 0.8 * (R1)
#    b = np.zeros_like(R1)
#    bco = np.arange(0.5,1.01,0.01)
#    #b = (R1) np.arange(0.5,1.,0.01)
#    #Flux should give you the mass of the galaxy
#    for j, R in enumerate(R1):
#        for k, b_temp in enumerate(bco):
#            Rb = b_temp * R
#            Rs = R * 0.18
#            out = 1.5*R
#            minus = 1000.**2.*Rs*(out+Rs)*2.*np.pi*np.exp((R-out)/R)
#            MGuess=2.*np.pi*np.exp((R-Rb)/Rs)*((Rb*1000.)**2.+1000.**2.*Rs*(Rb+Rs)) - minus
#            if j==i:
#                MGuess_i=MGuess
#                minus_i = minus
#            if abs(100*((MGuess-Mass[j])/Mass[j]) ) < 10.:
#                b[j] = Rb
#                #if j==i:
#                    #print(100*((MGuess_i-Mass[j])/Mass[j]),MGuess_i/Mass[i]*100,np.array([MGuess_i]),Mass[i],R,b_temp)
#                break
#    Rs   = (R1)*0.18
#    Fluxc = np.exp((R1 - b)/Rs) 
#    print(Fluxc,b/R1)
#    print(MGuess_i/Mass[i]*100,'% subtract part',minus_i/Mass[i]*100.,'%')
#
#    return Rs, Fluxc,b


#def expdisk(R1,Mass,i):
#    Rs   = (R1)*0.18
#    #Have to find Sigma_centre
#    b = 0.8 * (R1)
#    bco = np.arange(0.5,1.01,0.01)
#    #Flux should give you the mass of the galaxy
#    Fluxc = np.exp((R1 - b)/Rs) #Mass / (2. * np.pi * (0.5*( b*1000.)**2. + Rs*1000.*(Rs*1000.+b*1000.)*np.exp(R1/Rs)*np.exp(-b/Rs)))
#    #print('Flux',Fluxc)
#    #Sig0 = 1./np.exp(-(DHI/2.)/Rs)
#    out = 2.*R1[i]
#    minus = 1000.**2.*Rs[i]*(out+Rs[i])*2.*np.pi*np.exp((R1[i]-out)/Rs[i])
#    MGuess=2.*np.pi*np.exp((R1[i]-b[i])/Rs[i])*((b[i]*1000.)**2.+1000.**2.*Rs[i]*(b[i]+Rs[i])) - minus
#    print(MGuess/Mass[i]*100,'% subtract part',minus/Mass[i]*100.,'%')
#    return Rs, Fluxc,b

def nearest(find,Mass):
    #Mtest = 2. * np.pi * Rs**2. * Sig0 * 0.236 * 1000.**2.
    # I need Sigma_0 and the high point ~4.5 Msun/pc^2
    #Mtest = 2. * np.pi * Rs (Rs + b)*np.exp(b/Rs)
    #print('Mtest',Mtest[i],'%',Mtest[i]/MHI[i])
    distmin = 1.E15
    for i, valu in enumerate(np.log10(Mass)):
        dist = abs(valu**2. - find**2.)
        if dist < distmin:
            distmin = dist
            j = i
    return np.array([j])


####################################
MHI = np.round(10.**(np.arange(4.,12.1,.1)),1)
i=input('Input HI Mass (dex):\n')
print(10.**float(i))
#i=np.where(MHI==round(10.**float(i),0))
#i=np.where(float(i)==np.log10(MHI))
i = nearest(float(i),MHI)
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
# Bradford et al 2015, right after eq 4
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
Rs, Sig0, b = expdisk(DHI/2.,MHI,i)
print('HI Scale length:',Rs[i]*u.kpc)
print('Central density:',Sig0[i]*u.Msun/u.pc**2.)

radi = np.arange(0.,DHI[i],0.001)
sbr = np.zeros_like(radi)
for j, r in enumerate(radi):
    if r < b[i]:
        sbr[j] = Sig0[i]
    else:
        sbr[j] = (Sig0[i]/np.exp(-b[i]/Rs[i])) * np.exp(-r/Rs[i])

plt.semilogy(radi/(DHI[i]/2.),sbr)
plt.axvline(1.)
plt.axhline(1.)
plt.title(str(np.log10(MHI[i]))+' dex Solar Massses of HI')
plt.xlabel('R / R1,  R1 ='+str(DHI[i]/2.)+' kpc')
plt.ylabel('HI Mass Density [Msun pc$^{-2}$]')
plt.xlim(0,2.)
plt.ylim(0.1,10)
plt.show()
#
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
