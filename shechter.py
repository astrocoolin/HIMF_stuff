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
    #Integrated Flux should give you the mass of the galaxy
    b = np.zeros_like(R1)
    bco = np.arange(.65,.75,0.001)
    for j, R in enumerate(R1):
        Rb = bco * R
        Rs = R * 0.20
        out = 1.5*R
        ins = 0.1*R
        #ex1 = Rs/(R-Rb)
        #ex=(Rb/R)**2.
        #ex2 = -(ex)*Rs/(R-Rb)
        #a = Rb/R
        #w = (Rs/(R-Rb))
        #print(np.min(-a**2./w),np.max(-a**2./w))
        inside  = 2.*np.pi*1000.**2. *np.exp((R-Rb )/Rs) *  (Rb**2.  - ins**2.)
        #inside  = 2.*np.pi*1000.**2. *(0.5)*(w)*(1.-np.exp(-a**2./w))
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
            #print(ex[k],ex2[k])
            print('{:.3e}'.format(MGuess_i),'{:.3e}'.format(Mass[j]),'{:.3e}'.format(inside_i),'{:.3e}'.format(exp_i),'{:.3e}'.format(edge_i))
            #print('poop',Mass[j],bco[k],MGuess_i)

    Rs   = (R1)*0.18
    Fluxc = np.exp((R1 - b)/Rs) 
    #print(Fluxc,b/R1)
    print(MGuess_i/Mass[i]*100,'%, Mass % outside '+ str(out/R)+'*R1=',edge_i/Mass[i]*100.,'%','left out mass', np.log10(edge_i), 'dex, b=',b[i]/R1[i])

    return Rs, Fluxc,b

def nearest_guess(find,Mass):
    #distmin = 9E99
    np.argmin(Mass-find)
    #for i, valu in enumerate(Mass):
    #    dist = abs(valu**2. - find**2.)
    #    if dist < distmin:
    #        distmin = dist
    #        j = i
    return np.array([j])

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
DHI_slope       =   0.506
DHI_intercept   =  -3.293
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
    if r < 0.1*DHI[i]/2.:
        sbr[j] = 0.
    elif r < b[i]:
        sbr[j] = Sig0[i]
        #sbr[j] = Sig0[i]*np.exp(-(r/Rs[i])**2 * (DHI[i]/2. - b[i])/Rs[i])
    else:
        sbr[j] = (Sig0[i]/np.exp(-b[i]/Rs[i])) * np.exp(-r/Rs[i])

plt.semilogy(radi/(DHI[i]/2.),sbr)
plt.axvline(1.)
plt.axhline(1.)
plt.title(str(np.log10(MHI[i]))+' dex Solar Massses of HI')
plt.xlabel('R / R1,  R1 ='+str(DHI[i]/2.)+' kpc')
plt.ylabel('HI Mass Density [Msun pc$^{-2}$]')
plt.xlim(0,2.)
plt.ylim(0.2,10)
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
