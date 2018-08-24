#!/usr/bin/env python3
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import astropy.units as u
from scipy.optimize import curve_fit
from scipy import integrate
from Input_output import rothead

def func(x,a,b,c,d):
    #exponential fit for V0, rPE
    return a*np.exp(-x*b-c) +d*x

def func2(x,b,c):
    #Linear fit for A, Rt
    return b*x + c

def func3(x,a,b,c,d,e):
    #exponential fit for V0, rPE
    #return  -a*np.exp(x*b-c) -d*x
    return a*np.log10(-x) - b

def make_vrot(radi,Mag,Rd,v,RHI,mstar):
    V0 = np.array([275.,255.,225.,200.,170.,148.,141.,122.,103.,85.])
    dV0 = np.array([6.,2.,1.,1.,1.,2.,2.,2.,2.,5.])
    A=np.array([0.008,0.002,0.003,0.002,0.011,0.022,0.010,0.020,0.029,0.019])
    dA = np.array([0.003,0.001,0.001,0.001,0.001,0.002,0.003,0.005,0.005,0.015])
    rPE = np.array([0.126,0.132,0.149,0.164,0.178,0.201,0.244,0.261,0.260,0.301])
    drPE = np.array([0.007,0.003,0.003,0.002,0.003,0.004,0.005,0.008,0.008,0.002])
    m=np.array([-23.76,-23.37,-22.98,-22.60,-22.19,-21.80,-21.41,-21.02,-20.48,-19.38])

    V0, foo = curve_fit(func, m, V0,sigma=dV0)
    rPE,foo = curve_fit(func3, m, rPE,sigma=drPE)
    A, foo  = curve_fit(func2, m, A,sigma=dA)

    vt=func(Mag,*V0)
    a=func2(Mag,*A)
    rt=Rd*func3(Mag,*rPE) 

    return vt*(1.-np.exp(-radi/rt))*(1.+a*radi/rt),rt

def Magcalc(vrot,Rd,RHI,mstar):
    # Find Mag, and slope based on
    # Catinella et al 2007
    # https://arxiv.org/abs/astro-ph/0512051
    # Dutton et al 2018
    # https://arxiv.org/abs/1807.10518
    A_=np.array([0.008,0.002,0.003,0.002,0.011,0.022,0.010,0.020,0.029,0.019])
    dA = np.array([0.003,0.001,0.001,0.001,0.001,0.002,0.003,0.005,0.005,0.015])
    rPE_ = np.array([0.126,0.132,0.149,0.164,0.178,0.201,0.244,0.261,0.260,0.301])
    drPE = np.array([0.007,0.003,0.003,0.002,0.003,0.004,0.005,0.008,0.008,0.002])
    V0_ = np.array([275.,255.,225.,200.,170.,148.,141.,122.,103.,85.])
    dV0 = np.array([6.,2.,1.,1.,1.,2.,2.,2.,2.,5.])
    m=np.array([-23.76,-23.37,-22.98,-22.60,-22.19,-21.80,-21.41,-21.02,-20.48,-19.38])

    # Best fitting lines
    V0, foo  = curve_fit(func , m, V0_,sigma=dV0)
    rPE, foo = curve_fit(func3, m, rPE_,sigma=drPE)
    A, foo   = curve_fit(func2, m, A_,sigma=dA)

    # Make parameters for all Magnitudes
    Mag = np.arange(-25,0.,0.001)
    a=func2(Mag,*A)
    vt_0=func(Mag,*V0)
    rt=Rd*func3(Mag,*rPE)

    if False:
        plt.plot(Mag,rt/Rd)
        plt.scatter(m,rPE_)
        plt.show()

    # Optical Radius, also Slope from NIHAO 17 
    R_opt = (3.31+err(0.01))*Rd
    slope_sparc = 0.123 - 0.137*(np.log10(mstar)-9.471) + err(0.19)

    # Find Vrot, then Alpha, then check again to make sure Vrot is consistent
    for i in range(0,6):
        # Make parameters for all Magnitudes
        Mag = np.arange(-25,0.,0.001)
        vt_0=func(Mag,*V0)
        rt=Rd*func3(Mag,*rPE)
       
        # Outer edge, and half of it for the slope
        x2 = RHI * 2.
        x1 = RHI
        # Calculate rotation velocities at R_opt for all vt_0, rt
        vt = vt_0 * ( 1. - np.exp(-R_opt/rt) ) * ( 1. + a * R_opt/rt )

        # Best guess for Magnitude based on vrot with other params
        # Finds index of vt that most closely matches vrot and 
        # that matches the Magnitude
        ind = np.argmin(abs(vt-vrot))
        Mag = round(Mag[ind],2)
        if len(np.array([a])) != 1.: a = a[ind]
        print('1: Target Vrot:',np.round(vrot,2),'Current Vrot:',np.round(vt[ind],2),"Mag:",Mag,'Alpha',np.round(a,6),'Max Possible',np.max(vt))

        vt_0_plt = vt_0
        rt_plt   = rt

        vt = vt[ind]
        rt = rt[ind]
        vt_0 = vt_0[ind]
        
        a = np.arange(-0.04,0.4,0.0001)
        a_plt = a
   
        slope1 = ((1.-np.exp(-x2/rt))*(1.+a*x2/rt))
        slope2 = ((1.-np.exp(-x1/rt))*(1.+a*x1/rt))
        slope1_log = np.log10(slope1[(slope1 > 0) & (slope2 > 0)])
        slope2_log = np.log10(slope2[(slope1 > 0) & (slope2 > 0)])
        a = a[(slope1 > 0) & (slope2 > 0)]
        slope = (slope1_log-slope2_log) / (np.log10(x2)-np.log10(x1))
        print(len(a),len(slope))
        a = a[np.argmin(abs(slope - slope_sparc))]

   #     slope1 = np.zeros_like(a)*9E9
   #     slope2 = np.zeros_like(a)*9E9
   #     slope1_log = np.zeros_like(a)*9E9
   #     slope2_log = np.zeros_like(a)*9E9
   #     slope  = np.zeros_like(a)*9E9
   #
   #     for ind, value in enumerate(a):
   #         slope1[ind] = ((1.-np.exp(-x2/rt))*(1.+a[ind]*x2/rt))
   #         slope2[ind] = ((1.-np.exp(-x1/rt))*(1.+a[ind]*x1/rt))
   #         if slope1[ind] > 0 and slope2[ind] >0:
   #             slope1_log[ind] = np.log10(slope1[ind])
   #             slope2_log[ind] = np.log10(slope2[ind])
   #             slope[ind] = (slope1_log[ind]-slope2_log[ind]) / (np.log10(x2)-np.log10(x1))
    
   #     a = a[np.argmin(abs(slope - slope_sparc))]
    
        vt = vt_0 * ( 1. - np.exp(-R_opt/rt) ) * ( 1. + a * R_opt/rt )
        vtest = vt_0_plt * ( 1. - np.exp(-R_opt/rt_plt) ) * ( 1. + a * R_opt/rt_plt )
        #print(len(rt),len(vt_0))
        #print('HERE: adjusted vt',vt,'adjusted a',a,'best slope',slope[np.argmin(abs(slope[np.isfinite(slope)] - slope_sparc))],'real slope',slope_sparc)
        print('2: Target Vrot:',np.round(vrot,2),'Current Vrot:',np.float(np.round(vt,2)),"Mag:",Mag,'Alpha',np.round(a,6))
    if True:
            Mag_plt = np.arange(-25,0.,0.001)
            plt.plot(Mag_plt[np.argmax(vtest):],vtest[np.argmax(vtest):])
            plt.plot(Mag_plt[np.argmax(vtest):],vt_0_plt[np.argmax(vtest):])
            plt.show()
    if False:
            plt.plot(a_plt[(slope1>0) & (slope2>0)],slope)
            plt.xlabel('a: polyex outer slope')
            plt.ylabel('$\Delta$log(v)/$\Delta$log(r)')
            plt.show()
    return Mag,a,slope

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


def err(errbar):
    return errbar*(np.random.random()*2. - 1.)

def phi(MHI, Mstar, alpha, phi_0):
    #Mass Function
    return np.log(10.) *phi_0* (MHI/Mstar)**(alpha+1.) * np.exp(-MHI/Mstar)

def DHI_calc(MHI, slope,const):
    #Diameter-Mass relation
    slope = slope[0] + err(slope[1])
    const = const[0] + err(const[1])
    return 10.**(slope*np.log10(MHI)+const)

def Mstar_calc(Mgas,slope,const,split,scatr):
    #Stellar Mass calculator
    mass = np.log10(Mgas)
    if mass < split:
        slope = slope[0,0] + err(slope[0,1])
        const = const[0,0] + err(const[0,1])+err(scatr[0,0]+err(scatr[0,1]))
    else:
        slope = slope[1,0] + err(slope[1,1])    
        const = const[1,0] + err(const[1,1])+err(scatr[1,0]+err(scatr[1,1]))

    Mstar = mass * 1./slope - const/slope
    return 10.** Mstar       

def BTFR(Mbar,slope,const,scatr):
    #Baryonic Tully Fisher
    slope = slope[0] + err(slope[1])
    const = const[0] + err(const[1])+err(scatr[0]+err(scatr[1]))
    logv = np.log10(Mbar) * slope + const
    return 10.**(logv)

def TFR(slope,const,vflat):
    # Tully Fisher
    slope = slope[0] + err(slope[1])
    const = const[0] + err(const[1])
    Mag = slope * (np.log10(vflat) -2.3)+const
    return Mag

def expdisk(v,slope,const,scatr):
    #scale length for the polyex fit
    slope = slope[0] + err(slope[1])
    const = const[0] + err(const[1]) + err(scatr)
    return 10.**(const + slope * np.log10(v))

def setup_relations(mass,beams,thicc):
    ######################################################
    MHI = np.round(10.**(np.arange(6.,11.1,.1)),1)
    mass=10.**float(mass)
    i = np.array(np.argmin(abs(float(mass)-MHI)))
    MHI = MHI[i]
    # Msun

    ######################################################
    # Martin et al 2010 
    # https://arxiv.org/abs/1008.5107
    phi_0           = 0.0048 #\pm 0.3E-3
    Mstar           = 10.**9.96 #\pm 0.02 dex
    alpha           = -1.33 #\pm 0.02
    HIMF = phi(MHI,Mstar,alpha,phi_0)
    # Mpc^-3 dex^-1

    ######################################################
    # Jing Wang, Koribalski, et al 2016
    # https://arxiv.org/abs/1605.01489
    slope = np.array([0.506,0.003])
    const = np.array([-3.293,0.009])
    scatter = 0.06 #dex
    DHI = DHI_calc(MHI,slope, const)
    # kpc

    ######################################################
    # Bradford et al 2015, Fig 5
    # https://arxiv.org/abs/1505.04819
    split           = 9.2832
    Mgas            = MHI * 1.4
    slope = np.array([[1.052,0.058],[0.461,0.011]])
    const = np.array([[0.236,0.476],[5.329,0.112]])
    scatr = np.array([[0.285,0.019],[0.221,0.006]])
    Mstar = Mstar_calc(Mgas,slope,const,split,scatr) 
    Mbar = Mstar + Mgas
    # Msun

    ######################################################
    # Bradford et al 2015, Fig 6
    # https://arxiv.org/abs/1505.04819
    slope = np.array([0.277,0.004])
    const = np.array([-0.672,0.041])
    scatr = np.array([0.075,0.002])
    vflat = BTFR(Mbar,slope,const,scatr)
    # km/s

    ######################################################
    # Jing et al 2014
    # https://arxiv.org/abs/1401.8164
    # HI scale length = 0.18 RHI
    Rs = (DHI/2.)*0.18
    # kpc

    ######################################################
    # Saintonge et al 2007
    # https://arxiv.org/abs/0710.0760
    slope = np.array([0.56,0.04])
    const = np.array([-0.36,0.08])
    scatr = np.array([0.16])
    Rd = expdisk(vflat,slope,const,scatr)
    # Rd I-band

    #####################################################
    # Calculating Magnitude from vmax
    # Catinella et al 2005
    # https://arxiv.org/abs/astro-ph/0512051
    Mag,alpha,rc_slope = Magcalc(vflat,Rd,DHI/2.,Mstar)
    # I-band mag

    #####################################################
    # Velocity dispersion to range from 8km/s
    # for most massive, to 20km/s for least
    # massive?
    # Constant for now.
    Vdisp = 8.

    dist = DHI / (4.*beams * (np.pi/162000.))

    dist = dist * u.kpc
    dist = dist.to_value(u.kpc)

    #print('distance [kpc]', round(dist,2))
    delta = ((thicc*u.arcsec).to_value(u.rad)*(dist))* u.kpc
    #print('ringsize',np.round(delta/(1*u.kpc),4))
    #print('rings',np.round(DHI*u.kpc /delta,2))
    radi = np.arange(0.,DHI,delta/u.kpc)

    vrot,rPE = make_vrot(radi,Mag,Rd,vflat,DHI/2.,Mstar)
    sbr  = (1./dist)*(1/0.236)*make_sbr(radi,Rs,DHI,vflat,mass)
    #print('Sigma0',round(np.max(sbr),6))
    z    = make_z(radi,vrot,Vdisp)

    mid = int(len(radi)/2)-1
    end = len(radi)-1
    slope = (np.log10(vrot[end])-np.log10(vrot[mid]))/(np.log10(radi[end])-np.log10(radi[mid]))
   
    make_plots = False
    if (make_plots):
        plt.figure(1)
        plt.title(str(np.log10(mass))+' dex M$_{\odot}$')
        plt.semilogy(radi,sbr)
        plt.xlabel('R [kpc]')
        plt.ylabel('SBR [Jy km s$^{-1}$ arcsec$^{-1}$]')
        plt.axvline(DHI/2.)
        plt.savefig('SBR.png')
        plt.close(1)
        plt.figure(2)
        plt.title(str(np.log10(mass))+' dex M$_{\odot} HI$\n'+str(round(np.log10(Mbar),3))+' dex M$_{\odot} Mstar$')
        plt.plot(radi,vrot)
        plt.xlabel('R [kpc]')
        plt.ylabel('Vc [km/s]')
        plt.axvline(DHI/2.)
        plt.savefig('VROT.png')
        plt.close(2)
    rothead(MHI,Mag,Vdisp,Mbar,Mstar,DHI,vflat,Rs,dist)
    return radi, sbr, vrot, Vdisp, z, MHI, DHI, Mag, dist, alpha,vflat,Mstar,slope,Rd*3.31,rPE
