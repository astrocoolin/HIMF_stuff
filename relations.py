#!/usr/bin/env python3
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.units as u
from scipy.optimize import curve_fit
from scipy import integrate
from scipy import ndimage
from Input_output import rothead

def func(x,a,b,c,d):
    # Fit for V0
    return a*np.exp(-x*b-c) +d*x

def func2(x,m,b):
    # Linear fit for A
    return m*x + b

def func3(x,m,b):
    # Fit for rPE
    #return b*np.log10(-x) - m
    return m*x + b

def make_vrot(radi,Mag,Ropt,alpha):
    # Returns a polyex rotation curve
    # Input a magnitude, Ropt, radii, and alpha
    # Catinella et al 2007
    # https://arxiv.org/abs/astro-ph/0512051
    V0 = np.array([275.,255.,225.,200.,170.,148.,141.,122.,103.,85.])
    dV0 = np.array([6.,2.,1.,1.,1.,2.,2.,2.,2.,5.])
    rPE = np.array([0.126,0.132,0.149,0.164,0.178,0.201,0.244,0.261,0.260,0.301])
    drPE = np.array([0.007,0.003,0.003,0.002,0.003,0.004,0.005,0.008,0.008,0.002])
    m=np.array([-23.76,-23.37,-22.98,-22.60,-22.19,-21.80,-21.41,-21.02,-20.48,-19.38])

    V0, foo = curve_fit(func, m, V0,sigma=dV0)
    rPE,foo = curve_fit(func3, m, rPE,sigma=drPE)
    vt=func(Mag,*V0)
    rt=Ropt*func3(Mag,*rPE) 
    a = alpha

    return vt*(1.-np.exp(-radi/rt))*(1.+a*radi/rt)

def Magcalc(vrot,Ropt,RHI,mstar,multiplier):
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
    perr = np.sqrt(np.diag(foo))

    # Make parameters for all Magnitudes
    Mag = np.arange(-24.,0.,0.001)
    Mag_ = np.arange(-24.,-19.,0.001)
    a=func2(Mag,*A)
    vt_0=func(Mag,*V0)
    rt=Ropt*func3(Mag,*rPE)

   # import matplotlib.pyplot as plt
   # plt.plot(Mag,func2(Mag_,*A))
   # plt.scatter(m,A_)
   # plt.savefig('A.png')
   # plt.close()
   # #plt.show()
   # plt.plot(Mag,func3(Mag_,*rPE))
   # plt.scatter(m,rPE_)
   # plt.savefig('rPE.png')
   # plt.close()
   # #plt.show()
   # plt.plot(Mag,func(Mag_,*V0))
   # plt.scatter(m,V0_)
   # plt.savefig('V0.png')
   # plt.close()
    #plt.show()

    # Set slope from NIHAO 17 
    slope_sparc = 0.123 - 0.137*(np.log10(mstar)-9.471) + err(0.19)*multiplier
    #print(slope_sparc,np.log10(mstar),mstar)

    # Find Vrot, then Alpha, then check again to make sure Vrot is consistent
    for i in range(0,1):

        # Make parameters for all Magnitudes
        Mag = np.arange(-24.,0.,0.001)
        vt_0=func(Mag,*V0)
        print(Ropt)
        rt=Ropt*func3(Mag,*rPE)
        a=func2(Mag,*A)
       
        # Outer edge, and half of it for the slope
        x2 = RHI * 1.
        x1 = RHI * 0.5
        # Calculate rotation velocities at Ropt for all vt_0, rt
        vt = vt_0 * ( 1. - np.exp(-Ropt/rt) ) * ( 1. + a * Ropt/rt )

        # Best guess for Magnitude based on vrot with other params
        # Finds index of vt that most closely matches vrot and 
        # that matches the Magnitude
        ind = np.argmin(abs(vt-vrot))
        Mag  = Mag[ind]
        vt   = vt[ind]
        rt   = rt[ind]
        vt_0 = vt_0[ind]
        a = a[ind]
       

    vt_2  = vt_0*(1.-np.exp(-x2/rt))*(1.+a*x2/rt)
    vt_1  = vt_0*(1.-np.exp(-x1/rt))*(1.+a*x1/rt)
    slope = (np.log10(vt_2)-np.log10(vt_1))/(np.log10(x2)-np.log10(x1))

    return Mag,a,slope,vt_0,rt

def sbr_calc(radi,RHI,x,dx,vt,Rs):
    # Return the surface brightness profile using the formula from PK
    sbr = np.zeros_like(radi)
    # Go through every point in the rotation curve
    # (For speed improvement maybe just make this array operations)
    for j, r in enumerate(radi):
        sig1 = np.exp(-((r-0.4*RHI)/(np.sqrt(2)*(x+dx)*RHI))**2.)
        sig2 = (np.sqrt(vt/120.)-1.)*np.exp(-r/Rs)
        sbr[j] = sig1-sig2
        if sbr[j] < 0:
            sbr[j] = 0
    # Find the closest point to the 1 Msun/pc radius
    R_HI= np.argmin(abs(radi - RHI))

    # Return normalized surface brightness profile
    sbr = sbr/sbr[R_HI]
    return sbr

def make_sbr(radi,Rs,DHI,vt,mass):
    # Make the surface brightness profile
    RHI=DHI/2.
    x=0.36

    # consider a range of x+dx to get closest match to HI mass
    delta = np.arange(-0.15,0.151,0.005)
    Mass_guess = np.zeros_like(delta)
    for i, dx in enumerate(delta):
        sbr = sbr_calc(radi,RHI,x,dx,vt,Rs)
        Mass_guess[i] = (integrate.trapz(sbr*2*np.pi*radi,radi)*1000.**2.)
    Mj = np.argmin(abs(Mass_guess- mass))
    dx = delta[Mj]

    # When closest one is found, calculate it and return it
    sbr = sbr_calc(radi,RHI,x,dx,vt,Rs)
    Mass_guess = (integrate.simps(sbr*2.*np.pi*radi,radi)*1000.**2.)
    #print(np.log10(Mass_guess),'Mass_guess')
    if round(dx,3) <= -0.15 or round(dx,3) >= 0.151:
        while True:
            print("FAILURE",dx,0.36+dx)
            stop

    return sbr,dx

def make_z(radi,vrot,sigma):
    z = np.zeros_like(radi)
    for i, r in enumerate(radi):
        if i != 0:
            z[i] = sigma / ( np.sqrt(2./3.) * vrot[i]/r)
        if i == 0:
            z[i] = sigma / ( np.sqrt(2./3.) * vrot[1]/radi[1])
    return  z

def err(errbar):
    temp_err =np.random.normal(loc=0.,scale=errbar)
    while temp_err > 2.5 * errbar:
        temp_err =np.random.normal(loc=0.,scale=errbar)
    #return(temp_err)
    return(-errbar)

def phi(MHI, Mstar, alpha, phi_0):
    #Mass Function
    return np.log(10.) *phi_0* (MHI/Mstar)**(alpha+1.) * np.exp(-MHI/Mstar)

def DHI_calc(MHI, slope,const,scatr):
    #Diameter-Mass relation
    slope = slope[0] + err(slope[1])
    const = const[0] + err(const[1]) + err(scatr)
    DHI = 10.**(slope*np.log10(MHI)+const)
    print('DHI',DHI)
    return DHI

def Mstar_calc(Mgas,slope,const,split,scatr):
    #Stellar Mass calculator
    mass = np.log10(Mgas)
    if mass < split:
        slope = slope[0,0] #+ err(slope[0,1])
        const = const[0,0] - err(const[0,1])-err(scatr[0,0]+err(scatr[0,1]))
    else:
        slope = slope[1,0] #+ err(slope[1,1])    
        const = const[1,0] - err(const[1,1])-err(scatr[1,0]+err(scatr[1,1]))

    Mstar = mass * 1./slope - const/slope
    print('Mstar',Mstar)
    return 10.** Mstar       

def BTFR(Mbar,slope,const,scatr):
    #Baryonic Tully Fisher
    slope = slope[0] + err(slope[1])
    const = const[0] + err(const[1])+err(scatr[0]+err(scatr[1]))
    logv = np.log10(Mbar) * slope + const
    print("VELOCITY BRADFORD",10.**logv)
    return 10.**(logv)

def expdisk(v,slope,const,scatr):
    #scale length for the polyex fit
    slope = slope[0] + err(slope[1])
    const = const[0] + err(const[1]) + err(scatr)
    return float(10.**(const + slope * np.log10(v)))

def setup_relations(mass,beams,beam,ring_thickness,scatter):
    if scatter == "False" or not scatter:
        Mstar_mult = 0.
        multiplier = 0.
        m_array1 = np.array([1.,0.])
        m_array2_slope = np.array([[1.,0.],[1.,0.]])
        m_array2_const = np.array([[1.,0.],[1.,0.]])
        #print('Not scattering')
    elif scatter == "Mstar":
        Mstar_mult = 1.
        multiplier = 0.
        m_array1 = np.array([1.,0.])
        m_array2_slope = np.array([[1.,0.],[1.,0.]])
        m_array2_const = np.array([[1.,1.],[1.,1.]])
    else:
        Mstar_mult = 1.
        multiplier = 1.
        m_array1 = np.array([1.,1.])
        m_array2_slope = np.array([[1.,1.],[1.,1.]])
        m_array2_const = np.array([[1.,1.],[1.,1.]])
    ######################################################
    MHI = np.round(10.**(np.arange(6.,11.1,.1)),1)
    mass=10.**float(mass)
    i = np.array(np.argmin(abs(float(mass)-MHI)))
    MHI = MHI[i]
    # Msun

    ######################################################
    # Martin et al 2010 
    # HI Mass function
    # https://arxiv.org/abs/1008.5107
    phi_0           = 0.0048 #\pm 0.3E-3
    Mstar           = 10.**9.96 #\pm 0.02 dex
    alpha           = -1.33 #\pm 0.02
    HIMF = phi(MHI,Mstar,alpha,phi_0)
    # Mpc^-3 dex^-1

    ######################################################
    # Jing Wang, Koribalski, et al 2016
    # HI Mass - Diameter relationship
    # https://arxiv.org/abs/1605.01489
    slope = np.array([0.506,0.003])*m_array1
    const = np.array([-3.293,0.009])*m_array1
    scatr = 0.06*multiplier #dex
    DHI = DHI_calc(MHI,slope, const,scatr)
    # kpc

    ######################################################
    # Bradford et al 2015, Fig 5
    # HI Mass - Stellar Mass Relationship
    # https://arxiv.org/abs/1505.04819
    split           = 9.2832
    Mgas            = MHI * 1.4
    slope = np.array([[1.052,0.058],[0.461,0.011]])*m_array2_slope
    const = np.array([[0.236,0.476],[5.329,0.112]])*m_array2_const
    scatr = np.array([[0.285,0.019],[0.221,0.006]])*Mstar_mult
    Mstar = Mstar_calc(Mgas,slope,const,split,scatr) 
    Mbar = Mstar + Mgas
    #print(np.log10(Mstar))
    # Msun

    ######################################################
    # Bradford et al 2015, Fig 6
    # Baryonic Tully-Fisher relationship
    # https://arxiv.org/abs/1505.04819
    slope = np.array([0.277,0.004])*m_array1
    const = np.array([-0.672,0.041])*m_array1
    scatr = np.array([0.075,0.002])*multiplier
    vflat = BTFR(Mbar,slope,const,scatr)
    # km/s

    ######################################################
    # Jing et al 2014
    # HI scale length = 0.18 RHI
    # https://arxiv.org/abs/1401.8164
    Rs = (DHI/2.)*0.18
    # kpc

    ######################################################
    # Saintonge et al 2007
    # Optical Radius (r83) - Vflat Relationship
    # https://arxiv.org/abs/0710.0760
    slope = np.array([0.56,0.04])*m_array1
    const = np.array([-0.36,0.08])*m_array1
    scatr = np.array([0.16])*multiplier
    Ropt = expdisk(vflat,slope,const,scatr)
    h = 0.70
    Ropt = Ropt*h
    # R_opt I-band

    #####################################################
    # Calculating Magnitude from vmax
    # Catinella et al 2005
    # https://arxiv.org/abs/astro-ph/0512051
    Mag,alpha,slope,v0,rPE = Magcalc(vflat,Ropt,DHI/2.,Mstar,multiplier)
    # I-band mag
    #####################################################
    # Compute radial sampling cadence
    # 30 arcseconds to radians, small angle apprx

    dist  = DHI * (206265./(beam*beams))
    delta = ((ring_thickness*u.arcsec).to_value(u.rad)*(dist))
    #####################################################
    # Compute radi, rotation curve, surface brightness profile
    radi     = np.arange(0.,DHI+delta,delta)
    vrot     = make_vrot(radi,Mag,Ropt,alpha)
    #####################################################
    # Convert SBR to Jy
    sbr,dx   = make_sbr(radi,Rs,DHI,vflat,mass)
    conv=6.0574E5*1.823E18*(2.*np.pi/np.log(256.))
    sbr = sbr*1.24756e+20/(conv)

    #####################################################
    # Velocity dispersion 8km/s
    # Constant for now.
    # Use it to calculate disk thickness based on
    # Puche et al 1992
    # http://adsabs.harvard.edu/abs/1992AJ....103.1841P
    Vdisp = 12.
    z    = make_z(radi,vrot,Vdisp)
    #####################################################
    # Set the radii, rotation curve, surface brightness prof
    radi = radi     / (dist) * 3600. * (180./np.pi)
    END  = DHI      / (dist) * 3600. * (180./np.pi)
    rPE_kpc = rPE
    rPE  = rPE      / (dist) * 3600. * (180./np.pi)
    #cflux = np.sum(sbr / 1.0E5)
    cflux = 1.0E-6
    ###############################################
    sbr2 = sbr/1.24756e+20*(conv)
    
    #return Vdisp, MHI, DHI, Mag, dist, alpha,vflat,Mstar,slope,Ropt,rPE,cflux,END,v0,dx,Rs
    #print('SLOPE,MSTAR,MHI',slope,Mstar,MHI)
    return np.log10(MHI),DHI,np.log10(Mstar),Ropt,vflat,Vdisp,alpha,rPE_kpc,v0,dx,Rs,Mag,slope
