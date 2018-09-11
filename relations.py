#!/usr/bin/env python3
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import astropy.units as u
from scipy.optimize import curve_fit
from scipy import integrate
from Input_output import rothead

def func(x,a,b,c,d):
    # Fit for V0
    return a*np.exp(-x*b-c) +d*x

def func2(x,b,c):
    # Linear fit for A
    return b*x + c

def func3(x,a,b):
    # Fit for rPE
    return a*np.log10(-x) - b

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

    return vt*(1.-np.exp(-radi/rt))*(1.+a*radi/rt),rt

def Magcalc(vrot,Ropt,RHI,mstar):
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
    rt=Ropt*func3(Mag,*rPE)

    # Set slope from NIHAO 17 
    slope_sparc = 0.123 - 0.137*(np.log10(mstar)-9.471) + err(0.19)

    # Find Vrot, then Alpha, then check again to make sure Vrot is consistent
    for i in range(0,6):

        # Make parameters for all Magnitudes
        Mag = np.arange(-25,0.,0.001)
        vt_0=func(Mag,*V0)
        rt=Ropt*func3(Mag,*rPE)
       
        # Outer edge, and half of it for the slope
        x2 = RHI * 2.
        x1 = RHI
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
       
        # Consider a range of values of alpha
        a = np.arange(-0.04,0.4,0.0001)
        slope1 = ((1.-np.exp(-x2/rt))*(1.+a*x2/rt))
        slope2 = ((1.-np.exp(-x1/rt))*(1.+a*x1/rt))
        # Only want values where logv is defined (v>0)
        slope1_log = np.log10(slope1[(slope1 > 0) & (slope2 > 0)])
        slope2_log = np.log10(slope2[(slope1 > 0) & (slope2 > 0)])
        a = a[(slope1 > 0) & (slope2 > 0)]

        # Calculate delta logv / delta log r
        # Find value of a that gives value closest to NIHAO
        slope = (slope1_log-slope2_log) / (np.log10(x2)-np.log10(x1))
        a = a[np.argmin(abs(slope - slope_sparc))]

    vt_1  = vt_0*(1.-np.exp(-x2/rt))*(1.+a*x2/rt)
    vt_2  = vt_0*(1.-np.exp(-x1/rt))*(1.+a*x1/rt)
    slope = (np.log10(vt_1)-np.log10(vt_2))/(np.log10(x2)-np.log10(x1))

    return Mag,a,slope

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
    delta = np.arange(-0.085,0.0851,0.001)
    Mass_guess = np.zeros_like(delta)
    for i, dx in enumerate(delta):
        sbr = sbr_calc(radi,RHI,x,dx,vt,Rs)
        Mass_guess[i] = (integrate.simps(sbr*2*np.pi*radi,radi)*1000.**2.)
    Mj = np.argmin(abs(Mass_guess- mass))
    dx = delta[Mj]

    # When closest one is found, calculate it and return it
    sbr = sbr_calc(radi,RHI,x,dx,vt,Rs)
    Mass_guess = (integrate.simps(sbr*2.*np.pi*radi,radi)*1000.**2.)
    print('Guess','{:.3f}'.format(np.log10(Mass_guess)),'Mass','{:.3f}'.format(np.log10(mass)))
    print('dx=',dx,'and sigma=',0.36+dx)
    if round(dx,3) <= -0.085 or round(dx,3) >= 0.085:
        while True:
            print("FAILURE",dx,0.36+dx)
            stop


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

def DHI_calc(MHI, slope,const,scatr):
    #Diameter-Mass relation
    slope = slope[0] + err(slope[1])
    const = const[0] + err(const[1]) + err(scatr)
    DHI = 10.**(slope*np.log10(MHI)+const)
    return DHI

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

def setup_relations(mass,beams,ring_thickness,make_plots):
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
    slope = np.array([0.506,0.003])
    const = np.array([-3.293,0.009])
    scatr = 0.06 #dex
    DHI = DHI_calc(MHI,slope, const,scatr)
    # kpc

    ######################################################
    # Bradford et al 2015, Fig 5
    # HI Mass - Stellar Mass Relationship
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
    # Baryonic Tully-Fisher relationship
    # https://arxiv.org/abs/1505.04819
    slope = np.array([0.277,0.004])
    const = np.array([-0.672,0.041])
    scatr = np.array([0.075,0.002])
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
    slope = np.array([0.56,0.04])
    const = np.array([-0.36,0.08])
    scatr = np.array([0.16])
    Ropt = expdisk(vflat,slope,const,scatr)
    # R_opt I-band

    #####################################################
    # Calculating Magnitude from vmax
    # Catinella et al 2005
    # https://arxiv.org/abs/astro-ph/0512051
    Mag,alpha,slope = Magcalc(vflat,Ropt,DHI/2.,Mstar)
    # I-band mag
    #####################################################
    # Compute radial sampling cadence
    # 30 arcseconds to radians, small angle apprx
    #dist  = DHI * 21600. / (beams*np.pi)
    dist  = DHI * 32400. / (beams*np.pi)
    dist = 7.83*1000

    delta = ((ring_thickness*u.arcsec).to_value(u.rad)*(dist))
    #####################################################
    # Compute radi, rotation curve, surface brightness profile
    radi     = np.arange(0.,DHI+1,delta)
    vrot,rPE = make_vrot(radi,Mag,Ropt,alpha)
    #####################################################
    # Convert SBR to Jy
    print('Dist','{:.3f}'.format(dist),'kpc')
    sbr      = make_sbr(radi,Rs,DHI,vflat,mass)
    print('Integrated Mass         [1D]:','{:.3f}'.format(np.log10(integrate.simps(2*np.pi*sbr*radi,radi)*1000.**2.)))
    #sbr      = (3600./(0.236*dist**2.))*sbr
    sbr = sbr * 1.24756e+20
    conv_column_arsec=605.7383*1.823E18*(2.*np.pi/(np.log(256.)))
    sbr = sbr/(conv_column_arsec*1000.)
    #####################################################
    # Set the radii, rotation curve, surface brightness prof
    radi = radi     / (dist) * 3600. * (180./np.pi)
    END  = DHI      / (dist) * 3600. * (180./np.pi)
    print('Integrated Flux to Mass [1D]:','{:.3f}'.format(np.log10((0.236*dist**2.)*integrate.simps(2*np.pi*sbr*radi,radi))))
    cflux = np.sum(sbr / 1.0E5)
    #####################################################
    # Velocity dispersion 8km/s
    # Constant for now.
    # Use it to calculate disk thickness based on
    # Puche et al 1992
    # http://adsabs.harvard.edu/abs/1992AJ....103.1841P
    Vdisp = 13.
    z    = make_z(radi,vrot,Vdisp)
    ###############################################
   
    if (make_plots):
        label_size=21.5
        lw=1.5
        mpl.rcParams['xtick.labelsize'] = label_size
        mpl.rcParams['ytick.labelsize'] = label_size
        mpl.rcParams['lines.linewidth'] = 3
        mpl.rcParams['xtick.major.size'] = 10
        mpl.rcParams['xtick.major.width'] = lw
        mpl.rcParams['xtick.minor.size'] = 5
        mpl.rcParams['xtick.minor.width'] = lw
        mpl.rcParams['ytick.major.size'] = 10
        mpl.rcParams['ytick.major.width'] = lw
        mpl.rcParams['ytick.minor.size'] = 5
        mpl.rcParams['ytick.minor.width'] = lw
        mpl.rcParams['axes.linewidth'] = lw
        mpl.rcParams['font.monospace'] = 'Courier'
        mpl.rcParams['legend.scatterpoints'] = '3'
        mpl.rcParams['mathtext.default'] = 'regular'
        mpl.rcParams['xtick.direction'] = 'in'
        mpl.rcParams['ytick.direction'] = 'in'

        fig, ax = plt.subplots(figsize=(20, 10))
        plt.title('log$_{10}$ MHI [M$_{\odot}$] ='+str(np.log10(mass))+';\tlog$_{10}$ MBar [M$_{\odot}$] = '+str(round(np.log10(Mbar),3)),fontsize=label_size)
        plt.semilogy(radi,sbr)
        plt.xlabel('R [kpc]',fontsize=label_size)
        plt.ylabel('SBR [Jy km s$^{-1}$ arcsec$^{-2}$]',fontsize=label_size)
        plt.axvline(DHI/2.)
        minorLocator = mpl.ticker.AutoMinorLocator()
        ax.xaxis.set_minor_locator(minorLocator)
        plt.savefig('SBR.png',bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(figsize=(20, 10))
        #plt.title('log$_{10}$ MHI [M$_{\odot}$] ='+str(np.log10(mass))+' \n'+'log$_{10}$ MBar [M$_{\odot}$] = '+str(round(np.log10(Mbar),3)),fontsize=label_size)
        plt.title('log$_{10}$ MHI [M$_{\odot}$] ='+str(np.log10(mass))+';\tlog$_{10}$ MBar [M$_{\odot}$] = '+str(round(np.log10(Mbar),3))+';\t$\Delta$log(V)/$\Delta$log(R) = '+str(round(slope,5)),fontsize=label_size)
        plt.plot(radi,vrot)
        plt.xlabel('R [kpc]',fontsize=label_size)
        plt.ylabel('Vc [km/s]',fontsize=label_size)
        plt.axvline(DHI/2.)
        minorLocator = mpl.ticker.AutoMinorLocator()
        ax.xaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_minor_locator(minorLocator)
        plt.savefig('VROT.png',bbox_inches='tight')
        plt.close()
    rothead(MHI,Mag,Vdisp,Mbar,Mstar,DHI,vflat,Rs,dist)
    return radi, sbr, vrot, Vdisp, z, MHI, DHI, Mag, dist, alpha,vflat,Mstar,slope,Ropt,rPE,cflux,END
