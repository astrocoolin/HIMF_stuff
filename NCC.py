#!/usr/bin/env python3
# cross validation
# also check residuals for patterns, fit a curve to them
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
import os
from Input_output import *
from relations import *
from science import second_beam

class Galaxy:
    def __init__(self):
	# Inclination
        self.inc = 60
	# HI mass, HI diameter
        self.MHI = 9.5
        self.DHI = 33.9396044996648
	# Stellar Mass, Baryonic mass
        self.Mstar = 9.31760909803635
        self.Mbar = np.log10(10**(self.Mstar) + 1.4*10**(self.MHI))
        # Optical Radius
        self.Ropt = 10.88891262333259
	# 'Vflat' from BTFR
        self.vflat = 140.13278394294278
	# sigma from vdisp
        self.sigma = 2.
	# Outer radius for cutoff 
        self.END = self.DHI
       
        ####
	# for TiRiFiC, this is ring width
        self.rwidth = 2.5
	# Number of beams across, beamsize in arcsec
        self.beams = 4.
        self.beamsize = 30.
        ####
	# Polyex params
        self.alpha = 0.017
        self.rt = 8.431914405739636
        self.v_0 = 114.75505945574942
        self.dx = -0.025
        self.Rs = 3.0487

	# Galaxy Magnitude
        self.Mag = -21.5
	# outer slope shape (Dutton + 2019)
        self.slope = 0.0
	# TiRiFiC file name
        self.defname ="cube_input.def"
        self.inset   ='empty.fits'
	# output name for noiseless cube
        self.outset  ='Cube_base.fits'
	# output name for completed cube
        self.outname ='Cube_ba_'+str(self.beams)+".mass_"+str(round(self.MHI,5))+".inc_"+\
                str(self.inc)+'.fits'
	# folder name for completed cube
        self.fname ="ba_"+str(self.beams)+".mass_"+str(round(self.MHI,5))+".inc_"+\
                str(self.inc)

    def save(self):
	# stuff that gets output when you call save
        print(
        "\nself.MHI   =",self.MHI,
        "\nself.DHI   =",self.DHI,
        "\nself.Mstar =",self.Mstar,
        "\nself.Ropt  =",self.Ropt ,
        "\nself.Rs    =",self.Rs ,
        "\nself.vflat =",self.vflat,
        "\nself.rwidth=",self.rwidth,
        "\nself.alpha =",self.alpha,
        "\nself.rt    =",self.rt  ,
        "\nself.v_0   =",self.v_0 ,
        "\nself.dx    =",self.dx  ,
        "\nself.Rs    =",self.Rs ,
        "\nself.Mag   =",self.Mag ,
        "\nself.slope =",self.slope,
        "\nself.Mbar  = np.log10(10**(self.Mstar) + 1.4*10**(self.MHI))",
        "\nself.END   =  self.DHI"
	)


    def reroll(self,Mass,beams,inclination):
        scatter = False
        self.inc = inclination

        # use this to set params for galaxy  of a given mass and resolution
        self.beams = beams
        self.beamsize = 30
        self.rwidth = 2.5

        self.MHI,self.DHI,self.Mstar,self.Ropt,self.vflat,self.sigma,self.alpha,self.rt,self.v_0,\
                self.dx,self.Rs,self.Mag,self.slope=setup_relations(Mass,self.beams,self.beamsize,self.rwidth,scatter)
        
        self.Mbar = np.log10(10**(self.Mstar) + 1.4*10**(self.MHI))
        self.END = self.DHI
       
        self.calc_dist(self.beams)

        self.outname ='Cube_ba_'+str(self.beams)+".mass_"+str(round(self.MHI,5))+".inc_"+\
                str(self.inc)+'.fits'
	# folder name for completed cube
        self.fname ="ba_"+str(self.beams)+".mass_"+str(round(self.MHI,5))+".inc_"+\
                str(self.inc)


    def calc_dist(self,beams):
	# distance calculations 
        self.beams=beams
	# sets what distance has to be based on mass and resolution
        self.dist = self.DHI * (206265./(self.beams*self.beamsize))
	# tells you the physical distance between rings
        self.delta = (self.rwidth/206265.)*self.dist
	# calculates the rings based on physical distance
        self.radi = np.arange(0,self.END+self.delta,self.delta)
	
	# create polyex curve, sbr curve, z curve 
        self.polyex=self.Polyex(self.alpha,self.rt ,self.v_0)
        self.sbr = self.SBR(self.dx,self.Rs,self.DHI,self.vflat)
        self.z = self.Z(self.sigma)
	
	# DHI in arcseconds 
        self.DHI_arcsec = self.DHI*206265./self.dist
	# save the profiles to the object
        self.profiles=self.Profiles(self.polyex,self.sbr,self.radi,self.dist,self.z)

    class Polyex():
	# polyex parameters
        def __init__(self,alpha,rt,v_0):
            self.alpha = alpha
            self.v_0 = v_0
            self.rt = rt 
        def v_curve(self,r):
            return self.v_0*(1.-np.exp(-r/self.rt))*(1.+self.alpha*r/self.rt)
    class Z():
	# z profile parametrs
        def __init__(self,sigma):
            self.sigma = sigma
        def disp(self,radi,v):
            z = np.zeros_like(radi)
            for i,r in enumerate(radi):
                if i !=0:
                    z[i] = self.sigma / (np.sqrt(2./3.) *v[i]/r)
                elif i ==0:
                    z[i] = self.sigma / (np.sqrt(2./3.) *v[1]/radi[1])
            return z

    class SBR():
	# calculate surface brightness profile
	# see thesis for more
        def __init__(self,dx,Rs,DHI,vflat):
            self.dx = .36+ dx
            self.Rs = Rs
            self.RHI = DHI/2.
            self.vflat = vflat
        def sbr_curve(self,radi):
            sbr = np.zeros_like(radi)
            for j, r in enumerate(radi):
                sig1 = np.exp(-((r-0.4*self.RHI)/(np.sqrt(2)*(self.dx)*self.RHI))**2.)
                sig2 = (np.sqrt(self.vflat/120.)-1.)*np.exp(-r/self.Rs)
                sbr[j] = sig1-sig2
                if sbr[j] < 0:
                    sbr[j] = 0
            #sbr[1:2] = sbr[0]
            R_HI= np.argmin(abs(radi - self.RHI))
            sbr = sbr/sbr[R_HI]
            return sbr*1.24756e+20/(6.0574E5*1.823E18*(2.*np.pi/np.log(256.)))

    class Profiles():
        def __init__(self,polyex,sbr,radi,dist,z):
            self.radi_kpc = radi
            self.radi = radi/ (dist) * 3600. * (180./np.pi)
            self.sbr = sbr.sbr_curve(radi)
            self.vrot = polyex.v_curve(radi)
            self.z = z.disp(radi,self.vrot)

    def make_fits(self):
        #self.make_plots(False)
        
        #Make_Output
        filecheck = Path(self.defname)
        if filecheck.is_file(): os.system("rm "+self.defname)
        
        rothead(self.MHI,self.Mag,self.sigma,self.Mbar,self.Mstar,self.DHI,self.vflat,self.Rs,self.dist,self.slope,self.alpha,self.v_0,self.rt,self.Ropt,self.dx)
        rotfile(self.profiles.radi,self.profiles.vrot,self.profiles.sbr,self.profiles.z,len(self.profiles.radi))
        
        deffile(self.outset,self.inset,self.defname,self.profiles.radi,self.profiles.vrot,self.profiles.sbr,self.inc,\
            len(self.radi),8.0,self.profiles.z,5.0E-8)
        
        filecheck = Path(self.outname)
        if filecheck.is_file(): os.system("rm "+self.outname)
        filecheck = Path('empty.fits')
        if filecheck.is_file(): os.system("rm empty.fits")
        #filecheck = Path('Logfile.log')
        #if filecheck.is_file(): os.system("rm Logfile.log")
        
        emptyfits(self.inset)
        os.system("tirific deffile="+self.defname)
        print("Cube finished")
       
        filecheck = Path(self.fname)
        if filecheck.is_dir():
            os.system("rm -r "+self.fname)
            print("Refreshed folder")
        os.system("mkdir "+self.fname)
        os.system("cp "+self.defname+" RC.dat "+self.fname)
        ######################################################################
        second_beam(self.outset,self.outname,self.END/ (self.dist) * 3600. * (180./np.pi),self.beams,self.inc,self.MHI,self.dist,1.0E-6,self.beamsize,self.DHI)
        os.system("mv "+self.outname+" "+self.outset+" "+self.fname)
        os.system("rm "+self.outname)
        os.system("rm empty.fits")
        #os.system("rm  VROT.png SBR.png SBR_log.png RC.dat "+self.outset+" Logfile.log "+self.defname)
    def make_plots(self,show):
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

        fig1, ax1 = plt.subplots(figsize=(20, 10))
        plt.title('log$_{10}$ MHI [M$_{\odot}$] ='+str(self.MHI)+'\nlog$_{10}$ Mstar [M$_{\odot}$] ='+str(self.Mstar),fontsize=label_size)
        plt.plot(self.profiles.radi,self.profiles.sbr)
        plt.xlabel('R [arcsec]',fontsize=label_size)
        plt.ylabel('SBR [Jy km s$^{-1}$ arcsec$^{-2}$]',fontsize=label_size)
        plt.axvline((self.DHI/2.)/ (self.dist) * 3600. * (180./np.pi))
        minorLocator = mpl.ticker.AutoMinorLocator()
        ax1.xaxis.set_minor_locator(minorLocator)
        if show:
            plt.savefig('SBR.png',bbox_inches='tight')
        plt.close()

        fig2, ax2 = plt.subplots(figsize=(20, 10))
        plt.title('log$_{10}$ MHI [M$_{\odot}$] ='+str(self.MHI)+'\nlog$_{10}$ Mstar [M$_{\odot}$] ='+str(self.Mstar),fontsize=label_size)
        plt.semilogy(self.profiles.radi,self.profiles.sbr)
        plt.xlabel('R [arcsec]',fontsize=label_size)
        plt.ylabel('SBR [Jy km s$^{-1}$ arcsec$^{-2}$]',fontsize=label_size)
        plt.axvline((self.DHI/2.)/ (self.dist) * 3600. * (180./np.pi))
        minorLocator = mpl.ticker.AutoMinorLocator()
        ax2.xaxis.set_minor_locator(minorLocator)
        if show:
            plt.savefig('SBR_log.png',bbox_inches='tight')
        plt.close()

        fig3, ax3 = plt.subplots(figsize=(20, 10))
        plt.title('log$_{10}$ MHI [M$_{\odot}$] ='+str(self.MHI)+'\nlog$_{10}$ Mstar [M$_{\odot}$] ='+str(self.Mstar),fontsize=label_size)
        plt.plot(self.profiles.radi,self.profiles.vrot)
        plt.xlabel('R [arcsec]',fontsize=label_size)
        plt.ylabel('Vc [km/s]',fontsize=label_size)
        plt.axvline((self.DHI/2.)/ (self.dist) * 3600. * (180./np.pi))
        minorLocator = mpl.ticker.AutoMinorLocator()
        ax3.xaxis.set_minor_locator(minorLocator)
        if show:
            plt.savefig('VROT.png',bbox_inches='tight')
        plt.close()
        filecheck = Path(self.fname)
        if not filecheck.is_dir():
        	os.system("mkdir "+self.fname)
        os.system("mv VROT.png SBR.png SBR_log.png "+self.fname)
