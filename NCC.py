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
    def __init__(self):#self,MHI,DHI,Mstar,Ropt,vflat,dx,Rs,alpha,disp,rt,v_0,delta,beams,beamsize,snr,inc):
       
        self.snr = 8.0
        self.inc = 60
        
        self.MHI = 9.5
        self.DHI = 33.9396044996648
        
        self.Mstar = 9.31760909803635
        self.Mbar = np.log10(10**(self.Mstar) + 1.4*10**(self.MHI))
        
        self.Ropt = 10.88891262333259
        self.vflat = 140.13278394294278
        self.sigma = 2.
        
        self.END = self.DHI
       
        ####
        self.rwidth = 2.5
        self.beams = 4.
        self.beamsize = 30.
        ####

        self.alpha = 0.017
        self.rt = 8.431914405739636
        self.v_0 = 114.75505945574942
        self.dx = -0.025
        self.Rs = 3.0487
        self.Mag = -21.5
        self.slope = 0.0
        self.defname ="cube_input.def"
        self.inset   ='empty.fits'
        self.outset  ='Cube_base.fits'
        self.outname ='Cube_ba_'+str(self.beams)+".mass_"+str(round(self.MHI,5))+".inc_"+\
                str(self.inc)+".SN_"+str(self.snr)+'.fits'
        self.fname ="ba_"+str(self.beams)+".mass_"+str(round(self.MHI,5))+".inc_"+\
                str(self.inc)+".SN_"+str(self.snr)

    def example(self,vel):
        if vel == 12:
            self.MHI=7.0
            self.DHI=1.6190216491867229
            self.Mstar=6.395697326814118
            self.Mbar = np.log10(10**(self.Mstar) + 1.4*10**(self.MHI))
            self.Ropt =4.19285376433691
            self.Rs = 0.14571194842680504
            self.vflat=30.90142073662264
            self.END =  self.DHI
            self.rwidth=2.5
            self.alpha=-0.04
            self.rt = 1.953269232143619
            self.v_0 =38.27830523325882
            self.dx = -0.06
            self.Rs = 0.14571194842680504
            self.Mag =-15.566999999988472
            self.slope=0.7072740994972535
        elif vel == 25:
            self.MHI=7.5
            self.DHI=3.405473493928381
            self.Mstar=6.1049564614047735
            self.Mbar = np.log10(10**(self.Mstar) + 1.4*10**(self.MHI))
            self.Ropt = 3.496871254588844
            self.Rs = 0.3064926144535543
            self.vflat = 39.72029382879143
            self.END = self.DHI
            self.rwidth = 2.5
            self.alpha =0.156
            self.rt = 1.7311970466119948
            self.v_0 = 34.82190527193729
            self.dx = -0.02
            self.Rs = 0.3064926144535543
            self.Mag = -14.881999999987634
            self.slope = 0.638537433859023
        elif vel == 50:
            self.MHI   = 8.300000000006774 
            self.DHI   = 8.36380249860862 
            self.Mstar = 8.13596841863477 
            self.Ropt  = 4.1071482878868135 
            self.Rs    = 0.7527422248747758 
            self.vflat = 48.264404070377765 
            self.rwidth= 2.5 
            self.alpha = -0.02899999999999999 
            self.rt    = 1.5146795485236508 
            self.v_0   = 56.11453292421927 
            self.dx    = -0.024999999999999883 
            self.Rs    = 0.7527422248747758 
            self.Mag   = -17.842999999991253 
            self.slope = 0.2621613304301969 
            self.Mbar  = np.log10(10**(self.Mstar) + 1.4*10**(self.MHI)) 
            self.END   =  self.DHI
        elif vel == 100:
            self.MHI   = 9.200000000010656 
            self.DHI   = 22.911995207696453 
            self.Mstar = 9.676472179701443 
            self.Ropt  = 4.578824805661171 
            self.Rs    = 2.0620795686926807 
            self.vflat = 109.42280453194861 
            self.rwidth= 2.5 
            self.alpha = 3.469446951953614e-17 
            self.rt    = 1.1264317421095493 
            self.v_0   = 111.34582088687199 
            self.dx    = -0.0399999999999999 
            self.Rs    = 2.0620795686926807 
            self.Mag   = -20.72199999999477 
            self.slope = 0.008900425198016178 
            self.Mbar  = np.log10(10**(self.Mstar) + 1.4*10**(self.MHI)) 
            self.END   =  self.DHI

        elif vel == 250:
            self.MHI   = 10.500000000000222
            self.DHI   = 96.84753464891496
            self.Mstar = 11.413594821094087
            self.Ropt  = 6.9736743042379725
            self.Rs    = 8.716278118402347
            self.vflat = 254.26947571120243
            self.rwidth= 2.5
            self.alpha = 3.469446951953614e-17
            self.rt    = 0.9325062515078058
            self.v_0   = 254.44415010281998
            self.dx    = -0.05499999999999991
            self.Rs    = 8.716278118402347
            self.Mag   = -23.35499999999799
            self.slope = 7.640228717874136e-12
            self.Mbar  = np.log10(10**(self.Mstar) + 1.4*10**(self.MHI))
            self.END   =  self.DHI
        else:
            print("No example for that speed")

        self.calc_dist(4)

    def save(self):
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


    def reroll(self,Mass,beams,scatter):
        
        self.beams = beams
        self.beamsize = 30
        self.rwidth = 2.5

        self.MHI,self.DHI,self.Mstar,self.Ropt,self.vflat,self.sigma,self.alpha,self.rt,self.v_0,\
                self.dx,self.Rs,self.Mag,self.slope=setup_relations(Mass,self.beams,self.beamsize,self.rwidth,scatter)
        

        self.Mbar = np.log10(10**(self.Mstar) + 1.4*10**(self.MHI))
        self.END = self.DHI
        
        self.calc_dist(self.beams)
        print(self.MHI, self.DHI, self.Mstar,self.Ropt,self.vflat,self.alpha,self.dx,self.Mag,self.slope,self.dist,self.beams) 

    def calc_dist(self,beams):
        self.beams=beams
        self.dist = self.DHI * (206265./(self.beams*self.beamsize))
        self.delta = (self.rwidth/206265.)*self.dist
        self.radi = np.arange(0,self.END+self.delta,self.delta)
        
        self.polyex=self.Polyex(self.alpha,self.rt ,self.v_0)
        self.sbr = self.SBR(self.dx,self.Rs,self.DHI,self.vflat)
        self.z = self.Z(self.sigma)
        
        self.DHI_arcsec = self.DHI*206265./self.dist
        self.profiles=self.Profiles(self.polyex,self.sbr,self.radi,self.dist,self.z)
    class Polyex():
        def __init__(self,alpha,rt,v_0):
            self.alpha = alpha
            self.v_0 = v_0
            self.rt = rt 
        def v_curve(self,r):
            return self.v_0*(1.-np.exp(-r/self.rt))*(1.+self.alpha*r/self.rt)
    class Z():
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

    def make_fits(self,realizations):
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
       
        realizations = max(realizations,2)
        for num in range(1,realizations):
            filecheck = Path(self.fname+'.noise'+str(num))
            if filecheck.is_dir():
                os.system("rm -r "+self.fname+'.noise'+str(num))
                print("Refreshed folder")
            os.system("mkdir "+self.fname+'.noise'+str(num))
            os.system("cp "+self.defname+" VROT.png SBR.png SBR_log.png RC.dat "+self.fname+'.noise'+str(num))
            ######################################################################
            print("realization #",num)
            second_beam(self.outset,self.outname,self.END/ (self.dist) * 3600. * (180./np.pi),self.beams,self.snr,self.inc,self.MHI,self.dist,1.0E-6,self.beamsize,self.DHI)
            os.system("mv "+self.outname+" "+self.fname+'.noise'+str(num))
        os.system("rm "+self.outname)
        os.system("rm empty.fits")
        os.system("rm  VROT.png SBR.png SBR_log.png RC.dat "+self.outset+" Logfile.log "+self.defname)
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
        plt.show()
        plt.close()
