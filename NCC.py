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
        self.beams = 4.
        self.beamsize = 30.
        
        self.Mstar = 9.31760909803635
        self.Mbar = np.log10(10**(self.Mstar) + 1.4*10**(self.MHI))
        
        self.Ropt = 10.88891262333259
        self.vflat = 140.13278394294278
        self.sigma = 2.
        
        self.END = self.DHI
        
        self.rwidth = 2.5
        
        self.alpha = 0.017
        self.rt = 8.431914405739636
        self.v_0 = 114.75505945574942
        self.dx = -0.025
        self.Rs = 3.0487
        self.Mag = -21.5
        self.slope = 0.0
    def reroll(self,Mass,beams,beam_size,ring_thickness):
        
        self.MHI,self.DHI,self.Mstar,self.Ropt,self.vflat,self.sigma,self.alpha,self.rt,self.v_0,\
                self.dx,self.Rs,self.Mag,self.slope=setup_relations(Mass,beams,beam_size,ring_thickness)
        
        self.calc_dist(beams)

        self.Mbar = np.log10(10**(self.Mstar) + 1.4*10**(self.MHI))
        self.END = self.DHI
        

    def calc_dist(self,beams):
        self.dist = self.DHI * (206265./(beams*self.beamsize))
        self.delta = (self.rwidth/206265.)*self.dist
        self.radi = np.arange(0,self.END+self.delta,self.delta)
        
        self.polyex=self.Polyex(self.alpha,self.rt ,self.v_0)
        self.sbr = self.SBR(self.dx,self.Rs,self.DHI,self.vflat)
        self.z = self.Z(self.sigma)

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
        self.make_plots()
        defname ="cube_input.def"
        inset   ='empty.fits'
        outset  ='Cube_base.fits'
        outname ='Cube_ba_'+str(self.beams)+".mass_"+str(round(self.MHI,5))+".inc_"+\
                str(self.inc)+".SN_"+str(self.snr)+'.fits'
        fname ="ba_"+str(self.beams)+".mass_"+str(round(self.MHI,5))+".inc_"+\
                str(self.inc)+".SN_"+str(self.snr)
        
        #Make_Output
        filecheck = Path(defname)
        if filecheck.is_file(): os.system("rm "+defname)
        
        rothead(self.MHI,self.Mag,self.sigma,self.Mbar,self.Mstar,self.DHI,self.vflat,self.Rs,self.dist,self.slope,self.alpha,self.v_0,self.rt,self.Ropt,self.dx)
        rotfile(self.profiles.radi,self.profiles.vrot,self.profiles.sbr,self.profiles.z,len(self.profiles.radi))
        
        deffile(outset,inset,defname,self.profiles.radi,self.profiles.vrot,self.profiles.sbr,self.inc,\
            len(self.radi),0.0,self.profiles.z,1.0E-6)
        
        filecheck = Path(outname)
        if filecheck.is_file(): os.system("rm "+outname)
        filecheck = Path('empty.fits')
        if filecheck.is_file(): os.system("rm empty.fits")
        #filecheck = Path('Logfile.log')
        #if filecheck.is_file(): os.system("rm Logfile.log")
        
        emptyfits(inset)
        os.system("tirific deffile="+defname)
        print("Cube finished")
        
        num = 1
        filecheck = Path(fname+'.noise'+str(num))
        for num in range(1,realizations+1):
            if filecheck.is_dir():
                os.system("rm -r "+fname+'.noise'+str(num))
                print("Refreshed folder")
            os.system("mkdir "+fname+'.noise'+str(num))
            os.system("cp "+defname+' '+outset+" VROT.png SBR.png SBR_log.png RC.dat "+fname+'.noise'+str(num))
            ######################################################################
            print("realization #",num)
            second_beam(outset,outname,self.END/ (self.dist) * 3600. * (180./np.pi),self.beams,self.snr,self.inc,self.MHI,self.dist,1.0E-6,self.beamsize,self.DHI)
            os.system("mv "+outname+" "+fname+'.noise'+str(num))
        os.system("rm "+outname)
        os.system("rm empty.fits")
        os.system("rm  VROT.png SBR.png SBR_log.png RC.dat ")
    def make_plots(self):
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
        plt.savefig('SBR.png',bbox_inches='tight')
        #plt.show()
        plt.close()

        fig2, ax2 = plt.subplots(figsize=(20, 10))
        plt.title('log$_{10}$ MHI [M$_{\odot}$] ='+str(self.MHI)+'\nlog$_{10}$ Mstar [M$_{\odot}$] ='+str(self.Mstar),fontsize=label_size)
        plt.semilogy(self.profiles.radi,self.profiles.sbr)
        plt.xlabel('R [arcsec]',fontsize=label_size)
        plt.ylabel('SBR [Jy km s$^{-1}$ arcsec$^{-2}$]',fontsize=label_size)
        plt.axvline((self.DHI/2.)/ (self.dist) * 3600. * (180./np.pi))
        minorLocator = mpl.ticker.AutoMinorLocator()
        ax2.xaxis.set_minor_locator(minorLocator)
        plt.savefig('SBR_log.png',bbox_inches='tight')
        #plt.show()
        plt.close()

        fig3, ax3 = plt.subplots(figsize=(20, 10))
        plt.title('log$_{10}$ MHI [M$_{\odot}$] ='+str(self.MHI)+'\nlog$_{10}$ Mstar [M$_{\odot}$] ='+str(self.Mstar),fontsize=label_size)
        plt.plot(self.profiles.radi,self.profiles.vrot)
        plt.xlabel('R [arcsec]',fontsize=label_size)
        plt.ylabel('Vc [km/s]',fontsize=label_size)
        plt.axvline((self.DHI/2.)/ (self.dist) * 3600. * (180./np.pi))
        minorLocator = mpl.ticker.AutoMinorLocator()
        ax3.xaxis.set_minor_locator(minorLocator)
        plt.savefig('VROT.png',bbox_inches='tight')
        plt.close()
        #plt.show()
