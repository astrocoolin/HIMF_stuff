#!/usr/bin/env python3
import numpy as np
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, Gaussian1DKernel ,convolve, convolve_fft
from scipy.optimize import curve_fit
from decimal import Decimal
import random 
import os

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

def make_sbr(radi,IO,hr,rout):
    SBR = radi*1.0 
    IO2 = IO * np.exp(-1.*rout/hr) / np.exp(-1.*rout/(-1.*hr/20.))
    for i, r in enumerate(radi):
        if r < rout:
            SBR[i]=IO*np.exp(-1.*r/hr)
        elif r > rout:
            SBR[i]=IO2*np.exp(-1.*r/(hr/20.))
    return SBR #IO * np.exp(-radi/hr)

def first_beam(outset,outname,rmax,ba,sn,inc,Mag):
    hdulist = fits.open(outset)
    cube = hdulist[0].data
    delt_d = abs(hdulist[0].header['CDELT1']) # degrees / pixel
    delt = delt_d * 3600 # arcseconds / pixel
    
    print('------------------')

    #print(ba,' beams across Semi-Major axis')
    fwhm = 2.*np.sqrt(2.*np.log(2.))        #FWHM  = 2.355* sigma
    bmaj_fwhm = 7.5 #pixels; #30 arcsecond beam
    bmaj_sigma  = bmaj_fwhm / fwhm
    print('Semi-Major axis: ',round(rmax,2),' arcseconds,',round(rmax/delt,2),' pixels')
    print('BMAJ: (FWHM) ',round(bmaj_fwhm*delt,2),' arcseconds,',round(bmaj_fwhm,2),' pixels')
    gauss = Gaussian2DKernel(bmaj_sigma)
    print 
    print('Calculating Noise level')
    sn_np=np.array([])
    nfrac = 0.1
    cubemax = np.max(cube)
    for vel in range(0,cube.shape[0]):
        softbeam = cube[vel,:,:] 
        if np.size(softbeam[ softbeam[:,:] > nfrac*cubemax]) > 0:
            sum_41 = np.mean(softbeam[softbeam[:,:] > nfrac*cubemax])
            sn_np = np.append(sn_np,sum_41)

    signal = np.mean(sn_np[:])
    noise = signal / sn
    noise = noise * 7.5 * (2. / fwhm) * np.sqrt(np.pi)
    
    print('PSF convolution')
    for vel in range(0,cube.shape[0]):
        if  np.size(cube[vel,(cube[vel,:,:]) > nfrac*cube[:,:,:].max()]) > 0:
            cube[vel,:,:]=b_math(cube[vel,:,:],noise,gauss)
        else:
            cube[vel,:,:]=b_math(np.zeros_like(cube[vel,:,:]),noise,gauss)
 
    
    prihdr = hdulist[0].header
    prihdr['OBJECT'] = 'SimGal'
    prihdr['OBSERVER']= 'Colin'
    prihdr['CUNIT1']= 'DEGREE'
    prihdr['CUNIT2']= 'DEGREE'
    prihdr['CUNIT3']= 'M/S'
    prihdr['BMAJ']  =    (bmaj_fwhm)*delt_d # pixels * degrees/pixel
    prihdr['BMIN']  =    (bmaj_fwhm)*delt_d # pixels * degrees/pixel
    prihdr['COMMENT'] = 'SN:'+str(sn)
    prihdr['COMMENT'] = 'Beams Across: '+str(ba)
    prihdr['COMMENT'] = 'Inclination:  '+str(inc)
    prihdr['COMMENT'] = 'Magnitude: '+str(Mag)

    hdu = fits.PrimaryHDU(cube,header=prihdr)
    hlist = fits.HDUList([hdu])
    hlist.writeto(outname,overwrite=True)

def b_math(channel,noise,gauss):
    channel_noise = np.random.normal(loc=channel*0,scale=noise)+channel
    return convolve_fft(channel_noise,gauss)


