#!/usr/bin/env python3
import numpy as np
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, Gaussian1DKernel ,convolve, convolve_fft
from scipy.optimize import curve_fit
from decimal import Decimal
import random 
import os

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


