#!/usr/bin/env python3
import numpy as np
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, Gaussian1DKernel ,convolve, convolve_fft
from scipy.optimize import curve_fit
from scipy import ndimage
from scipy.spatial.distance import pdist
from decimal import Decimal
import random 
import os

def second_beam(outset,outname,rmax,ba,sn,inc,mass,dist,cflux_min,beam_arcsec,DHI):
    hdulist = fits.open(outset)
    cube = hdulist[0].data
    scoop=np.sum(cube)*dist**2.*0.236*abs(hdulist[0].header['CDELT3'])/1000.
    print(np.log10(scoop),mass)
    print('Percentage of Expected Mass:',(scoop-10.**mass)/(10.**mass)*100.,'%')

    delt_d = abs(hdulist[0].header['CDELT1']) # degrees / pixel
    delt = delt_d * 3600 # arcseconds / pixel
    print('------------------')
    beam=beam_arcsec/delt #arcseconds / (arcseconds / pixel)

    fwhm = 2.*np.sqrt(2.*np.log(2.))        #FWHM  = 2.355* sigma
    bmaj_fwhm = beam #pixels; #30 arcsecond beam
    bmaj_sigma  = bmaj_fwhm / fwhm
    print('Diameter: ',round(rmax,2),' arcseconds,',round(rmax/delt,2),' pixels')
    print('BMAJ: (FWHM) ',round(bmaj_fwhm*delt,2),' arcseconds,',round(bmaj_fwhm,2),' pixels')
    gauss = Gaussian2DKernel(bmaj_sigma)
    print('Calculating Noise level')
    cflux_min = 1.E-6
    cutoff = np.mean(cube[cube>cflux_min])/(4.*np.sqrt(np.pi)*bmaj_sigma)
    smooth = ndimage.gaussian_filter(cube,sigma=(0,bmaj_sigma,bmaj_sigma),order = 0)
    mean_signal = np.mean(smooth[smooth > cutoff])

    smooth[smooth < cutoff]=0.
    smooth[smooth > cutoff]=1.
    mask = smooth 

    noise = mean_signal/sn
    pixarea=np.pi * bmaj_sigma **2.* 2.
    noisescl = mean_signal/sn*bmaj_sigma*2*np.sqrt(np.pi)
    rms = 0.75 # in mJy
    noisescl = rms *0.001 / pixarea * bmaj_sigma*2.*np.sqrt(np.pi)

    cuberms = np.random.normal(scale=noisescl,size=np.shape(cube))
    cube = ndimage.gaussian_filter(cuberms+cube,sigma=(0,bmaj_sigma,bmaj_sigma),order = 0)
    cube = cube*pixarea

    print('RMS',np.sqrt(np.mean((cube[0,:,:])**2.))/0.001,'mJy')

    prihdr = hdulist[0].header
    prihdr['OBJECT'] = 'SimGal'
    prihdr['OBSERVER']= 'Colin'
    prihdr['CUNIT1']= 'DEGREE'
    prihdr['CUNIT2']= 'DEGREE'
    prihdr['CUNIT3']= 'm/s'
    prihdr['BMAJ']  =    (bmaj_fwhm)*delt_d # pixels * degrees/pixel
    prihdr['BMIN']  =    (bmaj_fwhm)*delt_d # pixels * degrees/pixel
    prihdr['COMMENT'] = 'SN:'+str(sn)
    prihdr['COMMENT'] = 'Beams Across: '+str(ba)
    prihdr['COMMENT'] = 'Inclination:  '+str(inc)
    prihdr['COMMENT'] = 'Mass: '+str(mass)
    
    hdu = fits.PrimaryHDU(cube,header=prihdr)
    hlist = fits.HDUList([hdu])
    hlist.writeto(outname,overwrite=True)
    
    mom0 = np.sum(cube,axis=0)*abs(float(hdulist[0].header['CDELT3'])/1000.)
    flux = ( 1.247E20 * (beam_arcsec*1.42)**2.) / ( 2.229E24 )

    beamarea=(np.pi*beam_arcsec**2.)/(4.*np.log(2.))
    pixperbeam=beamarea/(abs(prihdr['CDELT1']*3600.)*abs(prihdr['CDELT2']*3600.))
    totalsignal = np.sum(cube[mask > 0.5])/pixperbeam

    Mtest1 = 0.236*dist**2*totalsignal*prihdr['CDELT3']/1000.
    #Mtest=(0.236)*(dist)**2.*np.sum(cube)*prihdr['CDELT3']/1000./((np.pi*beam**2.)/(4.*np.log(2.)))
    print(np.log10(Mtest1))
    print('Final Cube Mass Frac:',(Mtest1-10.**mass)/(10.**mass)*100.,'%')
    #print(flux)
    
    if (False):
        mom_mask = np.nan * mom0
        mom_mask = np.array([mom_mask,mom_mask,mom_mask])
        print(np.shape(mom_mask),np.shape(mom0))
        for i in range(0,len(mom_mask[0,0,:])):
            for j in range(0,len(mom_mask[0,:,0])):
                if mom0[i,j] > flux:
                    mom_mask[0,i,j] = mom0[i,j]
                    mom_mask[1,i,j] = i
                    mom_mask[2,i,j] = j
    
        hdu = fits.PrimaryHDU(mom_mask,header=prihdr)
        hlist = fits.HDUList([hdu])
        hlist.writeto('mom_mask.fits',overwrite=True)
    
    

def b_math(channel,noise,gauss):
    channel_noise = np.random.normal(loc=channel,scale=noise)+channel
    return convolve_fft(channel_noise,gauss)


