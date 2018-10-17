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

def first_beam(outset,outname,rmax,ba,sn,inc,mass):
    hdulist = fits.open(outset)
    cube = hdulist[0].data
    cube_in = hdulist[0].data
    delt_d = abs(hdulist[0].header['CDELT1']) # degrees / pixel
    delt = delt_d * 3600 # arcseconds / pixel
    print('------------------')
    fwhm = 2.*np.sqrt(2.*np.log(2.))        #FWHM  = 2.355* sigma
    bmaj_fwhm = 7.5 #pixels; #30 arcsecond beam
    bmaj_sigma  = bmaj_fwhm / fwhm
    print('Diameter: ',round(rmax,2),' arcseconds,',round(rmax/delt,2),' pixels')
    print('BMAJ: (FWHM) ',round(bmaj_fwhm*delt,2),' arcseconds,',round(bmaj_fwhm,2),' pixels')
    gauss = Gaussian2DKernel(bmaj_sigma)
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
    noise = noise * 2.*np.sqrt(np.pi)* bmaj_sigma

    print('Noise:',noise)
    print('Signal:',signal)
    
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
    prihdr['COMMENT'] = 'Mass: '+str(mass)

    hdu = fits.PrimaryHDU(cube_in,header=prihdr)
    hlist = fits.HDUList([hdu])
    hlist.writeto(outname,overwrite=True)

def second_beam(outset,outname,rmax,ba,sn,inc,mass,dist,cflux_min,beam_arcsec):
    hdulist = fits.open(outset)
    cube = hdulist[0].data
    scoop=np.sum(cube)*dist**2.*0.236*abs(hdulist[0].header['CDELT3'])/1000.
    print('TestMass',mass,np.log10(scoop),(scoop-10.**mass)/(10.**mass)*100.,'%')

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

    cuberms = np.random.normal(scale=noisescl,size=np.shape(cube))
    cube = ndimage.gaussian_filter(0.*cuberms+cube,sigma=(0,bmaj_sigma,bmaj_sigma),order = 0)
    cube = cube*pixarea

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
    Mtest=(0.236)*(dist)**2.*np.sum(cube)*prihdr['CDELT3']/1000./((np.pi*beam**2.)/(4.*np.log(2.)))
    #print(flux)

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

    #print('determining the radius')
    mini = int(np.min(mom_mask[1,np.isfinite(mom_mask[1,:,:])]))
    minj = int(np.min(mom_mask[2,np.isfinite(mom_mask[2,:,:])]))
    maxi = int(np.max(mom_mask[1,np.isfinite(mom_mask[1,:,:])]))
    maxj = int(np.max(mom_mask[2,np.isfinite(mom_mask[2,:,:])]))
    #print(mini,minj,maxi,maxj)

    dists = maxi-mini
    #print(dists)
    f = open('distances.txt','a')
    f.write(str(np.log10(Mtest))+' '+str(np.log10(Mtest1))+' '+str(inc)+' '+str(dists)+" "+str(np.log10(dist))+'\n')
    f.close()

def b_math(channel,noise,gauss):
    channel_noise = np.random.normal(loc=channel,scale=noise)+channel
    return convolve_fft(channel_noise,gauss)


