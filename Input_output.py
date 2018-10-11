#!/usr/bin/env python3
import numpy as np
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, Gaussian1DKernel ,convolve, convolve_fft
from scipy.optimize import curve_fit
from decimal import Decimal
import random
import os
import datetime


def deffile(outset,inset,defname,radi,vrot,sbr,inc,END,condisp,z,cflux):
    logname='Logfile.log'
    
    file = open(defname,"w") 
    file.write("LOGNAME="+logname)
    file.write("\nACTION=1")
    file.write("\nPROMPT=")
    NCORES = os.cpu_count()
    file.write("\nNCORES="+str(NCORES))
    file.write("\n")
    file.write("\nINSET="+inset)
    file.write("\nOUTSET="+outset)
    file.write("\nOUTCUBUP=10000000")
    file.write("\n")
    file.write("\nPROGRESSLOG=")
    file.write("\nTEXTLOG=")
    file.write("\n")
    file.write("\nRMS= 1.00000e-10")
    file.write("\nBMAJ=0")
    file.write("\nBMIN=0")
    file.write("\nBPA=0")
    file.write("\n")
    file.write("\nNDISKS=")
    file.write("\nNUR="+str(END))
    
    vrot_str=np.array([])
    for i in range(0,END):
        vrot_str=np.append(vrot_str,str(vrot[i]))
    
    sbr  =  sbr[0:END]
    vrot = vrot[0:END]
    radi = radi[0:END]
    z    =    z[0:END]
    vrot_str=vrot_str[0:END]
    
    file.write("\n\nRADI=")
    for i in range(0,len(vrot_str)):
        file.write(' +')
        file.write('{:.3e}'.format(radi[i]))
    file.write("\n\nVROT=")
    for i in range(0,len(vrot_str)):
        file.write(' +')
        file.write('{:.3e}'.format(vrot[i]))
    file.write("\n\nSBR=")
    for i in range(0,len(vrot_str)):
        file.write(' +')
        file.write('{:.3e}'.format(sbr[i]))
    file.write("\n\nZ=")
    for i in range(0,len(vrot_str)):
        file.write(' +')
        file.write('{:.3e}'.format(z[i]))
    file.write("\nINCL= +"+str(float(inc)))
    file.write("\nPA= +4.50000E+00")
    file.write("\nXPOS= +2.77675430E+02")
    file.write("\nYPOS= +7.34348280E+01")
    file.write("\nVSYS= +1403.93164636")
    file.write("\nSDIS="+str(float(condisp)))
    file.write("\nCONDISP=0")
    file.write("\nLTYPE= 3")
    file.write("\n")
    file.write("\nCFLUX=")
    file.write('{:.3e}'.format(cflux)) 
    file.write("\nPENALTY= 0")
    file.write("\nWEIGHT= 0 ")
    file.write("\nRADSEP= 0.1 ")
    file.write("\nINIMODE= 0 ")
    file.write("\nISEED= 8981 ")
    file.write("\n")
    file.write("\nFITMODE= 1")
    file.write("\nLOOPS=0")
    file.write("\nMAXITER=")
    file.write("\nCALLITE=")
    file.write("\nSIZE=")
    file.write("\nINTY=")
    file.write("\nINDINTY=")
    file.write("\nPSSE=")
    file.write("\nPSNP=")
    file.write("\nPSCO=")
    file.write("\nPSSO=")
    file.write("\nPSMV=")
    file.write("\nPSNF=")
    file.write("\nPSII=")
    file.write("\nPSFI=")
    file.write("\nPSID=")
    file.write("\nPSDD=")
    if END > 6: 
        file.write("\n\nVARY= INCL 1:5, !INCL 6:8,  PA 1:5 , !PA 6:, !VROT 2:8, !SBR 2:8,  Z0 1:8")
    else:
        file.write("\n\nVARY= INCL 1:5, !INCL 6:"+str(END)+",  PA 1:5 , !PA 6:"+str(END)+", !VROT 2:"+str(END)+", !SBR 2:"+str(END)+",  Z0 1:"+str(END)+"")
    file.write("\nVARINDX=")
    file.write("\nPARMAX=   90  90  360  360  500    1  30")
    file.write("\nPARMIN=    0   0 -360 -360   20    0   0")
    file.write("\nMODERATE=  3   3    3    3    3    3   3")
    file.write("\nDELSTART=0.5   2  0.5    2   16 1E-5   1")
    file.write("\nDELEND=  0.1 0.5  0.1  0.5    8 1E-6 0.1")
    file.write("\nITESTART= 70  70   70   70   70   70  70")
    file.write("\nITEEND=   70  70   70   70   70   70  70")
    file.write("\nSATDELT=   2   4    2    4   10 5E-6   1")
    file.write("\nMINDELTA=0.1 0.5  0.1  0.5    8 5E-7 0.1")
    file.write("\n")
    
    file.write("\nREGPARA=")
    file.write("\nTIRSMO=")
    file.write("\nCOOLGAL=")
    file.write("\nCOOLBEAM=")
    file.write("\nTILT=")
    file.write("\nBIGTILT=\n")

def emptyfits(inset):
    teststr="\
SIMPLE  =                    T / Written by IDL:  Fri Mar 20 16:02:29 2015      \
BITPIX  =                  -64 / IEEE double precision floating point           \
NAXIS   =                    3 / NUMBER OF AXES                                 \
NAXIS1  =              400.000 /Number of positions along axis 1                \
NAXIS2  =              400.000 /Number of positions along axis 2                \
NAXIS3  =              120.000 /Number of positions along axis 3                \
BLOCKED =                    T / TAPE MAY BE BLOCKED                            \
CDELT1  =          -0.00111111 /                                                \
CRPIX1  =              200.000 / PRIMARY REFERENCE PIXEL                        \
CRVAL1  =        277.675431576 / PRIMARY REFERENCE VALUE                        \
CTYPE1  = 'RA---TAN'           / PRIMARY AXIS NAME                              \
CUNIT1  = 'DEGREE  '           / PRIMARY AXIS UNITS                             \
CDELT2  =           0.00111111 /                                                \
CRPIX2  =              200.000 / PRIMARY REFERENCE PIXEL                        \
CRVAL2  =        73.4348279512 / PRIMARY REFERENCE VALUE                        \
CTYPE2  = 'DEC--TAN'           / PRIMARY AXIS NAME                              \
CUNIT2  = 'DEGREE  '           / PRIMARY AXIS UNITS                             \
CDELT3  =              4000.00 / PRIMARY PIXEL SEPARATION                       \
CRPIX3  =              60.0000 / PRIMARY REFERENCE PIXEL                        \
CRVAL3  =        1403931.64636 / PRIMARY REFERENCE VALUE                        \
CTYPE3  = 'VELO-LSR'           / PRIMARY AXIS NAME                              \
CUNIT3  = 'M/S     '           / PRIMARY AXIS UNITS                             \
EPOCH   =   2.000000000000E+03 / EPOCH                                          \
MAPTYP  = 'MAP'                                                                 \
BUNIT   = 'Jy/Beam           '                                                  \
BMAJ    =        0.00000000000 /                                                \
BMIN    =        0.00000000000 /                                                \
BPA     =                    0 /                                                \
END                                                                             "
    header  = fits.header.Header.fromstring(teststr)
    
    cube = np.zeros((120,400,400))
    hdu = fits.PrimaryHDU(cube,header=header)
    hlist = fits.HDUList([hdu])
    hlist.writeto(inset,overwrite=True)

def rothead(mass,Mag,Vdisp,Mbar,Mstar,DHI,vflat,Rs,dist,slope,alpha,v0,rPE):
    fille = open('RC.dat',"w")
    fille.write("#  Date = "+str(datetime.datetime.now())[0:10]+"\n")
    fille.write('#  Mag  = '+str(Mag)+'\n')
    fille.write('#  MHI   [dex Mo] = '+str(round(np.log10(mass),2))+'\n')
    fille.write('#  Mbar  [dex Mo] = '+str(round(np.log10(Mbar),2))+'\n')
    fille.write('#  Mstar [dex Mo] = '+str(round(np.log10(Mstar),2))+'\n')
    fille.write('#  Vflat   [km/s] = '+str(round(vflat,2))+"\n")
    fille.write('#  Vdisp   [km/s] = '+str(round(Vdisp,2))+'\n')
    fille.write('#  Rs       [kpc] = '+str(round(Rs,2))+'\n')
    fille.write('#  Distance [kpc] = '+str('{:.2E}'.format(dist))+'\n')
    fille.write('#  DHI     [km/s] = '+str(round(DHI,2))+"\n")
    fille.write('#  DlogV/DlogR    = '+str(round(slope,2))+"\n \n")
    fille.write('#  Polyex Rotation curve parameters: \n')
    fille.write('#  V(r) = v0*(1-exp(-r/rPE)) * (1+a*r/rPE): \n')
    fille.write('#  a            = '+str(alpha)+"\n")
    fille.write('#  v0    [km/s] = '+str(v0)+"\n")
    fille.write('#  rPE [arcsec] = '+str(np.float32(rPE))+"\n \n")
    fille.write('#  Rotation curve \n \n')

def rotfile(radi,vrot,sbr,z,END):
    fille = open('RC.dat','a')
    #fille.write("#RAD \t VROT \t \t SBR \t \t Z \n")
    fille.write("#RAD \t\tVROT \t \tSBR \t \t \tZ \n")
    fille.write("#(\") \t\t(km/s) \t\t(Jy km s^-1 as^-2) \t(kpc) \n")
    for i in range(1,END):
        fille.write(str('{:.3e}'.format(radi[i]))+'\t'+str('{:.3E}'.format(vrot[i]))+'\t'+str('{:.3E}'.format(sbr[i]))+'\t'+'\t'+str('{:.3E}'.format(z[i]))+'\n')
