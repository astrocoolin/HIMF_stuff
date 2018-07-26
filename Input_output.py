#!/usr/bin/env python3
import numpy as np
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, Gaussian1DKernel ,convolve, convolve_fft
from scipy.optimize import curve_fit
from decimal import Decimal
import random
import os


def deffile(outset,inset,defname,radi,vrot,sbr,inc,END,condisp,z):
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
    file.write("\nRMS= 1.00000e-5")
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
    vrot_str=vrot_str[0:END]
    
    file.write("\n\nRADI=")
    for i in range(0,len(vrot_str)):
        file.write(' +')
        file.write('%E' % Decimal(radi[i]))
    file.write("\n\nVROT=")
    for i in range(0,len(vrot_str)):
        file.write(' +')
        file.write('%E' % Decimal(vrot[i]))
    file.write("\n\nSBR=")
    file.write('0.000000E+00')
    for i in range(1,len(vrot_str)):
        file.write(' +')
        file.write('%E' % Decimal(sbr[i]))
    file.write("\n\nZ0="+str(float(z)))
    file.write("\nINCL= +6.00000E+01")
    file.write("\nPA= +4.50000E+01")
    file.write("\nXPOS= +2.77675430E+02")
    file.write("\nYPOS= +7.34348280E+01")
    file.write("\nVSYS= +1403.93164636")
    file.write("\nSDIS=0")
    file.write("\nCONDISP="+str(float(condisp)))
    file.write("\nLTYPE= 3")
    file.write("\n")
    file.write("\nCFLUX=2E-5") 
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
    
    file.write("\n\nVARY= INCL 1:5, !INCL 6:8,  PA 1:5 , !PA 6:8, !VROT 2:8, !SBR 2:8,  Z0 1:8")
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
CROTA1  =   0.000000000000E+00 / PRIMARY ROTATION OF AXIS                       \
CDELT1  =          -0.00111111 /                                                \
CRPIX1  =              200.000 / PRIMARY REFERENCE PIXEL                        \
CRVAL1  =        277.675431576 / PRIMARY REFERENCE VALUE                        \
CTYPE1  = 'RA---TAN'           / PRIMARY AXIS NAME                              \
CUNIT1  = 'DEGREE  '           / PRIMARY AXIS UNITS                             \
CROTA2  =            0.0853287 / PRIMARY ROTATION OF AXIS                       \
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
DRVAL3  =   4.565000000000E+03 / SECONDARY REFERENCE VALUE                      \
DTYPE3  = 'VELO              ' / SECONDARY AXIS NAME                            \
DUNIT3  = 'M/S               ' / SECONDARY AXIS UNITS                           \
EPOCH   =   2.000000000000E+03 / EPOCH                                          \
INSTRUME= 'WSRT              ' / INSTRUMENT                                     \
BMMAJ   = 0.35913645070246D+02                                                  \
BMMIN   = 0.23912432616063D+02                                                  \
SUBTR   =                    0 /SOURCE SUBTRACTION                              \
CHANEND =                   51 / last usable continuum channel                  \
UVCDT   = 'NORMAL     '        /UV COORDINATE TYPE                              \
BLGRAD  = 'NATURAL    '        /TAPER TYPE                                      \
MAPTYP  = 'MAP'                                                                 \
INSECT  =                    1 /INPUT SECTORS.                                  \
BUNIT   = 'Jy/Beam           '                                                  \
FIRSTLCH=                  -41 / low velocity edge of line signal               \
DATE    = '12/12/01'                                                            \
CHANSTA =                  -58 / 1st usable continuum channel                   \
FREQR   =       1399164042.224 /REFERENCE FREQUENCY (HERTZ)                     \
LASTLCH =                   30 / high velocity edge of line signal              \
DATTYP  = 'NORMAL     '        /DATA TYPE                                       \
DATE-OBS=             2000.684                                                  \
CLIP    =                    0 /CLIPPING DONE                                   \
CRESL3  =             39062.50 /BANDWIDTH (HERTZ)                               \
CORGRID =                    1 /CORRECT FOR CONVOLUTION                         \
BANDW   =              4960938 /TOTAL BANDWIDTH OF OBS(HERTZ)                   \
VEL     =              4030617 /CENTRE VELOCITY (M/S)                           \
DEBEAM  =                    0 /DE-BEAM COUNT                                   \
NORM    =             14210.61 /NORM. FACTOR IN FFT                             \
VELR    =              4565000 /REFERENCE VELOCITY (M/S)                        \
SETNR   =                    1 /# OF SET.                                       \
BMAJ    =        0.00000000000 /                                                \
BMIN    =        0.00000000000 /                                                \
BPA     =                    0 /                                                \
END                                                                             "
    header  = fits.header.Header.fromstring(teststr)
    
    cube = np.zeros((120,400,400))
    hdu = fits.PrimaryHDU(cube,header=header)
    hlist = fits.HDUList([hdu])
    hlist.writeto(inset,overwrite=True)


def rotfile(radi,vrot,sbr,END):
    fille = open('RC.txt',"w")
    fille.write("#RAD \t VROT \t \t SBR \n")
    fille.write("#(\") \t (km/s) \t (Jy km s^-1 arcsec^-2) \n")
    for i in range(1,END):
        fille.write(str(radi[i])+'\t'+str(vrot[i])+'\t'+str(sbr[i])+'\n')