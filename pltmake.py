#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from pathlib import Path

import matplotlib.gridspec as gridspec


for gal in (1,2,3,4,5,):
 for mass in (6.5,6.0,7.5,8.0,8.5,9.0,):
  for ba in (5,):
    namestr = "ba_"+str(ba)+".0.mass_"+str(mass)+".inc_60.0.SN_8.0.gal_"+str(gal)+".noise"
    if namestr == "ba_5.0.mass_6.0.inc_60.0.SN_8.0.gal_5.noise":
        break
    if namestr == "ba_5.0.mass_8.5.inc_60.0.SN_8.0.gal_5.noise":
        break
    num = np.arange(1,51)
    
    fname = np.array([])
    for i, valu in enumerate(num):
        fname = np.append(fname,namestr+str(valu))
    
    infile = open(fname[0]+"/RC.dat")
    lines=infile.readlines()
    a = float(lines[14][18:])
    v0 = float(lines[15][18:])
    rPE = float(lines[16][18:])
    DHI = float(lines[9][19:])
    Distance = float(lines[8][19:])
    
    DHI = ((DHI/2.) / Distance)*3600.*180./np.pi
    
    VROT = np.zeros([50,50])*np.nan
    RADI = np.zeros([50,50])*np.nan
    
    instr = 0
    instr_len = 0
    for num,valu in enumerate(fname):
        existcheck = Path(valu+"/Finalmodel/Finalmodel.def")
        if existcheck.is_file():
            outfile = open(valu+"/Finalmodel/Finalmodel.def")
        
            lines = outfile.readlines()
            for i, line in enumerate(lines):
                if   'VROT' in line and "+" in line and "_2" not in line:
                    radi_test = instr
                    instr = np.array(np.float32(line.split()[1:]))
                    VROT[num,0:len(instr)]=instr
                if   'RADI' in line and "+" in line and "_2" not in line:
                    instr = np.array(np.float32(line.split()[1:]))
                    RADI[num,0:len(instr)] = instr
                    if len(instr) > instr_len:
                        instr_len = len(instr)
        else:
            print(num,'Doesnt Exist')

    RADI_med = np.unique(RADI)
    RADI_med = RADI_med[np.isfinite(RADI_med)]

    VROT_med  = np.array([])
    VROT_mean = np.array([])

    iqr = np.array([])
    devarray = np.array([])
    
    RADI_2 = np.arange(0,np.max(RADI_med)+0.1,0.1)
    polyex = v0 * (1. - np.exp(-RADI_2/rPE)) * (1. + RADI_2 * a /rPE )

    for i,radius in enumerate(RADI_med):

        VROT_med= np.append(VROT_med,
                np.median(VROT[(RADI==radius) & (np.isfinite(VROT))]))

        VROT_mean=np.append(VROT_mean,
                np.mean(VROT[(RADI==radius) & (np.isfinite(VROT))]))

        VROT_temp = VROT[(RADI==radius) & (np.isfinite(VROT))]

        if VROT_temp.size > 4  or (i < instr_len):
            top,bot = np.percentile(VROT_temp, [75 ,25],axis=0)

            iqr = np.append(iqr,top-bot)
            devarray = np.append(devarray,np.std(VROT_temp))

    delta_1 = np.array([])
    delta_2 = np.array([])

    for i,r in enumerate(RADI_med):
        delta_1 = np.append(delta_1,abs(VROT_med[i] -polyex[RADI_2 ==r]))
        delta_2 = np.append(delta_2,abs(VROT_mean[i]-polyex[RADI_2 ==r]))

    RADI_med = RADI_med[np.isfinite(VROT_med)]
    VROT_med = VROT_med[np.isfinite(VROT_med)]

    norm = len(iqr[np.nonzero(iqr)])
    chimed=np.array([])
    chi2=np.array([])
    
    chimed = np.sum(delta_1[np.nonzero(iqr)]**2./iqr[np.nonzero(iqr)])/norm
    chi2   = np.sum(delta_2[np.nonzero(iqr)]**2./devarray[np.nonzero(iqr)])/norm
   
    fig=plt.figure(figsize=(12,8))
    #gs = gridspec.GridSpec(2, 1,height_ratios=[3,1])

    gs0 = gridspec.GridSpec(1, 2)
    gs00 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0[0])

    ax1 = plt.Subplot(fig, gs00[0:2, :])
    fig.add_subplot(ax1)
    ax3 = plt.Subplot(fig, gs00[2, :])
    fig.add_subplot(ax3)


    #ax1 = plt.subplot(gs[0])
    #ax3 = plt.subplot(gs[1])
    
    ax1.plot(RADI_med,VROT_med, lw = 2.5, label = 'Recovered Curve')
    ax1.fill_between(RADI_med, VROT_med-iqr, VROT_med+iqr,alpha=0.4)
    ax1.plot(RADI_2,polyex, lw = 2.5 , label = 'Input Curve')
    ax1.set_ylabel('Vrot (km/s)')
    ax1.set_xlabel('Radius (arcsec)')

    ax2 = ax1.twiny()

    #print(Distance*DHI/(206265*2.))
    plt.title(namestr+" \nChi^2 = "+str(chi2),y=1.10)
    ax2.set_xlim(0,Distance*DHI/(206265*2.))
    ax2.set_xlabel('\n Radius (kpc)')

    ax1.axvline(DHI)
    ax1.legend()
    N=(len(np.unique(RADI[np.isfinite(RADI)])))
    ax3.hist(RADI[np.isfinite(RADI)],N,facecolor='white',edgecolor='#1f77b4')
    ax3.hist(RADI[np.isfinite(RADI)],N,facecolor='#1f77b4',edgecolor='#1f77b4',alpha=0.1)
    ax3.set_ylabel('Number of Points \n per Radial Bin')
    ax3.set_xlabel('Radius (arcsec)')

    from astropy.io import fits
    basename = "ba_"+str(ba)+".0.mass_"+str(mass)+".inc_60.0.SN_8.0"
    
    pvin  = basename+".gal_"+str(gal)+".noise1/PV-Diagrams/Cube_"+basename+"_preprocessed_small_0_xv.fits"
    existcheck = Path(pvin)
    if not existcheck.is_file():
        pvin = basename+".gal_"+str(gal)+".noise1/PV-Diagrams/Cube_"+basename+"_small_0_xv.fits"

    hdulist = fits.open(pvin)
    pvin_cube = hdulist[0].data
        
    pvout = basename+".gal_"+str(gal)+".noise1/PV-Diagrams/Finalmodel_xv.fits"
    hdulist = fits.open(pvout)
    pvout_cube = hdulist[0].data
    
    gs01 = gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec=gs0[1])
    gx1 = plt.Subplot(fig, gs01[0:1, 0:3])
    fig.add_subplot(gx1)

    gx2 = plt.Subplot(fig, gs01[0:1, 3:6])
    fig.add_subplot(gx2)

    gx3 = plt.Subplot(fig, gs01[1:2, 0:3])
    fig.add_subplot(gx3)
    
    gx4 = plt.Subplot(fig, gs01[1:2, 4:5])
    fig.add_subplot(gx4)
    
    wdiff = np.shape(pvin_cube)[1] - np.shape(pvout_cube)[1]
    if wdiff > 0:
        start = int(wdiff/2)
        pvin_cube=pvin_cube[:,start:-start]
    elif wdiff < 0:
        start = int(abs(wdiff/2))
        pvout_cube=pvout_cube[:,start:-start]

    top = np.max(np.array([pvout_cube,pvin_cube]))
    bot = np.min(np.array([pvout_cube,pvin_cube]))
    
    gx1.imshow(pvin_cube,vmin=bot,vmax=top,cmap='gray')
    gx1.axis('off')

    gx2.imshow(pvout_cube,vmin=bot,vmax=top,cmap='gray')
    gx2.axis('off')

    gx3.imshow(abs(pvin_cube-pvout_cube),vmin=bot,vmax=top,cmap='gray')
    gx3.axis('off')

    import matplotlib as mpl
    cmap = mpl.cm.gray
    norm = mpl.colors.Normalize(vmin=bot, vmax=top)
    cb1 = mpl.colorbar.ColorbarBase(gx4, cmap=cmap,
                                norm=norm,
                                orientation='vertical')

    gx1.set_title('Sum='+str(np.sum(pvin_cube))+"\nMax="+str(np.max(pvin_cube))+"\nMin="+str(np.min(pvin_cube)))
    gx2.set_title('Sum='+str(np.sum(pvout_cube))+"\nMax="+str(np.max(pvout_cube))+"\nMin="+str(np.min(pvout_cube)))
    gx3.set_title('Sum='+str(np.sum(pvin_cube-pvout_cube))+"\nMax="+str(np.max(pvin_cube-pvout_cube))+"\nMin="+str(np.min(pvin_cube-pvout_cube)))
    gx4.set_title('Colorbar')
   
    plt.tight_layout()
    plt.show()

