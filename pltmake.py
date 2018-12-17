#!/usr/bin/env python3
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import ascii
from pathlib import Path

plt1 = False

import matplotlib.gridspec as gridspec
alldelta = np.array([])
alldelta_r = np.array([])
#xy = np.array([])
xy = []

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
    infile.close()

    #Finding r where sbr drops < 0.5
    r_prof = ascii.read(fname[0]+"/RC.dat")['col1']
    sbr = ascii.read(fname[0]+"/RC.dat")['col3'] 
    conv=6.0574E5*1.823E18*(2.*np.pi/np.log(256.))
    sbr = sbr/1.24756e+20*(conv)
    R_HI5 = r_prof[np.argmin(abs(sbr - 0.5))]
    DHI = ((DHI/2.) / Distance)*3600.*180./np.pi
    #print(DHI,R_HI5)
    
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

    print(np.unique(RADI))
    RADI_med = np.unique(RADI)
    RADI_med = RADI_med[np.isfinite(RADI_med)]

    VROT_med  = np.array([])
    VROT_mean = np.array([])

    iqr = np.array([])
    devarray = np.array([])
    
    RADI_2 = np.arange(0,np.max(RADI_med)+0.1,0.1)
    polyex = v0 * (1. - np.exp(-RADI_2/rPE)) * (1. + RADI_2 * a /rPE )
    polyex_discrete = v0 * (1. - np.exp(-RADI_med/rPE)) * (1. + RADI_med * a /rPE )

    for i,radius in enumerate(RADI_med):

        VROT_med= np.append(VROT_med,
                np.median(VROT[(RADI==radius) & (np.isfinite(VROT))]))

        VROT_mean=np.append(VROT_mean,
                np.mean(VROT[(RADI==radius) & (np.isfinite(VROT))]))

        VROT_temp = VROT[(RADI==radius) & (np.isfinite(VROT))]

        #if VROT_temp.size > 4  or (i < instr_len):
        top,bot = np.percentile(VROT_temp, [75 ,25],axis=0)

        iqr = np.append(iqr,top-bot)
        devarray = np.append(devarray,np.std(VROT_temp))
    Bias = np.array([]) 
    for i in range(0,len(RADI)):
        RADI_temp = RADI[i][np.isfinite(RADI[i])]
        C = (1./50.) / len(RADI_temp[RADI_temp < R_HI5])
        polyex_discrete = v0 * (1. - np.exp(-RADI_temp/rPE)) * (1. + RADI_temp * a /rPE )
        B = np.array([])
        sig = np.array([])
        for j,r in enumerate(RADI_temp):
            if r < R_HI5:
                #print(VROT[i][j],polyex_discrete[j],iqr[RADI_med==r],r)
                B = np.append(B,VROT[i][j] - polyex_discrete[j])
                sig = np.append(sig,iqr[RADI_med==r])
        B = B[np.isfinite(B)]
        #print(np.sum((C*B/sig)[np.isfinite(C*B/sig)]))
        Bias = np.append(Bias,np.sum((C*B/sig)[np.isfinite(C*B/sig)]))
    Total_Bias = np.sum(Bias)
    
    polyex_discrete = v0 * (1. - np.exp(-RADI_med/rPE)) * (1. + RADI_med * a /rPE )
    residual = VROT_med-polyex_discrete
    residual_int = np.zeros_like(residual)*np.nan
    error_int = np.zeros_like(residual)*np.nan
    for i, r in enumerate(residual):
            residual_int[i] = np.sum(residual[0:i+1])
            error_int[i] = np.sum(iqr[0:i+1])

    delta_1 = np.array([])
    delta_2 = np.array([])
    iqr_within = np.array([])
    dev_within = np.array([])
    norm = 0
    for i,r in enumerate(RADI_med):
        if r <= R_HI5:
            norm = norm+1
            delta_1 = np.append(delta_1,VROT_med[i] -polyex[RADI_2 ==r])
            delta_2 = np.append(delta_2,VROT_mean[i]-polyex[RADI_2 ==r])
            iqr_within = np.append(iqr_within,iqr[i])
            dev_within = np.append(dev_within,devarray[i])

    RADI_med = RADI_med[np.isfinite(VROT_med)]
    VROT_med = VROT_med[np.isfinite(VROT_med)]
    
    alldelta = np.append(alldelta,delta_1)
    alldelta_r = np.append(alldelta_r,RADI_med[RADI_med <= R_HI5]/R_HI5)
    chimed=np.array([])
    chi2=np.array([])
    chimed = np.sum(delta_1[np.nonzero(iqr_within)]**2./iqr_within[np.nonzero(iqr_within)]**2.)/norm
    chi2   = np.sum(delta_2[np.nonzero(iqr_within)]**2./dev_within[np.nonzero(iqr_within)]**2.)/norm
    xy.append([[chimed,Total_Bias]])

    if plt1: 
        fig=plt.figure(figsize=(12,8))
        gs0 = gridspec.GridSpec(1, 2)
        gs00 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0[0])

        ax1 = plt.Subplot(fig, gs00[0:2, :])
        fig.add_subplot(ax1)
        ax3 = plt.Subplot(fig, gs00[2, :])
        fig.add_subplot(ax3)

        
        ax1.plot(RADI_med,VROT_med, lw = 2.5, label = 'Recovered Curve')
        ax1.fill_between(RADI_med, VROT_med-iqr, VROT_med+iqr,alpha=0.4)
        ax1.plot(RADI_2,polyex, lw = 2.5 , label = 'Input Curve')
        ax1.set_ylabel('Vrot (km/s)')
        ax1.set_xlabel('Radius (arcsec)')

        ax2 = ax1.twiny()
        plt.title(namestr+" \nChi^2 = "+str(chi2),y=1.10)
        ax2.set_xlim(0,Distance*DHI/(206265*2.))
        ax2.set_xlabel('\n Radius (kpc)')

        ax1.axvline(R_HI5)
        ax1.legend()
        N=(len(np.unique(RADI[np.isfinite(RADI)])))
        ax3.hist(RADI[np.isfinite(RADI)],N,facecolor='white',edgecolor='#1f77b4')
        ax3.hist(RADI[np.isfinite(RADI)],N,facecolor='#1f77b4',edgecolor='#1f77b4',alpha=0.1)
        ax3.set_ylabel('Number of Points \n per Radial Bin')
        ax3.set_xlabel('Radius (arcsec)')

        gs01 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0[1])
        gx1 = plt.Subplot(fig, gs01[0:2, :])
        fig.add_subplot(gx1)
        gx2 = plt.Subplot(fig, gs01[2, :])
        fig.add_subplot(gx2)
 
        gx1.plot(RADI_med,residual)
        gx1.fill_between(RADI_med, residual-iqr, residual+iqr,alpha=0.4)
        gx1.fill_between(RADI_med, residual, 0 ,where= residual>0,alpha=0.4,color='green',interpolate=True)
        gx1.fill_between(RADI_med, residual, 0 ,where= residual<0,alpha=0.4,color='red',interpolate=True)
        gx1.axvline(R_HI5)

        gx1.set_title('"Bias"='+str(Total_Bias))

        gx2.plot(RADI_med,residual_int)
        gx2.axvline(R_HI5)

        gx2.fill_between(RADI_med,residual_int, 0 \
                ,where= residual_int>0,alpha=0.4,color='green',interpolate=True)
        gx2.fill_between(RADI_med,residual_int, 0 \
                ,where= residual_int<0,alpha=0.4,color='red',interpolate=True)

        plt.tight_layout()
        plt.savefig(namestr+'plot.png',bbox_inches='tight')
        plt.close()

fig=plt.figure(figsize=(12,8))
xy = np.array(xy)
for i in range(0,len(xy)):
    if (xy[i,0][0]) < 0 or (xy[i,0][1]) < 0:
        plt.scatter((xy[i,0][0]),(xy[i,0][1]),color='red')
    else:
        plt.scatter((xy[i,0][0]),(xy[i,0][1]),color='blue')
plt.xscale('symlog')
plt.yscale('symlog')
plt.ylabel('"Bias"')
plt.xlabel('Chi2')
plt.savefig('test1.png',bbox_inches='tight')
plt.close()
fig=plt.figure(figsize=(12,8))
plt.xlabel('R/R_HI')
plt.ylabel('Vrot - Polyex')
plt.scatter(alldelta_r[alldelta<0],alldelta[alldelta<0])
plt.scatter(alldelta_r[alldelta>0],alldelta[alldelta>0])
plt.savefig('test2.png',bbox_inches='tight')

#plt.show()
