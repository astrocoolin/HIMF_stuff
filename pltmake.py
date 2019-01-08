#!/usr/bin/env python3
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import ascii
from pathlib import Path
import matplotlib.gridspec as gridspec




# am I plotting the data individually
plt1 = True

# array of all points 
alldelta = np.array([])
alldelta_r = np.array([])

# array of statistical measures
xy = []

# all galaxies
for mass in (7.5,):
  for ba in (4,5,6,8,):
      for inc in (10,35,65,80,):
        namestr = "ba_"+str(ba)+".0.mass_"+str(mass)+".inc_"+str(inc)+".0.SN_8.0.noise"
        num = np.arange(1,51)
       
        #names of all the stuff
        fname = np.array([])
        for i, valu in enumerate(num):
            fname = np.append(fname,namestr+str(valu))
        
        infile = open(fname[0]+"/RC.dat")
        lines=infile.readlines()
        a = float(lines[17][18:])
        v0 = float(lines[18][18:])
        rPE = float(lines[19][18:])
        DHI = float(lines[9][19:])
        Distance = float(lines[8][19:])
        infile.close()

        
        # in arcseconds
        rPE = rPE*206265/ (Distance)

        print(a,v0,rPE,DHI,Distance,fname[0])

        #Finding r where sbr drops < 0.5
        r_prof = ascii.read(fname[0]+"/RC.dat")['col1']
        sbr = ascii.read(fname[0]+"/RC.dat")['col3'] 
        conv=6.0574E5*1.823E18*(2.*np.pi/np.log(256.))
        sbr = sbr/1.24756e+20*(conv)
        R_HI5 = r_prof[np.argmin(abs(sbr - 0.5))]
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
            #else:
            #    print(num,'Doesnt Exist')
        
        # unique radii, finite values only
        RADI_med = np.unique(RADI)
        RADI_med = RADI_med[np.isfinite(RADI_med)]
        
        VROT_med  = np.array([])
        VROT_mean = np.array([])
        
        iqr = np.array([])
        devarray = np.array([])
       
        # smooth radii for analytical ployex
        RADI_2 = np.arange(0,np.max(RADI_med)+0.1,0.1)
        polyex = v0 * (1. - np.exp(-RADI_2/rPE)) * (1. + RADI_2 * a /rPE )
        # polyex curve with same points as the unique radii
        polyex_discrete = v0 * (1. - np.exp(-RADI_med/rPE)) * (1. + RADI_med * a /rPE )

        VROT_T = VROT[np.isfinite(RADI)]
        RADI_T = RADI[np.isfinite(RADI)]
        RADI_bind = np.array([])

        binw=5
        for i,radius in enumerate(RADI_med):
            close = (abs(RADI_T - radius) < binw) & (RADI_T != 0)
            if radius ==0:
                close = RADI_T==0
            if abs(RADI_med[i] - RADI_med[i-1]) > binw or RADI_med[i-1] == 0:

                RADI_bind = np.append(RADI_bind,radius)
                # median rotational velocity at each radius
                VROT_med= np.append(VROT_med,
                    np.median(VROT_T[(close) & (np.isfinite(VROT_T))]))
        
                # mean rotational velocity at each radius
                VROT_mean=np.append(VROT_mean,
                    np.mean(VROT_T[(close) & (np.isfinite(VROT_T))]))
              
                VROT_temp = VROT_T[(close) & (np.isfinite(VROT_T))]
        
                #if VROT_temp.size > 4  or (i < instr_len):
                top,bot = np.percentile(VROT_temp, [75 ,25],axis=0)
       
            
                iqr = np.append(iqr,top-bot)
                devarray = np.append(devarray,np.std(VROT_temp))



            RADI_med_stats = np.unique(RADI)
            RADI_med_stats = RADI_med[np.isfinite(RADI_med)]
        
            VROT_med_stats  = np.array([])
            iqr_stats = np.array([])
            devarray_stats = np.array([])
            for i,radius in enumerate(RADI_med):
                # median rotational velocity at each radius
                VROT_med_stats= np.append(VROT_med_stats,
                    np.median(VROT[(RADI==radius) & (np.isfinite(VROT))]))

                VROT_temp = VROT[(RADI==radius) & (np.isfinite(VROT))]

                #if VROT_temp.size > 4  or (i < instr_len):
                top,bot = np.percentile(VROT_temp, [75 ,25],axis=0)


                iqr_stats = np.append(iqr_stats,top-bot)
                devarray_stats = np.append(devarray_stats,np.std(VROT_temp))



        Bias = np.array([])
        Chi2 = np.array([])
        for i in range(0,len(RADI)):
            RADI_temp = RADI[i][np.isfinite(RADI[i])]
            if len(RADI_temp[RADI_temp < R_HI5]) <= 0: continue
            C = (1./50.) / len(RADI_temp[RADI_temp < R_HI5])
            polyex_discrete = v0 * (1. - np.exp(-RADI_temp/rPE)) * (1. + RADI_temp * a /rPE )
            B = np.array([])
            sig = np.array([])
            #print(VROT[i][np.isfinite(VROT[i])],polyex_discrete,i)

            for j,r in enumerate(RADI_temp):
                if r < R_HI5:
                    B = np.append(B,VROT[i][j] - polyex_discrete[j])
                    #print(B[i],sig[i],i)
                    top,bot = np.percentile(VROT[(RADI==r) & np.isfinite(VROT)], [75 ,25],axis=0)
                    sig = np.append(sig,top-bot)
            sig = sig[np.isfinite(B)]
            B = B[np.isfinite(B)]

            Bias = np.append(Bias,np.sum((C*B/sig)[np.isfinite(C*B/sig)]))
            Chi2 = np.append(Chi2,np.sum((C*B**2./sig**2.)[np.isfinite(C*B**2./sig**2.)]))

        Total_Bias = np.sum(Bias)
        Total_Chi = np.sum(Chi2)
        #print(Total_Bias,Total_Chi,namestr)
        
        polyex_discrete = v0 * (1. - np.exp(-RADI_bind/rPE)) * (1. + RADI_bind * a /rPE )
        residual = VROT_med-polyex_discrete
        residual_int = np.zeros_like(residual)*np.nan
        error_int = np.zeros_like(residual)*np.nan

        for i, r in enumerate(residual):
                residual_int[i] = np.sum(residual[0:i+1])
                error_int[i] = np.sum(iqr[0:i+1])
        
        delta = np.array([])
        RADI_med_stats = RADI_med_stats[np.isfinite(VROT_med_stats)]
        VROT_med_stats = VROT_med_stats[np.isfinite(VROT_med_stats)]

        for i,r in enumerate(RADI_med):
            if r <= R_HI5:
                delta = np.append(delta,VROT_med_stats[i] -polyex[RADI_2 ==r])
        
        alldelta = np.append(alldelta,delta)
        alldelta_r = np.append(alldelta_r,RADI_med_stats[RADI_med_stats <= R_HI5]/R_HI5)
        xy.append([[Total_Chi,Total_Bias]])
       
        if False:
            if namestr == "ba_6.0.mass_7.0.inc_10.0.SN_8.0.noise":
                for i in range(0,50):
                  if len(VROT[i][np.isfinite(VROT[i])]) != 0:
                    #print(fname[i])
                    print(RADI[i][np.isfinite(VROT[i])],i)
                    print(VROT[i][np.isfinite(VROT[i])])
                    #print(i)

        
        if plt1: 
            fig=plt.figure(figsize=(12,8))
            gs0 = gridspec.GridSpec(1, 2)
            gs00 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0[0])
        
            ax1 = plt.Subplot(fig, gs00[0:2, :])
            fig.add_subplot(ax1)
            ax3 = plt.Subplot(fig, gs00[2, :])
            fig.add_subplot(ax3)
        
            
            ax1.plot(RADI_bind,VROT_med, lw = 2.5, label = 'Recovered Curve')
            ax1.fill_between(RADI_bind, VROT_med-iqr, VROT_med+iqr,alpha=0.4)
            ax1.plot(RADI_2,polyex, lw = 2.5 , label = 'Input Curve')
            ax1.set_ylabel('Vrot (km/s)')
            ax1.set_xlabel('Radius (arcsec)')
 #           ax1.set_xlim(0,DHI)
        
            ax2 = ax1.twiny()
            plt.title(namestr+" \nChi^2 = "+str(Total_Chi),y=1.10)
#            ax2.set_xlim(0,Distance*DHI/(206265*2.))
            ax2.set_xlabel('\n Radius (kpc)')
        
            ax1.axvline(R_HI5)
            ax1.legend()
            N=(len(np.unique(RADI[np.isfinite(RADI)])))
            ax3.hist(RADI[np.isfinite(RADI)],N,facecolor='white',edgecolor='#1f77b4')
            ax3.hist(RADI[np.isfinite(RADI)],N,facecolor='#1f77b4',edgecolor='#1f77b4',alpha=0.1)
            ax3.set_ylabel('Number of Points \n per Radial Bin')
            ax3.set_xlabel('Radius (arcsec)')
  #          ax3.set_xlim(0,DHI)
        
            gs01 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0[1])
            gx1 = plt.Subplot(fig, gs01[0:2, :])
            fig.add_subplot(gx1)
            gx2 = plt.Subplot(fig, gs01[2, :])
            fig.add_subplot(gx2)
        
            gx1.plot(RADI_bind,residual)
            gx1.fill_between(RADI_bind, residual-iqr, residual+iqr,alpha=0.4)
            gx1.fill_between(RADI_bind, residual, 0 ,where= residual>0,alpha=0.4,color='green',interpolate=True)
            gx1.fill_between(RADI_bind, residual, 0 ,where= residual<0,alpha=0.4,color='red',interpolate=True)
            gx1.axvline(R_HI5)
   #         gx1.set_xlim(0,DHI)
        
            gx1.set_title('"Bias"='+str(Total_Bias))
        
            gx2.plot(RADI_bind,residual_int)
    #        gx2.set_xlim(0,DHI)
            gx2.axvline(R_HI5)
        
            gx2.fill_between(RADI_bind,residual_int, 0 \
                    ,where= residual_int>0,alpha=0.4,color='green',interpolate=True)
            gx2.fill_between(RADI_bind,residual_int, 0 \
                    ,where= residual_int<0,alpha=0.4,color='red',interpolate=True)
        
            plt.tight_layout()
            plt.savefig(namestr+'plot.png',bbox_inches='tight')
            plt.close()


R = alldelta_r
Y = alldelta

med = np.array([])
dmed = np.array([])
medR = np.array([])



fig=plt.figure(figsize=(12,8))
xy = np.array(xy)
for i in range(0,len(xy)):
    if (xy[i,0][0]) < 0 or (xy[i,0][1]) < 0:
        plt.scatter((xy[i,0][0]),(xy[i,0][1]),color='red')
    else:
        plt.scatter((xy[i,0][0]),(xy[i,0][1]),color='blue')
#plt.xlim(0.95*np.min(xy[:,0][0]),np.max(xy[:,0][1])*1.05)
#plt.ylim(1.05*np.min(xy[:,0][0]),np.max(xy[:,0][1])*1.05)
plt.xscale('symlog')
plt.yscale('symlog')
plt.ylabel('"Bias"',fontsize=21.5)
plt.xlabel('Chi2',fontsize=21.5)
plt.savefig('test1.png',bbox_inches='tight')
plt.close()

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
mpl.rcParams['axes.ymargin'] = 0.75
mpl.rcParams['axes.xmargin'] = 0.75
mpl.rcParams['xtick.major.pad'] = 15
mpl.rcParams['ytick.major.pad'] = 8
mpl.rcParams['font.monospace'] = 'Courier'
mpl.rcParams['legend.scatterpoints'] = '3'
mpl.rcParams['mathtext.default'] = 'regular'
mpl.rcParams['mathtext.it'] = 'serif:regular'
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

top,bot = np.percentile(np.unique(R), [75 ,25])
dxr=2 * (top-bot) / (len(R))**(1./3.)

fig=plt.figure(figsize=(12,8))
plt.xlabel('R/R_HI',fontsize=21.5)
#plt.ylabel('Vrot - Polyex')
plt.ylabel('$\Delta V_{rot}$ [km s$^{-1}$]',fontsize=21.5)
iqr = np.array([])
frange = np.arange(0,np.max(R),dxr)
print(frange)

for i,r in enumerate(frange):
    temp_med = np.median(Y[(R > r) & (R < r+dxr)])
    med = np.append(med,temp_med)
    medR = np.append(medR,r + dxr/2.)
    #plt.axhline(temp_med,xmin=r,xmax=r+dxr,color='blue',lw = 2)
    if len(Y[(R > r) & (R < r+dxr)]) > 0:
        top,bot = np.percentile(Y[(R > r) & (R < r+dxr)], [75 ,25])
        iqr = top-bot
        dmed = np.append(dmed,iqr)
    else :
        dmed = np.append(dmed,0)
        

        #plt.axhspan(ymin=temp_med-iqr, ymax=temp_med+iqr, xmin=r, xmax=r+dxr, color='red',alpha=0.45,zorder=1,edgecolor='None')

plt.plot(medR,med)
plt.errorbar(medR,med,yerr=dmed)
plt.axhline(y=0,color='black',zorder=2,lw=2)
plt.scatter(R,Y,marker='x',alpha=0.45)
plt.xlim(.95*np.min(R),np.max(R)*1.05)
plt.ylim(.95*np.min(Y),np.max(Y)*1.05)
#plt.hlines(y = med_delta,xmin = med_deltaR-0.09,xmax = med_deltaR+0.09,alpha=0.5)

print(medR,med)

plt.savefig('test2.png',bbox_inches='tight')
plt.close()
#plt.show()
