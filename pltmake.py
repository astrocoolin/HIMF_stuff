#!/usr/bin/env python3
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import ascii
from pathlib import Path

# am I plotting the data individually
plt1 = True

# array of all points 
alldelta = np.array([])
alldelta_r = np.array([])

# array of statistical measures
xy = []
        
def plyex(v0,rPE,a,R):
    return v0 * (1. - np.exp(-R/rPE)) * (1. + R * a /rPE )

# all galaxies
for mass in (7.5,):
  for ba in (4,5,6,8,):
      for inc in (10,35,65,80,):
        namestr = "ba_"+str(ba)+".0.mass_"+str(mass)+".inc_"+str(inc)+".0.SN_8.0.noise"
        num = np.arange(1,51)
       
        #name all the variables
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
        
        
        # smooth radii for analytical ployex. This is an array
        RADI_2 = np.arange(0,np.max(RADI[np.isfinite(RADI)])+0.1,0.1)
        polyex = v0 * (1. - np.exp(-RADI_2/rPE)) * (1. + RADI_2 * a /rPE )

        # This does the data for the plot of all rotation curves
        V = VROT[np.isfinite(RADI)]
        R = RADI[np.isfinite(RADI)] 
        
        # FOR STATISTICS
        # unique radii, finite values only
        RADI_med = np.unique(RADI)
        RADI_med_stats = RADI_med[np.isfinite(RADI_med)]
        
        VROT_med_stats  = np.array([])
        iqr_stats = np.array([])
        devarray_stats = np.array([])
        residuals = np.array([])
        med_residuals = np.array([])
        res_radi = np.array([])

        for i,radius in enumerate(RADI_med_stats):
            # All velocities at this radius
            VROT_temp = VROT[(RADI==radius) & (np.isfinite(VROT))]
            # median rotational velocity at this radius
            VROT_med_stats= np.append(VROT_med_stats,
                np.median(VROT_temp))

            # All residuals at this radius
            res_temp = VROT_temp - plyex(v0,rPE,a,radius)
            res_radi_temp = radius*np.ones_like(res_temp)
            # All residuals at all radii
            residuals = np.append(residuals,res_temp)
            res_radi  = np.append(res_radi,res_radi_temp)
            # Median residual at this radius
            med_residuals  = np.append(med_residuals,
                    np.median(VROT_temp) - plyex(v0,rPE,a,radius))

            if VROT_temp.size > 4  or (i < instr_len):
                top,bot = np.percentile(VROT_temp, [75 ,25],axis=0)

                iqr_stats = np.append(iqr_stats,top-bot)
                devarray_stats = np.append(devarray_stats,np.std(VROT_temp))
            else:
                iqr_stats = np.append(iqr_stats,0)
                devarray_stats = np.append(devarray_stats,0)
        
        
        # FOR PLOTTING
        # Find optimal spacing for the bins 
        top,bot = np.percentile(np.unique(R), [75 ,25])
        dxr=2 * (top-bot) / (len(R))**(1./3.)
        frange = np.arange(0,np.max(R),dxr)
        dmed = np.array([])
        med = np.array([])
        res = np.array([])
        res_med = np.array([])
        # Find iqr, and median within each of the bins
        for i,r in enumerate(frange):
            temp_med = np.median(V[(R > r) & (R < r+dxr)])
            med = np.append(med,temp_med)
            res = np.append(res,temp_med - plyex(v0,rPE,a,r+dxr/2.))
            if len(V[(R > r) & (R < r+dxr)]) > 0:
                top,bot = np.percentile(V[(R > r) & (R < r+dxr)], [75 ,25])
                iqr = top-bot
                dmed = np.append(dmed,iqr)
            else :
                dmed = np.append(dmed,0)

        # FOR BIAS AND CHI2
        Bias = np.array([])
        Chi2 = np.array([])
        for i in range(0,len(RADI)):
            RADI_temp = RADI[i][np.isfinite(RADI[i])]
            if len(RADI_temp[RADI_temp < R_HI5]) <= 0: continue
            C = (1./50.) / len(RADI_temp[RADI_temp < R_HI5])
            polyex_discrete = v0 * (1. - np.exp(-RADI_temp/rPE)) * (1. + RADI_temp * a /rPE )
            B = np.array([])
            sig = np.array([])

            for j,r in enumerate(RADI_temp):
                if r < R_HI5:
                    B = np.append(B,VROT[i][j] - polyex_discrete[j])
                    top,bot = np.percentile(VROT[(RADI==r) & np.isfinite(VROT)], [75 ,25],axis=0)
                    sig = np.append(sig,top-bot)
            sig = sig[np.isfinite(B)]
            B = B[np.isfinite(B)]
            
            Bias = np.append(Bias,np.sum((C*B/sig)[np.isfinite(C*B/sig)]))
            Chi2 = np.append(Chi2,np.sum((C*B**2./sig**2.)[np.isfinite(C*B**2./sig**2.)]))

        Total_Bias = np.sum(Bias)
        Total_Chi = np.sum(Chi2)
       
        xy.append([[Total_Chi,Total_Bias]])


        fig = plt.figure(figsize=(12,4))
        ax1=fig.add_subplot(1, 2, 1)
        ax2=fig.add_subplot(1, 2, 2)

        ax1.scatter(RADI,VROT, lw = 2.5, label = 'Recovered Curve',marker='x',alpha=0.25,color='blue')
        left, right = ax1.get_xlim()
        ax1.set_xlim(left,right)
        for i,r in enumerate(frange):
            print(i,r,med[i],namestr)
            dx_ax= dxr / (right - left)
            r_ax = r / (right - left)
            ax1.axhspan(ymin=med[i]-dmed[i],ymax=med[i]+dmed[i],xmin=r_ax,xmax=r_ax+dx_ax,color='blue',alpha=0.45)
            ax1.axhline(y=med[i],xmin=r_ax,xmax=r_ax+dx_ax,alpha=0.45)
        
        ax2.scatter(res_radi,residuals,color='blue',alpha=0.25)
        ax2.plot(frange+dxr/2.,res,color='blue')
        ax2.fill_between(frange+dxr/2.,res+dmed,res-dmed,color='blue',alpha=0.4)
        ax1.plot(RADI_2,polyex, lw = 2.5 , label = 'Input Curve',color='red')
        ax1.set_ylabel('Vrot (km/s)')
        ax1.set_xlabel('Radius (arcsec)')
        ax1.axvline(R_HI5)
        ax1.legend()
        
        
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
