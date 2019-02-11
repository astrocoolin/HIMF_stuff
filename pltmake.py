#!/usr/bin/env python3
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import *
from astropy.io import ascii
from pathlib import Path
from scipy import optimize

# am I plotting the data individually
plt1 = False

# array of all points 
alldelta = np.array([])
alldelta_r = np.array([])

# array of statistical measures
xy = []
xy_f = []

xy_d = []
xy_f_d = []
        
def plyex(v0,rPE,a,R):
    return v0 * (1. - np.exp(-R/rPE)) * (1. + R * a /rPE )

def zeros(diam, halfsolar):
    x = diam
    while (abs(halfsolar(x) - 0.5) > 0.001):
        x = x + 0.001
    return x

def rect(axob,xmin,ymin,dx,dy):
    rob=[]
    rob.append(Rectangle((xmin,ymin),dx,dy))
    coll=PatchCollection(rob, zorder=1,alpha=0.45,color='C0')
    axob.add_collection(coll)

# all galaxies
for mass in (7.5, 8.3, 9.2, 10.5,):
  for ba in (3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,):
      for inc in (10,35,65,80,):
          for ed in ("","_2",):
            namestr = str(mass)+ed+"/ba_"+str(ba)+".mass_"+str(mass)+".inc_"+str(inc)+".0.SN_8.0.noise"
            P1 = Path(namestr+'1/RC.dat')
            if P1.is_file():
                namestr = str(mass)+ed+"/ba_"+str(ba)+".mass_"+str(mass)+".inc_"+str(inc)+".0.SN_8.0.noise"
                saveloc = "ba_"+str(ba)+".mass_"+str(mass)+".inc_"+str(inc)+".0.SN_8.0.noise"
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

                vflat = float(lines[5][19:])
                Rs = float(lines[7][19:])
                dx = float(lines[14][18:])

                infile.close()
                
                # in arcseconds
                rPE = rPE*206265/ (Distance)

                #Finding r where sbr drops < 0.5
                r_prof = ascii.read(fname[0]+"/RC.dat")['col1']
                sbr = ascii.read(fname[0]+"/RC.dat")['col3'] 
                conv=6.0574E5*1.823E18*(2.*np.pi/np.log(256.))
                sbr = sbr/1.24756e+20*(conv)
                #print(DHI)
                DHI = ((DHI/2.) / Distance)*3600.*180./np.pi
                
                def halfsolar(r):
                    sig1 = np.exp(-((r-0.4*DHI)/(np.sqrt(2)*(dx+0.36)*DHI))**2.)
                    sig2 = (np.sqrt(vflat/120.)-1.)*np.exp(-r/Rs)
                    sig = sig1 - sig2
                    R_HI = DHI
                    SBR_R = np.exp(-((R_HI-0.4*DHI)/(np.sqrt(2)*(dx+0.36)*DHI))**2.)-(np.sqrt(vflat/120.)-1.)*np.exp(-R_HI/Rs)

                    return sig / SBR_R
                
                R_HI5 = zeros(DHI,halfsolar)
                #print(DHI,R_HI5)
                
                #R_HI5 = r_prof[np.argmin(abs(sbr - 0.5))]
                
                VROT = np.zeros([50,50])*np.nan
                RADI = np.zeros([50,50])*np.nan
                INCL = np.zeros([50,50])*np.nan
                XPOS = np.zeros([50,50])*np.nan
                YPOS = np.zeros([50,50])*np.nan
                PA   = np.zeros([50,50])*np.nan
                
                instr = 0
                instr_len = 0
                success = 0
                for num,valu in enumerate(fname):
                    existcheck = Path(valu+"/Finalmodel/Finalmodel.def")
                    if existcheck.is_file():
                        success = success+1
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
                            if   'INCL' in line and "+" in line and "ERR" not in line and "_2" not in line:
                                instr = np.array(np.float32(line.split()[1:]))
                                INCL[num,0:len(instr)]=instr
                            if   'XPOS' in line and "+" in line and "ERR" not in line and "_2" not in line:
                                instr = np.array(np.float32(line.split()[1:]))
                                XPOS[num,0:len(instr)]=instr
                            if   'YPOS' in line and "+" in line and "ERR" not in line and "_2" not in line:
                                instr = np.array(np.float32(line.split()[1:]))
                                YPOS[num,0:len(instr)]=instr
                            if   'PA' in line and "+" in line and "ERR" not in line and "_2" not in line:
                                instr = np.array(np.float32(line.split()[1:]))
                                PA[num,0:len(instr)]=instr
                


                # smooth radii for analytical ployex. This is an array
                if len(RADI[np.isfinite(RADI)]) == 0: continue
                #print(RADI)
                #print(np.max(RADI[np.isfinite(RADI)]),saveloc)
                RADI_2 = np.arange(0,np.max(RADI[np.isfinite(RADI)])+0.1,0.1)
                polyex = v0 * (1. - np.exp(-RADI_2/rPE)) * (1. + RADI_2 * a /rPE )

                # This does the data for the plot of all rotation curves
                V = VROT[np.isfinite(RADI)]
                R = RADI[np.isfinite(RADI)]
                I = INCL[np.isfinite(RADI)]
                VROT_f = VROT * np.sin(INCL*np.pi/180.)/np.sin(inc*np.pi/180.)
                V2 = V*np.sin(I*np.pi/180.)/np.sin(inc*np.pi/180.)
                
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
                # Find iqr, and median within each of the bins
                for i,r in enumerate(frange):
                    temp_med = np.median(V[(R > r) & (R < r+dxr)])
                    med = np.append(med,temp_med)
                    res_med = np.median(residuals[(res_radi > r) & (res_radi < r+dxr)])
                    if not np.isfinite(res_med): print(residuals[(res_radi > r) & (res_radi < r+dxr)])
                    res = np.append(res,res_med)
                    if len(V[(R > r) & (R < r+dxr)]) > 0:
                        top,bot = np.percentile(V[(R > r) & (R < r+dxr)], [75 ,25])
                        iqr = top-bot
                        dmed = np.append(dmed,iqr)
                    else :
                        dmed = np.append(dmed,0)

                # FOR BIAS AND CHI2
                Bias = np.array([])
                Chi2 = np.array([])

                Bias_f = np.array([])
                Chi2_f = np.array([])

                for i in range(0,len(RADI)):
                    RADI_temp = RADI[i][np.isfinite(RADI[i])]
                    if len(RADI_temp[RADI_temp < R_HI5]) <= 0: continue
                    C = (1./50.) / len(RADI_temp[RADI_temp < R_HI5])
                    polyex_discrete = v0 * (1. - np.exp(-RADI_temp/rPE)) * (1. + RADI_temp * a /rPE )
                    B = np.array([])
                    sig = np.array([])

                    B_f = np.array([])
                    sig_f = np.array([])

                    for j,r in enumerate(RADI_temp):
                        if r < R_HI5:
                            B = np.append(B,100.*(VROT[i][j] - polyex_discrete[j])/polyex_discrete[j])
                            B_f = np.append(B_f,100.*(VROT_f[i][j] - polyex_discrete[j])/polyex_discrete[j])

                            top,bot = np.percentile(VROT[(RADI==r) 
                                & np.isfinite(VROT)], [75 ,25],axis=0)
                            sig = np.append(sig,top-bot)

                            top,bot = np.percentile(VROT_f[(RADI==r) 
                                & np.isfinite(VROT_f)], [75 ,25],axis=0)
                            sig_f = np.append(sig_f,top-bot)

                    sig_f = sig_f[np.isfinite(B_f)]
                    sig = sig[np.isfinite(B)]

                    B = B[np.isfinite(B)]
                    B_f = B_f[np.isfinite(B_f)]
                    
                    Bias = np.append(Bias,np.sum((C*B/sig)[np.isfinite(C*B/sig)]))
                    Chi2 = np.append(Chi2,np.sum((C*B**2./sig**2.)[np.isfinite(C*B**2./sig**2.)]))
                    
                    Bias_f = np.append(Bias_f,np.sum((C*B_f/sig_f)[np.isfinite(C*B_f/sig_f)]))
                    Chi2_f = np.append(Chi2_f,np.sum((C*B_f**2./sig_f**2.)[np.isfinite(C*B_f**2./sig_f**2.)]))

                Total_Bias = (np.sum(Bias))
                Total_Chi = np.log10(np.sum(Chi2))
                
                Total_Bias_f = (np.sum(Bias_f))
                Total_Chi_f = np.log10(np.sum(Chi2_f))
                mass_=2
                ba_=3
                inc_=4
                succ_=5
                INCL_ =6
                XPOS_=7
                YPOS_=8
                PA_=9

                INCL = INCL[np.isfinite(INCL)]
                XPOS = XPOS[np.isfinite(XPOS)]
                YPOS = YPOS[np.isfinite(YPOS)]
                PA = PA[np.isfinite(PA)]
                if abs(np.median(INCL) - inc) > 40:
                    print(namestr)

                xy.append([[Total_Chi,Total_Bias,mass,ba,inc,100.*(1. - success/50.),np.mean(INCL),np.mean(XPOS),np.mean(YPOS),np.mean(PA)]])

                xy_f.append([[Total_Chi_f,Total_Bias_f,mass,ba,inc,100.*(1. - success/50.),np.mean(INCL),np.mean(XPOS),np.mean(YPOS),np.mean(PA)]])

                xy_d.append([[Total_Chi,Total_Bias,mass,ba,inc,100.*(1. - success/50.),np.median(INCL),np.median(XPOS),np.median(YPOS),np.median(PA)]])

                xy_f_d.append([[Total_Chi_f,Total_Bias_f,mass,ba,inc,100.*(1. - success/50.),np.median(INCL),np.median(XPOS),np.median(YPOS),np.median(PA)]])

                if plt1:
                    fig = plt.figure(figsize=(12,8))
                    ax1=fig.add_subplot(2, 2, 1)
                    ax2=fig.add_subplot(2, 2, 2)
                    ax3=fig.add_subplot(2, 2, 3)
                    ax4=fig.add_subplot(2, 2, 4)

                    ax1.scatter(R,V, lw = 2.5, label = 'Recovered Curve',
                            marker='x',alpha=0.25,color='C0',zorder=2)
                    ax2.scatter(res_radi,residuals,color='C0',alpha=0.25,zorder=2)
                    textstr = '\n'.join((
                        r'$\mathrm{Bias=}%.2f$' % (Total_Bias),
                        r'$\chi^2=%.2f$' % (Total_Chi)))
                    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                    ax2.text(0.70, 0.95, textstr, transform=ax2.transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)

                    for i,r in enumerate(frange):
                        rect(ax1,r,med[i]-dmed[i],dxr,2*dmed[i])
                        rect(ax2,r,res[i]-dmed[i],dxr,2*dmed[i])

                        ax1.hlines(y=med[i],xmin=r,xmax=r+dxr,alpha=0.45,color='blue',lw=1.5)
                        ax2.hlines(y=res[i],xmin=r,xmax=r+dxr,alpha=0.45,color='blue',lw=1.5)

                    ax2.axhline(y=0,color='red',alpha=0.45,lw=2.5,zorder=3)
                    ax1.plot(RADI_2,polyex, lw = 2.5 , label = 'Input Curve',color='red',zorder=3)
                    ax1.set_ylabel('Vrot (km/s)')
                    ax1.set_xlabel('Radius (arcsec)')
                    ax2.set_ylabel('Delta Vrot (km/s)')
                    ax2.set_xlabel('Radius (arcsec)')


                    ax1.legend()

                    ax3.scatter(R,V2, lw = 2.5, label = 'Adjusted Curve',
                            marker='x',alpha=0.25,color='C0',zorder=2)
                    ax3.plot(RADI_2,polyex, lw = 2.5 , label = 'Input Curve',color='red',zorder=3)
                    ax3.set_ylabel('Vrot (km/s)')
                    ax3.set_xlabel('Radius (arcsec)')
                    ax1.axvline(R_HI5)
                    ax2.axvline(R_HI5)
                    ax3.axvline(R_HI5)
                    ax4.axvline(R_HI5)

                    ax4.scatter(R,np.sin(I*np.pi/180.),lw = 2.5,
                            marker='x',alpha=0.25,color='C0',zorder=2)
                    ax4.set_ylabel('Sin(i)')
                    ax4.set_xlabel('Radius (arcsec)')
                    ax4.axhline(np.sin(inc*np.pi/180.))
                    
                    plt.tight_layout()
                    plt.savefig(saveloc+'plot.png',bbox_inches='tight')
                    plt.close()

                alldelta_r = np.append(alldelta_r,RADI_med_stats/DHI)
                alldelta = np.append(alldelta,med_residuals)


R = alldelta_r
Y = alldelta

med = np.array([])
dmed = np.array([])
medR = np.array([])

fig, axes =plt.subplots(2,2,sharey=True,sharex=True, figsize=(12,8))
xy = np.array(xy)

CB1 = axes[0,0].scatter((xy[:,0,1]),(xy[:,0,0]),c=xy[:,0,mass_],cmap='coolwarm',marker='x')
cbar = fig.colorbar(CB1,ax=axes[0,0])
x1,x2 = axes[0,0].get_xlim()
y1,y2 = axes[0,0].get_ylim()
axes[0,0].set_xlim(x1,x2)
axes[0,0].set_ylim(y1,y2)
axes[0,0].set_xlabel('"Bias"',fontsize=21.5)
axes[0,0].set_ylabel('Chi2',fontsize=21.5)
axes[0,0].xaxis.set_tick_params(which='both', labelbottom=True)

CB2 = axes[0,1].scatter((xy[:,0,1]),(xy[:,0,0]),c=xy[:,0,ba_],cmap='coolwarm',marker='x')
cbar = fig.colorbar(CB2,ax=axes[0,1])
axes[0,1].set_xlabel('"Bias"',fontsize=21.5)
axes[0,1].set_ylabel('Chi2',fontsize=21.5)
axes[0,1].xaxis.set_tick_params(which='both', labelbottom=True)

CB3 = axes[1,0].scatter((xy[:,0,1]),(xy[:,0,0]),c=xy[:,0,inc_],cmap='coolwarm',marker='x')
cbar = fig.colorbar(CB3,ax=axes[1,0])
axes[1,0].set_xlabel('"Bias"',fontsize=21.5)
axes[1,0].set_ylabel('Chi2',fontsize=21.5)
axes[1,0].xaxis.set_tick_params(which='both', labelbottom=True)

CB4 = axes[1,1].scatter((xy[:,0,1]),(xy[:,0,0]),c=xy[:,0,succ_],cmap='coolwarm',marker='x')
cbar = fig.colorbar(CB4,ax=axes[1,1])
axes[1,1].set_xlabel('"Bias"',fontsize=21.5)
axes[1,1].set_ylabel('Chi2',fontsize=21.5)
axes[1,1].xaxis.set_tick_params(which='both', labelbottom=True)

plt.xscale('symlog')
#plt.yscale('log')
plt.xlabel('"Bias"',fontsize=21.5)
plt.ylabel('Chi2',fontsize=21.5)
plt.savefig('ChiBias.png',bbox_inches='tight')
plt.close()

fig, axes =plt.subplots(2,2,sharey=True,sharex=True, figsize=(12,8))
xy_f = np.array(xy_f)

CB1 = axes[0,0].scatter((xy_f[:,0,1]),(xy_f[:,0,0]),c=xy_f[:,0,mass_],cmap='coolwarm',marker='x')
cbar = fig.colorbar(CB1,ax=axes[0,0])
axes[0,0].set_xlim(x1,x2)
axes[0,0].set_ylim(y1,y2)
axes[0,0].set_xlabel('"Bias"',fontsize=21.5)
axes[0,0].set_ylabel('Chi2',fontsize=21.5)
axes[0,0].xaxis.set_tick_params(which='both', labelbottom=True)

CB2 = axes[0,1].scatter((xy_f[:,0,1]),(xy_f[:,0,0]),c=xy_f[:,0,ba_],cmap='coolwarm',marker='x')
cbar = fig.colorbar(CB2,ax=axes[0,1])
axes[0,1].set_xlabel('"Bias"',fontsize=21.5)
axes[0,1].set_ylabel('Chi2',fontsize=21.5)
axes[0,1].xaxis.set_tick_params(which='both', labelbottom=True)

CB3 = axes[1,0].scatter((xy_f[:,0,1]),(xy_f[:,0,0]),c=xy_f[:,0,inc_],cmap='coolwarm',marker='x')
cbar = fig.colorbar(CB3,ax=axes[1,0])
axes[1,0].set_xlabel('"Bias"',fontsize=21.5)
axes[1,0].set_ylabel('Chi2',fontsize=21.5)
axes[1,0].xaxis.set_tick_params(which='both', labelbottom=True)

CB4 = axes[1,1].scatter((xy_f[:,0,1]),(xy_f[:,0,0]),c=xy_f[:,0,succ_],cmap='coolwarm',marker='x')
cbar = fig.colorbar(CB4,ax=axes[1,1])
axes[1,1].set_xlabel('"Bias"',fontsize=21.5)
axes[1,1].set_ylabel('Chi2',fontsize=21.5)
axes[1,1].xaxis.set_tick_params(which='both', labelbottom=True)

plt.xscale('symlog')
#plt.yscale('log10')
#axes[1,1].set_yscale('log')
plt.savefig('ChiBias_f.png',bbox_inches='tight')
plt.close()

fig, ax =plt.subplots(figsize=(12,8))
xy_f = np.array(xy_f)
print('plotting delta')
for i,y in enumerate(xy_f[:,0,1]):
    dx = xy_f[i,0,1] - xy[i,0,1]
    dy = xy_f[i,0,0] - xy[i,0,0]
    ax.arrow(xy[i,0,1],xy[i,0,0],dx=dx,dy=dy,head_width=0.15,facecolor='red')
    ax.scatter((xy[:,0,1]),(xy[:,0,0]),color='blue',marker='x')

ax.set_xlim(x1,x2)
ax.set_ylim(y1,y2)
ax.set_xlabel('"Bias"',fontsize=21.5)
ax.set_ylabel('Chi2',fontsize=21.5)
plt.savefig('ChiBias_delta.png',bbox_inches='tight')
plt.close()

fig, ax =plt.subplots(figsize=(12,8))
xy_f = np.array(xy_f)
print('plotting delta')
for i,y in enumerate(xy[:,0,1]):
    if (xy_f[i,0,1] - xy[i,0,1] > 0 and xy[i,0,1] > 0) or \
            (xy_f[i,0,1] - xy[i,0,1] < 0 and xy[i,0,1] < 0):
        dx = xy_f[i,0,1] - xy[i,0,1]
        dy = xy_f[i,0,0] - xy[i,0,0]
        ax.arrow(xy[i,0,1],xy[i,0,0],dx=dx,dy=dy,head_width=0.15,facecolor='red')
        ax.scatter((xy[:,0,1]),(xy[:,0,0]),color='blue',marker='x')

ax.set_xlim(x1,x2)
ax.set_ylim(y1,y2)
ax.set_xlabel('"Bias"',fontsize=21.5)
ax.set_ylabel('Chi2',fontsize=21.5)
plt.savefig('ChiBias_delta_g.png',bbox_inches='tight')
plt.close()

np.save('xy.bin',xy)
np.save('xy_fixed.bin',xy_f)

xy_f_d = np.array(xy_f_d)
xy_d = np.array(xy_d)

fig, ((ax1,ax2),(ax3,ax4)) =plt.subplots(2,2,figsize=(12,8))

ax1.scatter(xy_f[:,0,inc_],xy_f[:,0,INCL_]-xy_f[:,0,inc_])
ax1.set_ylabel('$\Delta$INCL')
ax1.set_xlabel('INCL')

ax2.scatter(xy_f[:,0,inc_],xy_f[:,0,PA_]-45.)
ax2.set_ylabel('$\Delta$PA')
ax2.set_xlabel('INCL')

ax3.scatter(xy_f[:,0,ba_],xy_f[:,0,INCL_]-xy_f[:,0,inc_])
ax3.set_ylabel('$\Delta$INCL')
ax3.set_xlabel('Beams Across')

ax4.scatter(xy_f[:,0,ba_],xy_f[:,0,INCL_]-xy_f[:,0,inc_])
ax4.set_ylabel('$\Delta$INCL')
ax4.set_xlabel('Beams Across')

#plt.show()
#plt.close()

fig2, ((ax5,ax6),(ax7,ax8)) =plt.subplots(2,2,figsize=(12,8))

ax5.scatter(xy_f_d[:,0,inc_],xy_f_d[:,0,INCL_]-xy_f_d[:,0,inc_])
ax5.set_ylabel('$\Delta$INCL')
ax5.set_xlabel('INCL')

ax6.scatter(xy_f_d[:,0,inc_],xy_f_d[:,0,PA_]-45.)
ax6.set_ylabel('$\Delta$PA')
ax6.set_xlabel('INCL')

ax7.scatter(xy_f_d[:,0,ba_],xy_f_d[:,0,INCL_]-xy_f_d[:,0,inc_])
ax7.set_ylabel('$\Delta$INCL')
ax7.set_xlabel('Beams Across')

ax8.scatter(xy_f_d[:,0,ba_],xy_f_d[:,0,INCL_]-xy_f_d[:,0,inc_])
ax8.set_ylabel('$\Delta$INCL')
ax8.set_xlabel('Beams Across')

plt.show()
plt.close()

if False:
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
    
    for i,r in enumerate(frange):
        temp_med = np.median(Y[(R > r) & (R < r+dxr)])
        med = np.append(med,temp_med)
        medR = np.append(medR,r + dxr/2.)
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
    
    plt.savefig('TotalOverlay.png',bbox_inches='tight')
    plt.close()
