#!/usr/bin/env python3
import numpy as np
import os
from science import first_beam
from Input_output import  deffile, rotfile, emptyfits, rothead
from pathlib import Path

from relations import *
#########################################################
# Setting constants: KPC, MPC scales, central SB
# Delta is the increment range in arcseconds
#########################################################
pi = np.pi
KPC =1.0E3
MPC =1.0E6
delta = 5.
#########################################################
# Ranges of all of the parameters to be varied
# # beams, inclination, magnitude, S/N ratio
#########################################################

beam_list  = [32.]
inc_list   = [80.]
mass_list  = [9.5]
sn_list    = [16.]

beam_list  = [3.,4.,5.,6.,7.,8.,12.,16.,18.]
inc_list   = [20.,40.,60.,80.,90.]
mass_list  = np.arange(5.,10.6,0.5)#[5.,5.5,6.,6.5,7.,7.5,8.,8.5,9.,9.5,10.,10.5]
sn_list    = [16.,8.,4.,2.]

catalog = 'sample_10.txt'
file = open(catalog,'w')

file.write("mass "+"RHI "+ "Mag "+"Alpha "+"Vmax "+"Vflat "+"Mstar "+"slope "+" rd"+" rPE"+"\n")
print('beams:',beam_list)
print('mass:',mass_list)
print('inc:',inc_list)
print('sn:',sn_list)
#########################################################
# Main loop
#########################################################
for inc in inc_list:
    for i, mass in enumerate(mass_list):
        for snr in sn_list:
            for beams in beam_list:
                #print('inc:',[inc])
                #print('mass:',[mass])
                #print('snr:',[snr])
                #print('beams:',[beams])
                #########################################################
                # Names of files
                #########################################################
                defname ="cube_input.def"
                inset   ='empty.fits'
                outset  ='Cube_base.fits'
                outname ='Cube_ba_'+str(beams)+".mass_"+str(mass)+".inc_"+\
                    str(inc)+".SN_"+str(snr)+'.fits'
                fname ="ba_"+str(beams)+".mass_"+str(mass)+".inc_"+\
                    str(inc)+".SN_"+str(snr)
                #########################################################
                # Scaling everything in terms of arcseconds instead of
                # in terms of kilparsecs; divide by distance
                #########################################################
                radi,sbr,vrot,condisp,z,MHI,DHI,Mag,dist,alpha,vflat,Mstar,slope,rd,rPE = \
                        setup_relations(mass,beams,delta)
                #sbr = sbr * 0.5E-2
                #########################################################
                #print('------------------')
                #print('dist [kpc]:',        round(dist,2))
                #print('sdisp[km/s]:',     round(condisp,2))
                #print('------------------')
                #########################################################
                # Set the radii, rotation curve, surface brightness
                # profile
                #print(len(radi),min(radi),max(radi))
                radi = radi / (dist) * 3600. * (180./np.pi)
                #def rotfile(radi,vrot,sbr,END):
                ########################################################
                #radi=np.arange(0.,outside+delta,delta)
                #vrot=make_vrot(radi,mag,hr)
                #sbr=make_sbr(radi,IO,hr,edge)
                #########################################################
                filecheck = Path('empty.fits')
                if filecheck.is_file(): os.system("rm empty.fits")
                filecheck = Path('Logfile.log')
                if filecheck.is_file(): os.system("rm Logfile.log")
                filecheck = Path('outname')
                if filecheck.is_file(): os.system("rm "+outname)
                filecheck = Path('defname')
                if filecheck.is_file(): os.system("rm "+defname)
                #########################################################
                # Make a file containing the rotation curve and SBP
                # Make an input file for TiRiFiC
                # Make an empty fits file
                #########################################################
                rotfile(radi,vrot,sbr,z,len(radi))
                deffile(outset,inset,defname,radi,vrot,sbr,inc,\
                        len(radi),condisp,z)
                #########################################################
                # Make new cube, folder for it, clear old files
                #########################################################
                make_cube = True
                if (make_cube):
                    emptyfits(inset)
                    os.system("tirific deffile="+defname)
                    print("Cube finished")
                #########################################################
                filecheck = Path(fname)
                if filecheck.is_dir (): os.system("rm -r "+fname)
                os.system("mkdir "+fname)
                #print("Refreshed folder")
                #########################################################
                if (make_cube):
                    first_beam(outset,outname,DHI/2.,beams,snr,inc,mass)
                    os.system("mv "+outname+" "+fname)
                    os.system("rm "+outset)
                    os.system("rm empty.fits Logfile.log")
                #########################################################
                os.system("mv "+defname+" VROT.png SBR.png "+fname)
                os.system("mv "+defname+" "+fname)
                os.system("cp RC.dat "+fname)
                #########################################################
                file.write(str(mass)+" "+str(DHI/2.)+" "+str(Mag)+" "+str(alpha)+" "+str(np.max(vrot))+" "+str(np.max(vflat))+" "+str(Mstar)+" "+str(slope)+" "+str(rd)+" "+str(rPE)+"\n")
    os.system("rm RC.dat")
