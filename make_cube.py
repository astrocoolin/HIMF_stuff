#!/usr/bin/env python3
import numpy as np
import os
from science import first_beam
from Input_output import  deffile, rotfile, emptyfits
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
# Distances in KPC; scale length is half the edge of
# the disk. The galaxy is extrapolated to twice the
# edge of the disk with the scale length decreased 
# by a factor of 20 outside the edge
#########################################################
base_distance   = 12.8915/16. * MPC #Mpc
#########################################################
# Ranges of all of the parameters to be varied
# # beams, inclination, magnitude, S/N ratio
#########################################################
beam_list  = [3.,4.,5.,6.,7.]
inc_list   = [20.,40.,60.,80.,90.]
mass_list  = [7.,8.,9.]#,10.]
sn_list    = [16.,8.,4.]

beam_list  = [16.]
inc_list   = [20.]
#mass_list  = [9.]
mass_list  = [7.]#,10.]
sn_list    = [16.]
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
                dist    =  beams
                radi,sbr,vrot,condisp,z,MHI,DHI,Mag,dist = \
                        setup_relations(mass,beams,delta)
                #sbr = sbr * 0.5E-2
                #########################################################
                print('------------------')
                print('dist [Mpc]:',        round(dist / MPC,2))
                print('sdisp[km/s]:',     round(condisp,2))
                print('------------------')
                #########################################################
                # Set the radii, rotation curve, surface brightness
                # profile
                #print(len(radi),min(radi),max(radi))
                radi = radi / (dist/1000.) * 3600. * (180./np.pi)
                #########################################################
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
                rotfile(radi,vrot,sbr,len(radi))
                deffile(outset,inset,defname,radi,vrot,sbr,inc,\
                        len(radi),condisp,z)
                emptyfits(inset)
                #########################################################
                # Make new cube, folder for it, clear old files
                #########################################################
                if (True):
                    os.system("tirific deffile="+defname)
                    print("Cube finished")
                    #########################################################
                    filecheck = Path(fname)
                    if filecheck.is_dir (): os.system("rm -r "+fname)
                    os.system("mkdir "+fname)
                    print("Refreshed folder")
                    #########################################################
                    first_beam(outset,outname,DHI/2.,beams,snr,inc,mass)
                    os.system("mv "+outname+" "+defname+" "+fname)
                    os.system("cp RC.txt "+fname)
                    os.system("rm "+outset)
                    os.system("rm empty.fits Logfile.log")
                    #########################################################
    os.system("rm RC.txt")
