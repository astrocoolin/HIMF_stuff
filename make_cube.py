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
beam_list  = [3.,4.,5.,6.,7.]
inc_list   = [20.,40.,60.,80.,90.]
mass_list  = [7.,8.,9.]
sn_list    = [16.,8.,4.]

beam_list  = [16.]
inc_list   = [20.]
mass_list  = [9.]
sn_list    = [2.]

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
                radi,sbr,vrot,condisp,z,MHI,DHI,Mag,dist = \
                        setup_relations(mass,beams,delta)
                #sbr = sbr * 0.5E-2
                #########################################################
                print('------------------')
                print('dist [kpc]:',        round(dist,2))
                print('sdisp[km/s]:',     round(condisp,2))
                print('------------------')
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
                make_cube = False
                if (make_cube):
                    emptyfits(inset)
                    os.system("tirific deffile="+defname)
                    print("Cube finished")
                #########################################################
                filecheck = Path(fname)
                if filecheck.is_dir (): os.system("rm -r "+fname)
                os.system("mkdir "+fname)
                print("Refreshed folder")
                #########################################################
                if (make_cube):
                    first_beam(outset,outname,DHI/2.,beams,snr,inc,mass)
                    os.system("mv "+outname+" "+fname)
                    os.system("rm "+outset)
                    os.system("rm empty.fits Logfile.log")
                #########################################################
                os.system("mv "+defname+" VROT.png SBR.png "+fname)
                os.system("cp RC.dat "+fname)
                #########################################################
    os.system("rm RC.dat")
