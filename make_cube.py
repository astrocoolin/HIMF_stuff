#!/usr/bin/env python3
import numpy as np
import os
from pathlib import Path
import json

from science import first_beam, second_beam
from Input_output import  deffile, rotfile, emptyfits, rothead
from relations import *

#########################################################
# Setting constants: KPC, MPC scales, central SB
# Delta is the increment range in arcseconds
#########################################################
pi = np.pi
KPC =1.0E3
MPC =1.0E6

with open('config.json') as f:
    params = json.load(f)

make_cube   = params["Make_Cube"]
make_output = params["Make_Output"]
make_folder = params["Make_Folder"]

if make_folder == False:
    print("No output files being created.")
    make_output = False
    make_cube = False

#########################################################
# Ranges of all of the parameters to be varied
# # beams, inclination, magnitude, S/N ratio
#########################################################

delta     = np.float32(params["delta"])
inc_list  = np.float32(params["incs"])
mass_list = np.float32(params["masses"])
beam_list = np.float32(params["beams"])
sn_list   = np.float32(params["sn"])

iters = params["Ranges"]
if 'masses' in iters:
    mass_list = np.round(np.arange(mass_list[0],mass_list[1],mass_list[2]),5)
if 'incs' in iters:
    inc_list = np.round(np.arange(inc_list[0],inc_list[1],inc_list[2]),5)
if 'beams' in iters:
    beam_list = np.round(np.arange(beam_list[0],beam_list[1],beam_list[2]),5)
if 'sn' in iters:
    sn_list = np.round(np.arange(sn_list[0],sn_list[1],sn_list[2]),5)
    
#########################################################
print('Parameter Range:\n')
print('beams:',beam_list)
print('mass:',mass_list)
print('inc:',inc_list)
print('snr:',sn_list)
#########################################################
catalog = 'sample.txt'
file = open(catalog,'w')
file.write("mass "+"RHI "+ "Mag "+"Alpha "+"Vmax "+\
        "Vflat "+"Mstar "+"slope "+" rd"+" rPE"+"\n")
#########################################################
# Main loop
#########################################################
for inc in inc_list:
    for beams in beam_list:
        for snr in sn_list:
            for i, mass in enumerate(mass_list):
                print('\nThis Galaxy:\n')
                print('beams:',[beams])
                print('inc:',[inc])
                print('mass:',[mass])
                print('snr:',[snr])
                ######################################################################
                # Names of files
                ######################################################################
                defname ="cube_input.def"
                inset   ='empty.fits'
                outset  ='Cube_base.fits'
                outname ='Cube_ba_'+str(beams)+".mass_"+str(mass)+".inc_"+\
                        str(inc)+".SN_"+str(snr)+'.fits'
                fname ="ba_"+str(beams)+".mass_"+str(mass)+".inc_"+\
                        str(inc)+".SN_"+str(snr)
                ######################################################################
                # Use scaling relations to set up the galaxy
                ######################################################################
                radi,sbr,vrot,condisp,z,MHI,DHI,Mag,dist,alpha,vflat,\
                        Mstar,slope,rd,rPE,cflux,END = \
                        setup_relations(mass,beams,delta,make_output)
                ######################################################################
                # Make a file containing the rotation curve and SBP
                # Make an input file for TiRiFiC, make an empty FITS file
                ######################################################################
                if (make_output):
                    filecheck = Path('defname')
                    if filecheck.is_file(): os.system("rm "+defname)

                    rotfile(radi,vrot,sbr,z,len(radi))
                    deffile(outset,inset,defname,radi,vrot,sbr,inc,\
                        len(radi),condisp,z,cflux)
                ######################################################################
                # Make new cube, folder for it, clear old files
                ######################################################################
                if (make_cube):
                    filecheck = Path('outname')
                    if filecheck.is_file(): os.system("rm "+outname)
                    filecheck = Path('empty.fits')
                    if filecheck.is_file(): os.system("rm empty.fits")
                    filecheck = Path('Logfile.log')
                    if filecheck.is_file(): os.system("rm Logfile.log")

                    emptyfits(inset)
                    os.system("tirific deffile="+defname)
                    print("Cube finished")
                ######################################################################
                if (make_folder):
                    filecheck = Path(fname)
                    if filecheck.is_dir (): 
                        os.system("rm -r "+fname)
                        print("Refreshed folder")
                    os.system("mkdir "+fname)
                    os.system("mv "+defname+" VROT.png SBR.png SBR_log.png RC.dat "+fname)
                ######################################################################
                if (make_cube):
                    second_beam(outset,outname,END,beams,snr,inc,mass,dist,cflux/2.)
                    os.system("mv "+outname+" "+outset+" "+fname)
                    os.system("rm empty.fits Logfile.log")
                ######################################################################
                #os.system('eog '+fname+'/VROT.png')
                file.write(str(mass)+" "+str(DHI/2.)+" "+str(Mag)+" "+str(alpha)+\
                        " "+str(np.max(vrot))+" "+str(np.max(vflat))+" "+str(Mstar)+\
                        " "+str(slope)+" "+str(rd)+" "+str(rPE)+"\n")
