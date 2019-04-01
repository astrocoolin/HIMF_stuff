#!/usr/bin/env python3
import numpy as np
import os
from pathlib import Path
import json

import matplotlib as mpl
mpl.use('Agg')

from science import second_beam
from Input_output import  *
from relations import *
from NCC import *

m1 = np.array([7.5,8.3,9.2,10.5])
m2 = np.array([25,50,100,250])



def automate(same):
    #########################################################
    # Setting constants: KPC, MPC scales, central SB
    # Delta is the increment range in arcseconds
    #########################################################
    pi = np.pi
    KPC =1.0E3
    MPC =1.0E6
    beam_width = 30.
    
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
        
    copies = int(np.float32(params["Gal_Realizations"])+1)
    reals  = int(np.float32(params["Noise_Realizations"])+1)
    #########################################################
    print('Parameter Range:\n')
    print('beams:',beam_list)
    print('mass:',mass_list)
    print('inc:',inc_list)
    print('snr:',sn_list)
    #########################################################
    # Main loop
    #########################################################
    for i, mass in enumerate(mass_list):
        if same:
            current = Galaxy()
            current.reroll(mass,beam_list[0],'True')
            #vrot = int(m2[m1 == mass])
            #current.example(vrot)
        for inc in inc_list:
            for beams in beam_list:
                for snr in sn_list:
                    for j in range(1,copies):
                        print('\nThis Galaxy (copy #'+str(j)+' of '+str(copies-1)+'):\n')
                        print('beams:',[beams])
                        print('inc:',[inc])
                        print('mass:',[mass])
                        print('snr:',[snr])
                        ######################################################################
                        # Use scaling relations to set up the galaxy
                        # Make a file containing the rotation curve and SBP
                        # Make an input file for TiRiFiC, make an empty FITS file
                        ######################################################################
                        if not same:
                            current = Galaxy()
                            current.reroll(mass,beams,beam_width,delta)
                        ######################################################################
                        # Names of files
                        ######################################################################
                        current.defname ="cube_input.def"
                        current.inset   ='empty.fits'
                        current.outset  ='Cube_base.fits'
                        current.outname ='Cube_ba_'+str(beams)+".mass_"+str(round(current.MHI,5))+".inc_"+\
                                str(inc)+".SN_"+str(snr)+'.fits'
                        current.fname ="ba_"+str(beams)+".mass_"+str(round(current.MHI,5))+".inc_"+\
                                str(inc)+".SN_"+str(snr)
                        current.snr=snr
                        current.inc=inc
                        current.calc_dist(beams)
                        current.make_plots(True)
                        current.make_fits(reals)
                        #radi,sbr,vrot,condisp,z,MHI,DHI,Mag,dist,alpha,vflat,\
                        #        Mstar,slope,rd,rPE,cflux,END = \
                        #        setup_relations(mass,beams,beam,delta,make_output)
                        #galaxy.calc_dist()
                        ######################################################################
