#!/usr/bin/env python3
import numpy as np
import os
import json

with open('config.json') as f:
    params = json.load(f)

#########################################################
# Ranges of all of the parameters to be varied
# # beams, inclination, magnitude, S/N ratio
#########################################################

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


print("number|distance|directoryname|cubename")
j=0
for inc in inc_list:
    for i, mass in enumerate(mass_list):
        for snr in sn_list:
            for beams in beam_list:
                for noise in range(1,51):
                    outname ='Cube_ba_'+str(beams)+".mass_"+str(mass)+".inc_"+\
                        str(inc)+".SN_"+str(snr)+'.fits'
                    fname ="ba_"+str(beams)+".mass_"+str(mass)+".inc_"+\
                        str(inc)+".SN_"+str(snr)+'.noise'+str(noise)
                    f=open(fname+'/RC.dat','r')
                    for i, line in enumerate(f):
                        if i == 8:
                            dist = float(line[19:])/1000.
                        elif i > 8:
                            break
                    f.close()
                    print(str(j)+'|'+str(dist)+'|'+fname+'|'+outname)
                    j = j+1
