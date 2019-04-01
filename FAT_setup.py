#!/usr/bin/env python3
import numpy as np
import os
import json
import sys


infile = sys.argv[1]

num = sys.argv[2]

with open(infile) as f:
    params = json.load(f)

#########################################################
# Ranges of all of the parameters to be varied
# # beams, inclination, magnitude, S/N ratio
#########################################################
reals  = int(np.float32(params["Noise_Realizations"])+1)
copies = int(np.float32(params["Gal_Realizations"])+1)
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


f2=open("catalog_"+num+".txt",'w')
f2.write("number|distance|directoryname|cubename\n")
k=0
for inc in inc_list:
    for i, mass in enumerate(mass_list):
        for snr in sn_list:
            for beams in beam_list:
                for j in range(1,copies):
                    for noise in range(1,reals):
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
                        f2.write(str(k)+'|'+str(dist)+'|'+fname+'|'+outname+"\n")
                        k = k+1

f2.close()
f3=open("config"+num+".config","w")
f3.write("catalogue=catalog_"+num+".txt\n")
f3.write("maindir="+os.getcwd()+"\n")
f3.write("outputcatalogue="+os.getcwd()+"/out_"+num+".txt\n")
f3.write("outputlog=fitlog_"+num+".txt\n")
f3.write("new_log='y'\n")
f3.write("new_output=y\n")
f3.write("allnew=-1\n")

f4 = open('m'+str(mass)+'_'+num+'.txt','w')
f4.write("#!/bin/bash \n")
f4.write("srun --ntasks=1 echo -e \".r /home/colin/Storage/PhD/Codes/FAT/FAT.pro\\nFAT, configuration_file='config"+str(num)+".config', Support='/home/colin/Storage/PhD/Codes/FAT/Support/'\" | gdl \n")
