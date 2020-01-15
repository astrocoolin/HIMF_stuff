#!/usr/bin/env python3
from NCC import *
from config import *
instance = Galaxy()
instance.reroll(Mass,Beams,Inclination)
instance.make_plots(Save)
#instance.make_fits(1)
