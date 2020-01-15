# HIMF_stuff
Spit out information for a given HI mass, Makes the galaxy

This is the core of my PhD project, I took an object-oriented approach to creating galaxies.

First, set the HI Mass as `Mass`, # of beams across the HI Diameter as `Beams`, and the inclination as `Inclination`in `config.py`. `Save_plots` can be set to `True` to save copies of the png files, and FITS can be set to `True` to create fits files using TiRiFiC, if it is installed. 

The galaxy is then created using `./make_galaxy.py`
