#!/usr/bin/env python

import argparse
import sys, os
import subprocess as sp
import numpy as np
import netCDF4 as nc

ROOT=os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT,'oasisgrids','esmgrids'))

from esmgrids.mom_grid import MomGrid
from esmgrids.woa_grid import WoaGrid
from helpers import setup_test_input_dir, setup_test_output_dir
from helpers import calc_regridding_err

EARTH_RADIUS = 6370997.0
EARTH_AREA = 4*np.pi*EARTH_RADIUS**2

def get_grid(filename):

    try:
        grid = MomGrid.fromfile(filename)
        return grid
    except KeyError as e:
         print("Not a mom grid: ",type(e),str(e))

    try:
        grid = WoaGrid(filename)
        return grid
    except KeyError as e:
         print("Not a WOA grid: ",type(e),str(e))

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert grid file to GRIDSPEC format")
    parser.add_argument("-v","--verbose", help="Verbose output", action='store_true')
    parser.add_argument("inputs", help="grid files", nargs='+')
    args = parser.parse_args()

    verbose=args.verbose

    # Loop over all the inputs from the command line. 
    for input in args.inputs:

        if verbose: print("Processing {}".format(input))

        suffix = '.nc'

        if input.endswith(suffix):
            output = input[:-len(suffix)] + '_gridspec.nc'
        else:
            output = input + '_gridspec.nc'

        if verbose:
            print("Input file: {}".format(input))
            print("GRIDSPEC output file: {}".format(output))

        # grid = MomGrid.fromfile(input)
        grid = get_grid(input)
        grid.write_gridspec(output)

