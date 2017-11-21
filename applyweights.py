#!/usr/bin/env python

import argparse
import sys, os
import subprocess as sp
import numpy as np
import numba
import netCDF4 as nc

ROOT=os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT,'oasisgrids','esmgrids'))

from esmgrids.mom_grid import MomGrid
from helpers import setup_test_input_dir, setup_test_output_dir
from helpers import calc_regridding_err

EARTH_RADIUS = 6370997.0
EARTH_AREA = 4*np.pi*EARTH_RADIUS**2

def col_idx_largest_lat(lats):
    """
    The col index with the largest lat.
    """
    _, c  = np.unravel_index(np.argmax(lats), lats.shape)

    return c

def create_output_ncfile(input_file, var_name, weights, filename, time_units = "", history=""):

    # Grab information about the size of the source and destination data
    # from the weights file
    with nc.Dataset(weights, 'r') as f:
        src_shape = f.variables['src_grid_dims'][:]
        dst_shape = f.variables['dst_grid_dims'][:]

        dst_lons = np.reshape(f.variables['xc_b'], dst_shape[::-1])
        dst_lats = np.reshape(f.variables['yc_b'], dst_shape[::-1])

    with nc.Dataset(filename, 'w') as outf, nc.Dataset(input_file, 'r') as inf:

        if not var_name in inf.variables:
            print("Variable {} not found in input file {}".format(var_name, input_file))
            sys.exit(1)

        var = inf.variables[var_name]

        # Only support 2D (lon x lat) or 3D (lon x lat x time) input data
        if len(var.dimensions) > 3:
            print("Variable {} has too many dimensions ({}), expect a maximum of 3".format(var_name, len(var.dimensions)))
            sys.exit(1)
    
        # Compare the last two dimensions of the variable to the src_shape from the
        # weights file (it has fastest varying dimension first, so must be reversed) 
        if not all(var.shape[-2:] == src_shape[::-1]):
            print("Input variable {} shape {} does not match source shape {} in weights file".format(var_name,var.shape[-2:],src_shape[::-1]))
            sys.exit(1)
        
        # Copy global attributes from source to destination
        outf.setncatts({k: inf.getncattr(k) for k in inf.ncattrs()})

        outVardims = []

        # If there is a third (assume time) dimension, copy 
        if len(var.dimensions) == 3:
            timedimname = var.dimensions[0]
            # outf.createDimension(timedimname, inf.dimensions[timedimname].size)
            # Use a size of non to make this an unlimited dimension
            outf.createDimension(timedimname, None)
            time = outf.createVariable(timedimname, 'f8', (timedimname))

            outVardims.append(timedimname)

            # Set some defaults
            time.long_name = 'time'
            # time.units = time_units
            time.cartesian_axis = "T"
            time.axis = "T"
            time.calendar_type = "noleap"
            time.calendar = "noleap"
            time.modulo = " "

            # If there is a variable with the same name as the time dimension
            # override with its attributes
            if timedimname in inf.variables:
                timevar = inf.variables[timedimname]
                time.setncatts({k: timevar.getncattr(k) for k in timevar.ncattrs()})
                time[:] = inf.variables[timedimname][:]
        else:
            time = None

        lonname = var.dimensions[-1]
        latname = var.dimensions[-2]

        outf.createDimension(lonname, dst_shape[0])
        outf.createDimension(latname, dst_shape[1])

        print("longitude name: {} size: {}".format(lonname,dst_shape[0]))
        print("latitude name: {} size: {}".format(latname,dst_shape[1]))
    
        outVardims.extend([latname, lonname])

        outVar = outf.createVariable(var.name, var.datatype, outVardims)
        outVar.setncatts({k: var.getncattr(k) for k in var.ncattrs()})

        lons = outf.createVariable(lonname, 'f8', (lonname))
        lons.long_name = 'Nominal Longitude of cell center'
        lons.units = 'degree_east'
        lons.modulo = 360.
        lons.point_spacing = 'even'
        lons.axis = 'X'
        # MOM needs this to be a single dimension, so assume even spacing of
        # longitude at the bottom and grab first row
        lons[:] = dst_lons[0,:]

        lats = outf.createVariable(latname, 'f8', (latname))
        lats.long_name = 'Nominal Latitude of cell center'
        lats.units = 'degree_north'
        lats.point_spacing = 'uneven'
        lats.axis = 'Y'
        # MOM needs this to be a single dimension, so find the column with the
        # largest latitude (least tripolar distortion)
        col = col_idx_largest_lat(dst_lats[:])
        lats[:] = dst_lats[:,col]
    
        if time is None:
            return None
        else:
            return len(time)

def write_output(filename, var_name, var_data, time_idx=None):

    with nc.Dataset(filename, 'r+') as f:

        var = f.variables[var_name]

        if (time_idx is None):
            var[:] = var_data[:]
        else:
            var[time_idx, :] = var_data[:]

@numba.jit
def apply_weights(src, dest_shape, n_s, n_b, row, col, s):
    """
    Apply ESMF regridding weights.
    """

    dest = np.ndarray(dest_shape).flatten()
    dest[:] = 0.0
    src = src.flatten()

    for i in range(n_s):
        dest[row[i]-1] = dest[row[i]-1] + s[i]*src[col[i]-1]

    return dest.reshape(dest_shape)


def remap(src_data, weights):
    """
    Regrid a 2d field and see how it looks.
    """

    with nc.Dataset(weights) as wf:
        dst_shape = wf.variables['dst_grid_dims'][::-1]
        n_s = wf.dimensions['n_s'].size
        n_b = wf.dimensions['n_b'].size
        row = wf.variables['row'][:]
        col = wf.variables['col'][:]
        s = wf.variables['S'][:]

    dest_data = np.ndarray(dst_shape)

    dest_data[:, :] = apply_weights(src_data[:, :], dst_shape,
                                       n_s, n_b, row, col, s)

    return dest_data

def remap_to_output(input_file, output_file, weights_file, method='conserve', varname='runoff', verbose=False, clobber=False, check=False):
    """
    Remapping and check that it is conservative.
    """

    if verbose: print("Opening input file: {}".format(input_file))

    with nc.Dataset(input_file) as f:

        var = f.variables[varname]

        ntime = create_output_ncfile(input, args.varname, weights_file, output_file, None)

        if ntime is None:
            ntime = 1
            time_axis = False
        else:
            time_axis = True

        for idx in range(ntime):

            if time_axis:
                src = var[idx, :]
            else:
                src = var[:]

            if verbose: print("Remapping using weights for index: {}".format(idx))
            dest = remap(src, weights_file)

            if check:
                rel_err = calc_regridding_err(weights_file, src, dest)
                print('ESMF relative error {}'.format(rel_err))
            
            if time_axis:
                write_output(output_file, varname, dest, idx)
            else:
                write_output(output_file, varname, dest)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Regrid a SSS file to a MOM ocean grid")
    parser.add_argument("-i","--inputdir", help="Specify an input directory to find grid files", default=os.getcwd())
    parser.add_argument("-o","--outputdir", help="Specify an output directory to save regridded files", default=os.getcwd())
    parser.add_argument("-v","--verbose", help="Verbose output", action='store_true')
    parser.add_argument("-c","--check", help="Check regridding error", action='store_true')
    parser.add_argument("-va","--varname", help="Variable name of field to regrid")
    parser.add_argument("-w","--weights", help="Weights file to apply")
    parser.add_argument("-f","--force", help="Force writing of output file even if it already exists (default is no)", action='store_true')
    parser.add_argument("inputs", help="files to regrid", nargs='+')
    args = parser.parse_args()

    verbose=args.verbose

    if args.weights is None:
        weights_file = 'weights.nc'
    else:
        weights_file = args.weights

    # Loop over all the inputs from the command line. 
    for input in args.inputs:

        output = os.path.join(args.outputdir,os.path.basename(input))

        if (output == input):
            print("WARNING! Command will overwrite input file. Aborting")
            print("Specify output directory (-o)")
            print("Aborting")
            sys.exit(1)

        if not os.path.isabs(input):
            input = os.path.join(args.inputdir,input)

        if verbose:
            print("Processing input file: {}".format(input))
            print("Output file: {}".format(output))

        remap_to_output(input, output, method='conserve', weights_file = weights_file, varname=args.varname, verbose=args.verbose, clobber=args.force, check=args.check)

