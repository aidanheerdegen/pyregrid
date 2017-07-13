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

def create_mom_output(ocean_grid, filename, time_units, history=""):

    g = nc.Dataset(ocean_grid, 'r')

    x_t = g.variables['geolon_t'][:]
    y_t = g.variables['geolat_t'][:]

    g.close()

    f = nc.Dataset(filename, 'w')

    f.createDimension('xt_ocean', x_t.shape[1])
    f.createDimension('yt_ocean', x_t.shape[0])
    f.createDimension('time')

    lons = f.createVariable('xt_ocean', 'f8', ('xt_ocean'))
    lons.long_name = 'Nominal Longitude of T-cell center'
    lons.units = 'degree_east'
    lons.modulo = 360.
    lons.point_spacing = 'even'
    lons.axis = 'X'
    # MOM needs this to be a single dimension
    print(lons.shape,x_t.shape,x_t[x_t.shape[0] // 2, :].shape)
    lons[:] = x_t[x_t.shape[0] // 2, :]

    lats = f.createVariable('yt_ocean', 'f8', ('yt_ocean'))
    lats.long_name = 'Nominal Latitude of T-cell center'
    lats.units = 'degree_north'
    lats.point_spacing = 'uneven'
    lats.axis = 'Y'
    # MOM needs this to be a single dimension
    col = col_idx_largest_lat(y_t[:])
    lats[:] = y_t[:, col]

    time = f.createVariable('time', 'f8', ('time'))
    time.long_name = 'time'
    time.units = time_units
    time.cartesian_axis = "T"
    time.axis = "T"
    time.calendar_type = "noleap"
    time.calendar = "noleap"
    time.modulo = " "

    f.close()

    return x_t.shape

def write_mom_output(filename, var_name, var_longname, var_units,
                             var_data, time_idx, time_pt, write_ic=False):

    with nc.Dataset(filename, 'r+') as f:
        if not var_name in f.variables:
            var = f.createVariable(var_name, 'f8',
                                   ('time', 'yt_ocean', 'xt_ocean'),
                                   fill_value=-1.e+34)
            var.missing_value = -1.e+34
            var.long_name = var_longname
            var.units = var_units

        var = f.variables[var_name]

        if write_ic:
            var[0, :] = var_data[:]
            f.variables['time'][0] = time_pt
        else:
            var[time_idx, :] = var_data[:]
            f.variables['time'][time_idx] = time_pt


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


def remap(src_data, weights, dest_shape):
    """
    Regrid a 2d field and see how it looks.
    """

    dest_data = np.ndarray(dest_shape)

    with nc.Dataset(weights) as wf:
        n_s = wf.dimensions['n_s'].size
        n_b = wf.dimensions['n_b'].size
        row = wf.variables['row'][:]
        col = wf.variables['col'][:]
        s = wf.variables['S'][:]

    dest_data[:, :] = apply_weights(src_data[:, :], dest_data.shape,
                                       n_s, n_b, row, col, s)

    return dest_data

def remap_to_mom(input_file, mom_hgrid, output_file, weights_file, method='conserve', varname='runoff', verbose=False, clobber=False):
    """
    Remapping and check that it is conservative.
    """

    if verbose: print("Opening input file: {}".format(input_file))

    output_dir = os.path.dirname(output_file)

    if verbose: print("Output directory: {}".format(output_dir))

    weights = os.path.join(output_dir, weights_file)

    if not os.path.exists(weights):

        print("Weights file does not exist: {}".format(weights))
    
    with nc.Dataset(input_file) as f:

        var = f.variables[varname]

        if "time" in f.variables:
            time = f.variables["time"]
            time_units = time.units
        else:
            time = numpy.arange(0,var.shape[0])
            time_units = "days since 1900-01-01"
    
        (ny, nx) = create_mom_output(mom_hgrid, output_file, time_units)

        print(nx,ny)

        for idx in range(var.shape[0]):

            src = var[idx, :]

            if verbose: print("Remapping using weights for index: {}".format(idx))
            dest = remap(src, weights, (ny,nx))

            # Cannot calculate error as there are no valid area values in weights file
            # rel_err = calc_regridding_err(weights, src, dest)
            # print('ESMF relative error {}'.format(rel_err))

            write_mom_output(output_file, varname, var.long_name, var.units, dest, idx, time[idx])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Regrid a SSS file to a MOM ocean grid")
    parser.add_argument("-i","--input_dir", help="Specify an input directory to find grid files", default=os.getcwd())
    parser.add_argument("-o","--output_dir", help="Specify an output directory to save regridded files", default=os.getcwd())
    parser.add_argument("-v","--verbose", help="Verbose output", action='store_true')
    parser.add_argument("-va","--varname", help="Variable name of field to regrid")
    parser.add_argument("-w","--weights", help="Weights file to apply")
    parser.add_argument("-t","--template", help="MOM template file with grid information", default='ocean_grid.nc')
    parser.add_argument("-f","--force", help="Force writing of output file even if it already exists (default is no)", action='store_true')
    parser.add_argument("inputs", help="files to regrid", nargs='+')
    args = parser.parse_args()

    verbose=args.verbose

    suffix = '.nc'


    mom_hgrid = os.path.join(args.input_dir, 'ocean_grid.nc')

    if args.weights is None:
        weights_file = 'weights.nc'
    else:
        weights_file = args.weights

    if verbose: print("MOM hgrid file: {}".format(mom_hgrid))

    # Loop over all the inputs from the command line. 
    for input in args.inputs:

        if verbose: print("Processing {}".format(input))

        mom_output = os.path.join(args.output_dir,os.path.basename(input))

        if mom_output.endswith(suffix):
            mom_output = mom_output[:-len(suffix)] + '_mom.nc'
        else:
            mom_output = mom_output + '_mom.nc'

        if not os.path.isabs(input):
            input = os.path.join(args.input_dir,input)

        if verbose: print("MOM output file: {}".format(mom_output))

        remap_to_mom(input, mom_hgrid, mom_output, method='conserve', weights_file = weights_file, varname=args.varname, verbose=args.verbose, clobber=args.force)

