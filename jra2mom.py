#!/usr/bin/env python

import argparse
import sys, os
import subprocess as sp
import numpy as np
import numba
import netCDF4 as nc
# from scipy import ndimage as nd

sys.path.append('/short/v45/aph502/regrid/esmgrids')
from esmgrids.mom_grid import MomGrid
from esmgrids.jra55_grid import Jra55Grid

from helpers import setup_test_input_dir, setup_test_output_dir
from helpers import calc_regridding_err

EARTH_RADIUS = 6370997.0
EARTH_AREA = 4*np.pi*EARTH_RADIUS**2

## def fill_mask_with_nearest_neighbour(field, field_mask):
##     """
##     This is the Python way using grid-box nearest neighbour, an alternative is
##     to do nn based on geographic distance using the above.
##     """

##     new_data = np.ma.copy(field)

##     ind = nd.distance_transform_edt(field_mask,
##                                     return_distances=False,
##                                     return_indices=True)
##     new_data[:, :] = new_data[tuple(ind)]

##     return new_data


def col_idx_largest_lat(lats):
    """
    The col index with the largest lat.
    """
    _, c  = np.unravel_index(np.argmax(lats), lats.shape)

    return c

def create_mom_output(ocean_grid, filename, time_units, history=""):

    f = nc.Dataset(filename, 'w')

    f.createDimension('GRID_X_T', ocean_grid.num_lon_points)
    f.createDimension('GRID_Y_T', ocean_grid.num_lat_points)
    f.createDimension('time')

    lons = f.createVariable('GRID_X_T', 'f8', ('GRID_X_T'))
    lons.long_name = 'Nominal Longitude of T-cell center'
    lons.units = 'degree_east'
    lons.modulo = 360.
    lons.point_spacing = 'even'
    lons.axis = 'X'
    # MOM needs this to be a single dimension
    lons[:] = ocean_grid.x_t[ocean_grid.x_t.shape[0] // 2, :]

    lats = f.createVariable('GRID_Y_T', 'f8', ('GRID_Y_T'))
    lats.long_name = 'Nominal Latitude of T-cell center'
    lats.units = 'degree_north'
    lats.point_spacing = 'uneven'
    lats.axis = 'Y'
    # MOM needs this to be a single dimension
    col = col_idx_largest_lat(ocean_grid.y_t[:])
    lats[:] = ocean_grid.y_t[:, col]

    time = f.createVariable('time', 'f8', ('time'))
    time.long_name = 'time'
    time.units = time_units
    time.cartesian_axis = "T"
    time.axis = "T"
    time.calendar_type = "noleap"
    time.calendar = "noleap"
    time.modulo = " "

    f.close()

def write_mom_output(filename, var_name, var_longname, var_units,
                             var_data, time_idx, time_pt, write_ic=False):

    with nc.Dataset(filename, 'r+') as f:
        if not var_name in f.variables:
            var = f.createVariable(var_name, 'f8',
                                   ('time', 'GRID_Y_T', 'GRID_X_T'),
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
    Apply ESMF regirdding weights.
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

def remap_to_mom(input_file, mom_hgrid, mom_mask, output_file, method='conserve', varname='runoff', verbose=False, clobber=False):
    """
    Remapping JRA and check that it is conservative.

    It will be necessary to use a destination mask. We want to move all
    points from src into unmasked parts of the destination.
    """

    if verbose: print("Opening input file: {}".format(input_file))

    oasis_grids_dir = os.environ["OASIS_GRIDS_DIR"]
    cmd = [os.path.join(oasis_grids_dir, 'remapweights.py')]

    # input_grid = Jra55Grid(input_file)

    output_dir = os.path.dirname(output_file)

    if verbose: print("Output directory: {}".format(output_dir))

    weights = os.path.join(output_dir, 'JRA_MOM_conserve.nc')

    if not os.path.exists(weights):

        args = ['JRA55', 'MOM', '--src_grid', input_file,
                '--dest_grid', mom_hgrid, '--dest_mask', mom_mask,
                '--method', method, '--output', weights]
    
        if verbose: print("Generating weights file : {} using\n{}".format(weights," ".join(cmd + args)))
    
        ret = sp.call(cmd + args)
        assert ret == 0
        assert os.path.exists(weights)

    
    # Only use these to pull out the dimensions of the grids.
    mom = MomGrid.fromfile(mom_hgrid, mask_file=mom_mask)

    with nc.Dataset(input_file) as f:

        var = f.variables[varname]

        if "time" in f.variables:
            time = f.variables["time"]
            time_units = time.units
        else:
            time = numpy.arange(0,var.shape[0])
            time_units = "days since 1900-01-01"
    
        create_mom_output(mom, output_file, time_units)

        for idx in range(var.shape[0]):

            src = var[idx, :]

            if verbose: print("Remapping using weights for index: {}".format(idx))
            dest = remap(src, weights, (mom.num_lat_points, mom.num_lon_points))

            rel_err = calc_regridding_err(weights, src, dest)
            print('ESMF relative error {}'.format(rel_err))

            write_mom_output(output_file, varname, var.long_name, var.units, dest, idx, time[idx])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Regrid a JRA55 file to a MOM ocean grid")
    parser.add_argument("-i","--input_dir", help="Specify an input directory to find grid files", default=os.getcwd())
    parser.add_argument("-o","--output_dir", help="Specify an output directory to save regridded files", default=os.getcwd())
    parser.add_argument("-v","--verbose", help="Verbose output", action='store_true')
    parser.add_argument("-va","--varname", help="Variable name of field to regrid")
    parser.add_argument("-f","--force", help="Force writing of output file even if it already exists (default is no)", action='store_true')
    parser.add_argument("inputs", help="JRA files", nargs='+')
    args = parser.parse_args()

    verbose=args.verbose

    # Loop over all the inputs from the command line. 
    for jrainput in args.inputs:

        if verbose: print("Processing {}".format(jrainput))

        suffix = '.nc'

        mom_output = os.path.join(args.output_dir,os.path.basename(jrainput))

        if mom_output.endswith(suffix):
            mom_output = mom_output[:-len(suffix)] + '_mom.nc'
        else:
            mom_output = mom_output + '_mom.nc'

        if not os.path.isabs(jrainput):
            jrainput = os.path.join(args.input_dir,jrainput)

        mom_hgrid = os.path.join(args.input_dir, 'ocean_hgrid.nc')
        mom_mask = os.path.join(args.input_dir, 'ocean_mask.nc')

        if verbose:
            print("Input file: {}".format(jrainput))
            print("MOM hgrid file: {}".format(mom_hgrid))
            print("MOM mask file: {}".format(mom_mask))
            print("MOM output file: {}".format(mom_output))

        remap_to_mom(jrainput, mom_hgrid, mom_mask, mom_output, method='conserve', varname=args.varname, verbose=args.verbose, clobber=args.force)

