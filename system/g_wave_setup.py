#!/usr/bin/env python
# encoding: utf-8


"""
g_wave_setup.py

Created by Chris Stevens 2023
Copyright (c) 2023 University of Canterbury. All rights reserved.
"""

# Import Python libraries
import sys
import os
import numpy as np
import h5py
import argparse
from mpi4py import MPI

# Import coffee 
from coffee import ibvp, actions, solvers, grid
from coffee.diffop.sbp import sbp

# Import system
import g_wave
import gwave_plotter as gw_plotter

np.set_printoptions(threshold=np.inf, precision=16)

################################################################################
# Parser settings 
################################################################################

# Initialise parser
parser = argparse.ArgumentParser(description=\
"""The Friedrich-Nagy gauge in plane symmetry.""")

# Parse files
parser.add_argument('-f','-file', help=\
"""The name of the hdf file to be produced.""")
parser.add_argument('-d','-display', default = False, 
    action='store_true', help=\
"""A flag to indicate if visual display is required.""")
parser.add_argument('-A','-Afn', type=float,help=\
"""The initial value of the metric function A.""")
parser.add_argument('-a','-amplitude', type=float, help=\
"""The amplitude of the wave(s).""")
parser.add_argument('-CFL','-CFLvalue', type=float, help=\
"""The value of the CFL.""")
parser.add_argument('-r','-resolution', type=int, help=\
"""The number of grid points.""")
parser.add_argument('-pl', nargs='*', type=float, help=\
"""The polarisations of the left boundary condition.""")
parser.add_argument('-pr', nargs='*', type=float, help=\
"""The polarisations of the right boundary condition.""")
     
args = parser.parse_args()

################################################################################  
# These are the commonly altered settings
################################################################################

# Output settings
store_output = args.f is not None
display_output = args.d

if args.f is not None:
    args.f = os.path.abspath(args.f)
    
# Check input
if args.a is None:
    print("g_wave_setup.py: error: argument -a/-amplitude is required.")
    sys.exit(1)
if args.A is None:
    print("g_wave_setup.py: error: argument -A/-Afn is required.")
    sys.exit(1)
if args.r is None:
    print("g_wave_setup.py: error: argument -r/-resolution is required.")
    sys.exit(1)
if args.CFL is None:
    print("g_wave_setup.py: error: argument -CFL/-CFLvalue is required.")
    sys.exit(1)
if args.pl is None:
    print("g_wave_setup.py: error: argument -pl is required.")
    sys.exit(1)
if args.pr is None:
    print("g_wave_setup.py: error: argument -pr is required.")
    sys.exit(1)

# Number of intervals (number of grid points -1)
N = args.r

# Spatial grid bounds
zstart = -2.
zstop  = 2.

# Times to run between
tstart = 0.0
tstop = 1000.

# SAT boundary method parameter
tau  = 1.0

# Select differential operator
diffop = sbp.D43_Strand(sbp.BOUNDARY_TYPE_GHOST_POINTS)

################################################################################      
## MPI set up                                                                         
################################################################################ 

dims = MPI.Compute_dims(MPI.COMM_WORLD.size, [0])                                    
periods = [0]                                                                        
reorder = True                                                                       
mpi_comm = MPI.COMM_WORLD.Create_cart(dims, periods=periods, reorder=reorder)        

################################################################################
# Grid construction
################################################################################

# Determine the boundary data
ghost_points = (diffop.ghost_points(),)
internal_points = (diffop.internal_points(),)
b_data = grid.MPIBoundary(
    ghost_points, 
    internal_points, 
    mpi_comm=mpi_comm, 
    number_of_dimensions=1
)

# Build grid
grid = grid.UniformCart(
        (N,), 
        [[zstart,zstop]],
        comparison = N,
        mpi_comm = mpi_comm,
        boundary_data=b_data
    )

# Global spatial grid points 
global_z = np.linspace(zstart, zstop, N+1)

################################################################################
# Initialise system
################################################################################

system = g_wave.G_wave(\
    diffop, tau, global_z, 
    CFL = args.CFL, 
    amplitude = args.a,
    A = args.A,
    pl = args.pl, pr = args.pr
    )

# Configuration of IBVP
solver = solvers.RungeKutta4(system)
maxIteration = 10000000

################################################################################
# Set up hdf file to store output
################################################################################

if store_output and mpi_comm.rank==0:
    hdf_file = h5py.File(args.f,"w")

################################################################################
# Set up action types for data storage in hdf file
################################################################################

if store_output:
    output_actions = [
        actions.SimOutput.Data(),
        actions.SimOutput.Times()
    ]

################################################################################
# Perform computation
################################################################################

# Construct Actions
actionList = []
if store_output and mpi_comm.rank == 0:
    actionList += [actions.SimOutput(\
        hdf_file,\
        solver, \
        system, \
        grid, \
        output_actions,\
        overwrite = True,\
        name = grid.name,\
        cmp_ = grid.comparison\
        )]
if display_output and mpi_comm.rank == 0:
    actionList += [gw_plotter.Plotter(
        system,
        frequency = 10, 
        xlim = (zstart, zstop),
        ylim = [-args.a*1.1, args.a*1.1], 
        findex = (7,8,3,4,5,6,2), 
        delay = 0.0001
    )]
problem = ibvp.IBVP(solver, system, grid = grid,\
        maxIteration = 1000000, action = actionList,\
        minTimestep = 1e-6)
problem.run(tstart, tstop)
print("Simulation complete.")