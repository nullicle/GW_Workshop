#import python libraries
from builtins import range
import sys
import os
import numpy as np
import h5py
import math
import argparse
from mpi4py import MPI
import time

#Import standard code base
from coffee import ibvp, actions, solvers, grid
from coffee.diffop import fd
from coffee.diffop.sbp import sbp

#import system to use
import g_wave
import gwave_plotter as gw_plotter

np.set_printoptions(threshold=np.inf, precision=16)

################################################################################
# Parser settings 
################################################################################
# Initialise parser
parser = argparse.ArgumentParser(description=\
"""This program numerically solves an IBVP for the EFE using the 
Friedrich-Nagy gauge.""")

# Parse files
parser.add_argument('-f','-file', help=\
"""The name of the hdf file to be produced.""")
parser.add_argument('-d','-display', default = False, 
    action='store_true', help=\
"""A flag to indicate if visual display is required.""")
parser.add_argument('-A','-Afn', type=float,help=\
"""The initial value of the metric function A.""")
parser.add_argument('-cv','-constraintvisualisation', type=int, help=\
"""The constraint you want to visualise.""")
parser.add_argument('-a','-amplitude', type=float, help=\
"""The amplitude of the wave(s).""")
parser.add_argument('-CFL','-CFLvalue', type=float, help=\
"""The value of the CFL.""")
parser.add_argument('-r','-resolution', type=int, help=\
"""The number of grid points.""")
parser.add_argument('-i', action='store_true', default=False, help=\
    """A flag to indicate the use of the second set of initial conditions.""")
parser.add_argument('-c', default = False, action='store_true',\
     help="""Whether to output constraints or not.""")
parser.add_argument('-pl', nargs='*', type=float, help=\
"""The polarisations of the left boundary condition.""")
parser.add_argument('-pr', nargs='*', type=float, help=\
"""The polarisations of the right boundary condition.""")
     
args = parser.parse_args()
################################################################################  
# These are the commonly altered settings
################################################################################

#output settings
store_output = args.f is not None
display_output = args.d
if store_output and args.f is None:
    print("g_wave_setup.py: error: argument -f/-file is required")
    sys.exit(1)

if args.c and args.f is None:
    print("GCFE_setup.py: error: argument -f/-file is required to output constraints")
    sys.exit(1)
    
# file settings
if store_output:
    args.f = os.path.abspath(args.f)
    try:
        if not os.path.exists(os.path.split(args.f)[0]):
            os.makedirs(os.path.split(args.f)[0])
    except OSError as oserror:
        if oserror.errno is not errno.EEXIST:
            raise oserror

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

# How many systems?
num_of_grids = 1

# How many grid points?
N = args.r

# What grid to use?
xstart = -2
xstop = 2

# Times to run between
tstart = 0.0
tstop = 10000.
# Configuration of System
CFLs = [args.CFL for i in range(num_of_grids)]
tau = 1.0

# Select diffop
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
# Grid point data      
raxis_gdp = [N*2**i for i in range(num_of_grids)]
#raxis_gdp = [N]

# Determine the boundary data
ghost_points = (diffop.ghost_points(),)
internal_points = (diffop.internal_points(),)
b_data = grid.MPIBoundary(
    ghost_points, 
    internal_points, 
    mpi_comm=mpi_comm, 
    number_of_dimensions=1
)

# Build grids
grids = [
    grid.UniformCart(
        (raxis_gdp[i],), 
        [[xstart,xstop]],
        comparison = raxis_gdp[i],
        mpi_comm = mpi_comm,
        boundary_data=b_data
    ) 
    for i in range(num_of_grids)
]

global_z = np.linspace(xstart, xstop, N+1)

################################################################################
# Initialise systems
################################################################################
systems = []
amplitudes = [args.a]
for i in range(num_of_grids):
    systems += [g_wave.G_wave(\
        diffop, tau, global_z, 
        CFL = CFLs[i], 
        amplitude = amplitudes[i],
        A=args.A,
        initialCond = args.i, constraints = args.c,
        pl = args.pl, pr = args.pr
        )]
# Configuration of IBVP
solvers = [solvers.RungeKutta4(sys) for sys in systems]
# solvers = [solvers.BackwardEuler(sys) for sys in systems]
maxIteration = 10000000

################################################################################
# Set up hdf file to store output
################################################################################
if store_output and mpi_comm.rank==0:
    hdf_file = h5py.File(args.f,"w")

################################################################################
# Set up action types for data storage in hdf file
################################################################################
def constraint(it, tslice, system):
    return system.constraint_violation(tslice)
def datawithCC(it, tslice, system):
    return system.datawithCC(tslice)

output_actions = [
    actions.SimOutput.Data(),
    actions.SimOutput.Times(),
    # actions.SimOutput.Constraints(),
    # actions.SimOutput.System(),
    #actions.SimOutput.Domains(),
    # actions.SimOutput.DerivedData("Constraints", constraint, frequency=10),
    # actions.SimOutput.DerivedData("Raw_Data", datawithCC, frequency=10)
    ]

if args.c:
    output_actions.append(actions.SimOutput.Constraints())

################################################################################
# Perform computation
################################################################################
# start_time = time.time()
for i, system in enumerate(systems):
        #Construct Actions
        actionList = []
        if store_output and mpi_comm.rank == 0:
            actionList += [actions.SimOutput(\
                hdf_file,\
                solvers[i], \
                system, \
                grids[i], \
                output_actions,\
                overwrite = True,\
                name = grids[i].name,\
                cmp_ = grids[i].comparison\
                )]
        if display_output and mpi_comm.rank == 0:
            actionList += [gw_plotter.Plotter(
                system,
                frequency = 10, 
                xlim = (xstart, xstop),
                #ylim = [-32.*amplitudes[i], 32.*amplitudes[i]], 
                #ylim = (-0.001, 0.001),
                ylim = [-amplitudes[i]*1.1, amplitudes[i]*1.1], 
                # ylim = [-3, 3], 
                findex = (7,8,3,4,5,6,2), 
                delay = 0.0001
            )]
            if (args.cv is not None):
                actionList += [constr_plotter.Plotter(system, 1, whichC = args.cv, \
                                frequency = 1, xlim = (xstart, xstop), \
                                ylim = [-20., 0.], delay = 0.0)]
        problem = ibvp.IBVP(solvers[i], system, grid = grids[i],\
                maxIteration = 1000000, action = actionList,\
                minTimestep = 1e-6)
        problem.run(tstart, tstop)
        # print("Finished a simulation")
# print(time.time() - start_time, "seconds")

amp = str(args.a).replace('.','p')

# if mpi_comm.rank == mpi_comm.Get_size()-1:
#     np.save("Boundary_Data_" + amp + ".npy",systems[0].boundary_data)
#     np.save("Times_" + amp + ".npy",systems[0].times)