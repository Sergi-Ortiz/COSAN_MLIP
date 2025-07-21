#!/Users/sergiortizropero/miniconda3/envs/ASE_all/bin/python
# Sergi Ortiz @ ICMAB
# 21 July 2025
# Perform MLIP MD simulations
# Analysis is performed using the associated .ipynb notebooks


#
#   IMPORT SECTION
#

# general imports
import os
import numpy as np
import pickle
import argparse

# plt
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
rc('lines', lw=1, color='b')
rc('legend', loc='best')
plt.rcParams["legend.fancybox"] = False
plt.rcParams["legend.edgecolor"] = 'black'
plt.rcParams['legend.borderpad'] = 0.25
plt.rcParams['legend.fontsize'] = 11
plt.rcParams.update({'pgf.preamble': r'\usepackage{amsmath}'})

# ASE imports
from ase.visualize import view
from ase.io import write, read
import ase.units as units
from ase import Atoms
from ase.calculators.tip3p import angleHOH, rOH
from ase.io.trajectory import Trajectory
from ase.md import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase.optimize import BFGS
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution



# CLI 
parser = argparse.ArgumentParser(description='Perform a MD simulation of either water or COSAN in void or in solution.')

# positional arguments
parser.add_argument(
    '-m', '--model', 
    type=str,
    default='MACE',
    required=False,
    help='MLIP model to use. Available: MACE, ORB and UMA. Default: MACE',
)
parser.add_argument(
    '-n', '--name',
    default=None,
    required=False,
    help='Name of the simulation. If exists, it will overwrite. Default: <void/solv>_<CALC_NAME>_<PROD_ns_length>.',
)
'''
parser.add_argument(
    '-s', '--spin', 
    type=int,
    default=1,
    required=False,
    help='Total spin of the system. Default: singlet (1).',
)
parser.add_argument(
    '-c', '--charge', 
    type=int,
    default=0,
    required=False,
    help='Total charge of the system. Default: neutral (0).',
)
'''
parser.add_argument(
    '-a', '--molecule', 
    type=str,
    choices=['water', 'cosan'],
    required=True,
    help='System to study. Options: water, cosan. Automatically sets spin and charge, if none given.',
)
parser.add_argument(
    '-v', '--solvation', 
    type=bool,
    default=False,
    choices=[False, True],
    required=False,
    help='Wether the system is solvated or not. Default: False. THERE IS A BUG, IF -v flag IS USED, IT WILL ALWAYS SAY SOLVATION:TRUE',
)
parser.add_argument(
    '-d', '--device', 
    type=str,
    default='cpu',
    required=False,
    help='Device to use. Options: cuda, cpu. Default: cpu.',
)
parser.add_argument(
    '-l', '--length', 
    type=int,
    default=10,
    required=False,
    help='Length of the simulation in ns. Default: 10 ns. Obs. Trajectory is divided into 1 ns chunks.',
)
parser.add_argument(
    '-t', '--temperature', 
    type=int,
    default=300,
    required=False,
    help='Temperature of the simulation in K. Default 300 K.',
)
parser.add_argument(
    '-p', '--pressure', 
    type=float,
    default=1.0,
    required=False,
    help='Pressure of NPT relaxations (in bar) (only solvated systems). Default 1 bar.',
)

# obtain the arguments
args = parser.parse_args()
model = args.model
sim_name = args.name
#mol_spin = args.spin
#mol_charge = args.charge
mol = args.molecule
device = args.device
bool_solv = args.solvation
sim_length = args.length
temp = args.temperature
pressure = args.pressure


#
#   PREPROCESSING
#

# SPIN and CHARGE
if mol == 'water':
    print('setting spin and charge for water')
    spin = 1
    charge = 0

elif mol == 'cosan':
    if bool_solv:
        print('setting spin and charge for solvated COSAN')
        # if solvated, COSAN is treated as a neutral singlet
        spin = 1
        charge = 0
    else:
        print('setting spin and charge for void COSAN (- singlet)')
        spin = 1
        charge = 0
else:
    raise ValueError('Molecule was not detected properly.')


#
#   MODEL IMPORTS
#
if model == 'MACE':
    try:
        # MACE
        print('loading MACE-MP-0 (medium)')
        from mace.calculators import mace_mp
        CALC_ASE = mace_mp(
            model="medium", 
            dispersion=False, 
            default_dtype="float32", 
            device=device,
            )
        CALC_NAME = 'MACE-MP'
    except:
        raise ValueError('Tried to import MACE but failed.')
elif model == 'ORB':
    try:
        # ORB
        print('loading ORB-v3-conservative-inf-omat')
        from orb_models.forcefield.pretrained import orb_v3_conservative_inf_omat
        from orb_models.forcefield.calculator import ORBCalculator
        orbff_v3 = orb_v3_conservative_inf_omat(device=device)
        CALC_ASE = ORBCalculator(orbff_v3, device=device)
        CALC_NAME = 'ORB'
    except:
        raise ValueError('Tried to import ORB but failed.')
elif model == 'UMA':
    try:
        # UMA
        print('loading UMA_small')
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.units.mlip_unit import load_predict_unit

        # define the UMA ASE calculator
        uma_predictor = load_predict_unit(
            path='/Users/sergiortizropero/TFG_phys/NNPs_TFG/models/ase_uma/uma-s-1p1.pt', 
            device=device,                   
            inference_settings='default',   
        )
        CALC_ASE = FAIRChemCalculator(
            uma_predictor,
            task_name='omol',               
        )
        CALC_NAME = 'UMA'
    except:
        raise ValueError('Tried to import UMA but failed.')
else:
    raise ValueError('The calculator was not detected properly.')


#
#   FILE PREPROCESSING
#

# define the simulation path, where all data will be stored
if sim_name is None:
    if bool_solv:
        sim_name = f'solvated_{CALC_NAME}_{sim_length}'
    else: 
        sim_name = f'void_{CALC_NAME}_{sim_length}'

sim_path = os.path.join('../simulations/', f'{mol}_mds', sim_name)
if not os.path.exists(sim_path):
    os.makedirs(sim_path)




#
#   MD FUNCTION DEFINITIONS
#

# MOLECULAR SYSTEM DEFINITION
def prepare_cosan(temp, bool_solv=False):

    # load the structure
    if bool_solv:
        # solvated
        print('Fetching solvated COSAN system')
        atoms = read(os.path.join('./structures/cosan/cosan_solv.pdb'))
        atoms.info = {
            'spin': spin,
            'charge': charge,
            'k': 0,
        }
        atoms.set_cell((28, 28, 32))
        atoms.center()
        atoms.set_pbc(True)

    else:
        # void
        print('Fetching void COSAN system')
        atoms = read(os.path.join('./structures/cosan/cosan_void.xyz'))
        atoms.info = {
            'spin': spin,
            'charge': charge,
            'k': 0,
        }
        atoms.set_cell((20, 20, 20))
        atoms.center()
        atoms.set_pbc(False)

    atoms.calc = CALC_ASE

    # initialize velocities according to MB distrobution
    MaxwellBoltzmannDistribution(atoms, temperature_K=temp)

    return atoms


def prepare_water(temp, bool_solv=False):

    # load the structure
    if bool_solv:
        # solvated
        print('Fetching solvated WATER system')
        atoms = read(os.path.join('./structures/water/water_box.pdb'))
        atoms.info = {
            'spin': spin,
            'charge': charge,
            'k': 0,
        }
        atoms.set_cell((20.5, 20.5, 20.5))
        atoms.center()
        atoms.set_pbc(True)
    else:
        # void
        print('Fetching void WATER system')
        x = angleHOH * np.pi / 180 / 2
        pos = [
            [0, 0, 0],
            [0, rOH * np.cos(x), rOH * np.sin(x)],
            [0, rOH * np.cos(x), -rOH * np.sin(x)],
        ]
        atoms = Atoms('OH2', positions=pos)
        atoms.info = {
            'spin': spin,
            'charge': charge,
            'k': 0,
        }
        atoms.set_cell((20, 20, 20))
        atoms.center()
        atoms.set_pbc(False)

    atoms.calc = CALC_ASE

    # initialize velocities according to MB distrobution
    MaxwellBoltzmannDistribution(atoms, temperature_K=temp)

    return atoms




def run_NVT(atoms, length, dt, T, sim_path, sim_name, save_int=100, overwrite=False):

    # define paths
    log_path = os.path.join(sim_path, sim_name + f'_NVT_equilibration.log')
    traj_path = os.path.join(sim_path, sim_name + f'_NVT_equilibration.traj')

    # NVT run
    md = Langevin(
        atoms,
        dt*units.fs,
        temperature_K=T,
        friction=0.01,
        logfile=log_path,
        loginterval=save_int,
    )

    # trajectory
    if os.path.exists(traj_path) and (not overwrite):
        raise ValueError('Trajectory file already exists!')
    traj = Trajectory(traj_path, 'w', atoms)

    # save every save_int steps
    md.attach(traj.write, interval=save_int)
    md.run(length)
    print(f'NVT equilibration has finished!')

    return atoms

def NVT_equilibration(atoms, nvt_eq_steps, dt, T, sim_path, sim_name, save_int):
    
    # PERFORM INITIAL NVT EQUILIBRATION
    run_NVT(
        atoms, 
        nvt_eq_steps, 
        dt, 
        T=T, 
        sim_path=sim_path, 
        sim_name=sim_name,  
        save_int=save_int, 
    )

    # save to a file
    atoms.info['positions'] = atoms.get_positions()
    atoms.info['velocities'] = atoms.get_velocities()
    #print(cosan.info['velocities'])
    #print(cosan.info['k_nvt'])

    # save object
    with open(os.path.join(sim_path, sim_name+f'_NVT_equilibration.pkl'), 'wb') as f:
        pickle.dump(atoms, f)

    return atoms




def run_NPT(atoms, length, dt, T, P, sim_path, sim_name, save_int=100, overwrite=False):

    # define paths
    log_path = os.path.join(sim_path, sim_name + f'_NPT_equilibration.log')
    traj_path = os.path.join(sim_path, sim_name + f'_NPT_equilibration.traj')

    # NPT run
    # Berendsen CANNOT BE USED FOR PRODUCTION
    md = NPTBerendsen(
        atoms,
        dt*units.fs,
        temperature_K=T,
        pressure=P,
        compressibility=4.6*10**-5, # that of water
        logfile=log_path,
        loginterval=save_int,
    )

    # trajectory
    if os.path.exists(traj_path) and (not overwrite):
        raise ValueError('Trajectory file already exists!')
    traj = Trajectory(traj_path, 'w', atoms)

    # save every save_int steps
    md.attach(traj.write, interval=save_int)
    md.run(length)
    print(f'NPT equilibration has finished!')

    return atoms

def NPT_equilibration(atoms, npt_eq_steps, dt, T, P, sim_path, sim_name, save_int=100):

    # NPT EQUILIBRATION
    run_NPT(
        atoms, 
        npt_eq_steps, 
        dt, 
        T=T, 
        P=P,
        sim_path=sim_path, 
        sim_name=sim_name, 
        save_int=save_int, 
    )

    # save to a file
    atoms.info['positions'] = atoms.get_positions()
    atoms.info['velocities'] = atoms.get_velocities()
    #print(cosan.info['velocities'])
    #print(cosan.info['k_npt'])

    # save object
    with open(os.path.join(sim_path, sim_name+f'_NPT_equilibratiin.pkl'), 'wb') as f:
        pickle.dump(atoms, f)

    return atoms




def run_prod(atoms, length, dt, T, k, sim_path, sim_name, save_int=100, overwrite=False):

    # define paths
    log_path = os.path.join(sim_path, sim_name + f'_PROD_{k}.log')
    traj_path = os.path.join(sim_path, sim_name + f'_PROD_{k}.traj')

    # NVT run
    md = Langevin(
        atoms,
        dt*units.fs,
        temperature_K=T,
        friction=0.01,
        logfile=log_path,
        loginterval=save_int,
    )

    # trajectory
    if os.path.exists(traj_path) and (not overwrite):
        raise ValueError('Trajectory file already exists!')
    traj = Trajectory(traj_path, 'w', atoms)

    # save every save_int steps
    md.attach(traj.write, interval=save_int)
    md.run(length)
    print(f'NVT prod {k} has finished')

    return atoms


def NVT_prod(atoms, sim_length, dt, T, sim_path, sim_name, save_int=100):

    # NVT PRODUCTION
    k = 0
    converged_prod = False
    while not converged_prod: 

        # start with k = 0
        # add 1
        atoms.info['k'] += 1
        k = atoms.info['k']

        # run a fragment
        # simulate k = 1
        run_prod(
            atoms, 
            sim_length, 
            dt, 
            T=T, 
            sim_path=sim_path, 
            sim_name=sim_name, 
            k=k, 
            save_int=save_int, 
            overwrite=restart,
        )

        # save to a file
        atoms.info['positions'] = atoms.get_positions()
        atoms.info['velocities'] = atoms.get_velocities()
        #print(atoms.info['velocities'])
        #print(atoms.info['k'])

        with open(os.path.join(sim_path, sim_name+f'_PROD_{k}.pkl'), 'wb') as f:
            pickle.dump(atoms, f)

        # check convergence
        if atoms.info['k'] >= K:
            converged_prod = True    

    print('NVT production has finished!')

    return atoms


# restart production if needed
def restart_md(pkl_path, verbose=True):
    '''
    Restart the simulation from a pkl file if needed.
    '''
    
    with open(pkl_path, 'rb') as f:
        atoms_restart = pickle.load(f)

    if verbose:
        print('restarting with Atoms.object with properties:')
        print(atoms_restart.info['charge'])
        print(atoms_restart.info['spin'])
        print(atoms_restart.info['k'])
        #print(cosan_restart.info['positions'])
        #print(cosan_restart.info['velocities'])

    return atoms_restart



# relaxation (geometry optimization)
def relax_structure(atoms, sim_path):

    # set convergence criterion depending on the system
    if mol == 'cosan':
        if bool_solv:
            convergence = 1000
        else:
            convergence = 2
    else: 
        if bool_solv:
            convergence = 0.05
        else:
            convergence = 0.05

    dyn = BFGS(atoms, trajectory=os.path.join(sim_path, 'relaxation.traj'))
    dyn.run(fmax=convergence)

    # save the optimized structure as a pkl file (if needed)
    with open(os.path.join(sim_path, 'optimized.pkl'), 'wb') as f:
        pickle.dump(atoms, f)


# ------------------------------------------------------------------------------

#
#   SIMULATION PARAMETERS
#

dt = 1                                      # in fs (hardcoded)
sim_steps = int(sim_length * 1000 / dt)     # total number of steps
K = int(sim_length)                         # simulation fragments (1 ns each)
save_int = 100                              # save every 0.1 ps (beware of temporal self-correlation)

# equilibration parameters
nvt_eq_steps = 500 * 1000                   # 0.5 ns NVT equilibration
npt_eq_steps = 500 * 1000                   # 0.5 ns NPT equilibration


# HOPE I DONT HAVE TO TOUCH THIS EVER
restart = False
restart_file = ''
if restart:
    print(f'restarting from {restart_file}')


#===============================#
#       ACTUAL SIMULATION       #
#===============================#

print('\n\nBEGINNING SIMULATION')
print(sim_name)
print(sim_path)
print()
print('SYSTEMS PARAMETERS')
print(f'molecule:    {mol}')
print(f'solvated:    {bool_solv}')
print(f'spin:        {spin}')
print(f'charge:      {charge}')
print(f'MLIP model:  {model}')
print(f'ML device:   {device}')
print()
print('SIMULATION PARAMETERS')
print(f'Prod length: {sim_length} (ns)')
print(f'1ns fragmts: {K}')
print(f'save intrvl: {save_int}')
print(f'Total steps: {sim_steps}')
print(f'Temperature: {temp} (K)')
if bool_solv:
    print(f'Pressure:    {pressure} (bar)')
print('\n\n')



# define the system
if mol == 'cosan':
    atoms = prepare_cosan(temp=temp, bool_solv=bool_solv)
elif mol == 'water':
    atoms = prepare_water(temp=temp, bool_solv=bool_solv)
else:
    raise ValueError('Molecule not detected')

# relax the system
relax_structure(atoms, sim_path)

# equilibrate if needed
if bool_solv:
    # if solvated, perform equilibration (short NVT and NPT)
    # else go straight to production
    NVT_equilibration(atoms, nvt_eq_steps, dt, temp, sim_path, sim_name, save_int)
    NPT_equilibration(atoms, npt_eq_steps, dt, temp, pressure, sim_path, sim_name, save_int)

# NVT production run of desired length
NVT_prod(atoms, sim_length, dt, temp, sim_path, sim_name, save_int)
print('Finished!')