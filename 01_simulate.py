import gc
import json
import os
import shutil
import click
import logging

import numpy as np
from polychrom.simulation import Simulation
from polychrom.forces import spherical_confinement, polynomial_repulsive, heteropolymer_SSW
from polychrom.starting_conformations import grow_cubic, create_random_walk
from polychrom.hdf5_format import HDF5Reporter
from polychrom.forcekits import polymer_chains
import openmm

def write_pdb(coordinates: np.array, dir: str, filename: str, compartment_markup: list[str], n_chains: int = 1):
    file_path = os.path.join(dir, filename)
    with open(file_path, 'w') as pdb_file:
        pdb_file.write('HEADER    Polymer\n')

        for i, (x, y, z) in enumerate(coordinates, start=1):
            chain = str(i // len(compartment_markup))
            compartment = compartment_markup[i % len(compartment_markup)]
            pdb_file.write(f'HETATM{i:5d}  C   CM{compartment} {chain}   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n')

        chain_length = len(coordinates) // n_chains
        for chain in range(n_chains):
            for i in range(chain * chain_length + 1, (chain + 1) * chain_length):
                pdb_file.write(f"CONECT{i:5d}{i+1:5d}\n")
        pdb_file.write('END\n')


def write_pdb_legacy(coordinates: np.array, dir: str, filename: str, n_chains: int = 1):
    file_path = os.path.join(dir, filename)
    with open(file_path, 'w') as pdb_file:
        pdb_file.write('HEADER    Polymer\n')

        for i, (x, y, z) in enumerate(coordinates, start=1):
            pdb_file.write(f'HETATM{i:5d}  C   UNK     1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n')

        chain_length = len(coordinates) // n_chains
        for chain in range(n_chains):
            for i in range(chain * chain_length + 1, (chain + 1) * chain_length):
                pdb_file.write(f"CONECT{i:5d}{i+1:5d}\n")
        pdb_file.write('END\n')


@click.command()
@click.option('--compartment-markup', help='Path to compartment markup file')
@click.option('--result-dir', help='Path to result directory')
@click.option('--n-chromosomes', default=1, help='Number of chromosomes')
@click.option('--num-runs', default=1, help='Number of runs')
@click.option('--num-blocks', default=100, help='Number of blocks')
@click.option('--block-steps', default=400, help='Number of steps in a block')
@click.option('--save-starting-conformation', is_flag=True, help='Save starting conformation')
@click.option('--save-block-conformation', is_flag=True, help='Save block conformations')
@click.option('--log', default='INFO', help='Logging level')
def main(
    num_runs: int,
    num_blocks: int,
    block_steps: int,
    compartment_markup: str,
    result_dir: str,
    save_starting_conformation: bool,
    save_block_conformation: bool,
    n_chromosomes: int,
    log: str
):
    logging.basicConfig(level=log)
    logger = logging.getLogger(__name__)
    logger.info('Logging level: %s', log)
    logger.info('Compartment markup: %s', compartment_markup)
    logger.info('Result directory: %s', result_dir)

    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir)

    bond_length = 1.0  # Длина связей
    bond_wiggle = 0.2  # Отклонение от длины связей
    angle_stiffness = 0.05  # Жёсткость углов
    repulsion_energy = 1.0  # Энергия отталкивания
    attraction_energy = 0.2  # Энергия притяжения
    repulsion_radius = 1.0  # Радиус отталкивания для вычислений
    attraction_radius = 1.02  # Радиус притягивания для вычислений
    selective_repulsion_energy = 1.5
    selective_attraction_energy = 0.75

    reporter = HDF5Reporter(folder=result_dir, max_data_length=200, overwrite=True)
    
    for run in range(num_runs):
        logger.info(f'Run {run+1}/{num_runs}')

        with open(f"{compartment_markup}/random_compartment{run}.json", 'r') as f:
            compartments = json.load(f)

        num_monomers = len(compartments)

        sim = Simulation(
        platform="CUDA",
        integrator="variableLangevin",    # Динамика Ланжевена 
        error_tol=0.01,
        collision_rate=0.03,
        N=num_monomers*n_chromosomes,
        temperature=300,
        reporters=[reporter],
        max_Ek = 100,
        )

        positions = create_random_walk(step_size=bond_length, N=num_monomers * n_chromosomes)

        # positions = grow_cubic(N=num_monomers * n_chromosomes, boxSize=int((num_monomers * n_chromosomes)**(1/3)*2), method='linear')
        sim.set_data(positions, center=False)

        if save_starting_conformation and run == 0:
            logger.info('Saving starting conformation')
            write_pdb(positions, result_dir, 'starting_conformation.pdb', compartments, n_chromosomes)

        letters = 'ABCDEFGHIJKLMNOP' # Кодировка типов мономеров (16 штук)

        monomer_dict = {letter: index for index, letter in enumerate(letters)}

        monomer_types = np.array([monomer_dict[c] for c in compartments])
    
        interaction_matrix = np.load(f"{compartment_markup}/interaction_matrix.npy")  # Загрузка interaction_matrix
        

        polymer_force = polymer_chains(
            sim,
            chains=[(begin, begin+num_monomers, False) for begin in range(0, num_monomers * n_chromosomes, num_monomers)],  # Линейная цепь
            bond_force_kwargs={
                "bondWiggleDistance": bond_wiggle,
                "bondLength": bond_length
                },
            angle_force_kwargs={
                "k": angle_stiffness
                },
            nonbonded_force_func=heteropolymer_SSW,
            nonbonded_force_kwargs={
                "interactionMatrix": interaction_matrix,
                "monomerTypes": np.tile(monomer_types, n_chromosomes),  
                "extraHardParticlesIdxs": [],
                "repulsionEnergy": repulsion_energy,  
                "attractionEnergy": attraction_energy,  
                #"repulsionRadius": repulsion_radius,
                #"attractionRadius": attraction_radius,
                #"selectiveRepulsionEnergy": selective_repulsion_energy,
                #"selectiveAttractionEnergy": selective_attraction_energy
                },
            except_bonds=True  
        )
        confinement_force = spherical_confinement(sim, k=5.0, density=0.07)
        sim.add_force(polymer_force)
        sim.add_force(confinement_force)

        sim.do_block(1000)  # Эквилибрация
        for _ in range(num_blocks):  # Итерация по блокам
            sim.do_block(block_steps)  # В каждом блоке по block_steps шагов итерации МД
        if save_block_conformation:
            write_pdb(sim.get_data(), result_dir, f'block_{run}.pdb', compartments, n_chromosomes)
        del sim
        gc.collect()


if __name__ == '__main__':
    main()
