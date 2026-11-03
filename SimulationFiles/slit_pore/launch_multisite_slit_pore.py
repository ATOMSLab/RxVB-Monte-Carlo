"""
Slit-pore reaction model with K fixed reaction sites (FEASST 0.25.21, TrialRxVB).

Fixed sites A (product C) react with fluid particles B (product D) through a
square-well potential. The square-well cutoff of every pair equals the
reaction-volume outer radius (--avb_outer), so changing --avb_outer also changes
the attractive range. Even simulations use TrialRxVB and odd simulations use
the unbiased TrialMorph, both with the same weight.

Site placement (--placement): flat puts every site at z = z_site (1.5, next to
the wall); spread places the sites across z in [-z_span, z_span], and a single
site at the pore centre (z = 0).

Example:
    python launch_multisite_slit_pore.py --tag dens --num_particles 450 --procs_per_job 32
"""
import argparse
import math
import numpy as np
import pandas as pd
from feasst import fstio


def xy_grid(k, L, min_sep=None):
    """K sites on a regular grid over the periodic L x L face, at least min_sep apart."""
    ncol = math.ceil(math.sqrt(k)); nrow = math.ceil(k / ncol)
    sx = L / ncol; sy = L / nrow
    xs = [(i - (ncol - 1) / 2.0) * sx for i in range(ncol)]
    ys = [(j - (nrow - 1) / 2.0) * sy for j in range(nrow)]
    if k > 1 and min_sep is not None:
        assert min(sx, sy) >= min_sep, (
            f"K={k}: site spacing min({sx:.2f},{sy:.2f}) < {min_sep:.2f} "
            "(reaction volumes would overlap) -- reduce K or enlarge the box")
    pts = []
    for j in range(nrow):
        for i in range(ncol):
            if len(pts) < k:
                pts.append((xs[i], ys[j]))
    return pts


def site_heights(k, mode, z_site, z_span):
    if mode == 'fixed':
        return [z_site] * k
    if k == 1:
        return [0.0]
    return [(-z_span + 2 * z_span * i / (k - 1)) for i in range(k)]


def parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--feasst_exe', type=str, default='feasst', help='FEASST executable')
    parser.add_argument('--num_sites', type=int, default=1, help='number of fixed reaction sites (A)')
    parser.add_argument('--placement', choices=['flat', 'spread'], default='flat')
    parser.add_argument('--tag', type=str, default='run',
                        help='label added to the output prefix')
    parser.add_argument('--spacing', type=float, default=3.0,
                        help='not used')
    parser.add_argument('--z_site', type=float, default=1.5,
                        help='site height for --placement flat')
    parser.add_argument('--z_span', type=float, default=1.5, help='spread mode: sites across [-z_span, z_span]')
    parser.add_argument('--avb_outer', type=float, default=1.05, help='reaction-volume outer radius and square-well cutoff')
    parser.add_argument('--avb_inner', type=float, default=1.0, help='reaction-volume inner radius')
    parser.add_argument('--beta', type=float, default=1., help='inverse temperature')
    parser.add_argument('--epsilonC_D', type=float, default=100)
    parser.add_argument('--delta_mu_rxn', type=float, default=-99, help='reaction chemical potential, applied to D')
    parser.add_argument('--mu_init', type=float, default=10, help='initial chemical potential')
    parser.add_argument('--num_particles', type=int, default=300, help='number of B fluid particles')
    parser.add_argument('--weight', type=float, default=0.0001,
                        help='weight of the reaction trial (TrialRxVB or TrialMorph)')
    parser.add_argument('--xy_side_length', type=float, default=12)
    parser.add_argument('--density', type=float, default=0.0,
                        help='if > 0, set the box length L from num_particles at this density')
    parser.add_argument('--z_slab_width', type=float, default=6)
    parser.add_argument('--padding', type=float, default=1.)
    parser.add_argument('--tpc', type=int, default=int(1e6), help='trials per cycle')
    parser.add_argument('--equilibration', type=int, default=1e2, help='cycles for equilibration')
    parser.add_argument('--production', type=int, default=1e4, help='cycles for production')
    parser.add_argument('--hours_checkpoint', type=float, default=1)
    parser.add_argument('--hours_terminate', type=float, default=24*7)
    parser.add_argument('--procs_per_job', type=int, default=32)
    parser.add_argument('--run_type', '-r', type=int, default=0, help='0: run, 1: queue, 2: post')
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--max_restarts', type=int, default=0)
    parser.add_argument('--num_jobs', type=int, default=1)
    parser.add_argument('--scratch', type=str, default=None)
    parser.add_argument('--job', type=int, default=0)
    parser.add_argument('--queue_id', type=int, default=-1)
    parser.add_argument('--queue_task', type=int, default=0)

    args, unknown_args = parser.parse_known_args()
    assert len(unknown_args) == 0, 'An unknown argument was included: '+str(unknown_args)
    params = vars(args)
    params['script'] = __file__
    # output prefix: tag, placement, K, N and avb_outer
    params['prefix'] = "{tag}_{placement}_k{ns}_n{n}_a{avb:g}_chemi".format(
        tag=params['tag'], placement=params['placement'], ns=params['num_sites'],
        n=params['num_particles'], avb=params['avb_outer'])
    params['sim_id_file'] = params['prefix']+ '_sim_ids.txt'
    params['minutes'] = int(params['hours_terminate']*60)
    params['hours_terminate'] = 0.95*params['hours_terminate'] - 0.05
    params['procs_per_sim'] = 1
    params['num_sims'] = params['procs_per_job']*params['num_jobs']
    params['half_z_slab_width'] = params['z_slab_width']/2.
    params['half_z_slab_width_minus_padding'] = params['half_z_slab_width'] - params['padding']
    # fixed density: L = sqrt(N / (density * accessible slab width))
    if params['density'] > 0:
        acc_z = 2.0 * params['half_z_slab_width_minus_padding']
        params['xy_side_length'] = (params['num_particles'] / (params['density'] * acc_z)) ** 0.5
    write_initial_xyz(params)
    return params, args


def write_initial_xyz(params):
    """Write the initial configuration (K fixed A sites) and the slab shape file."""
    xy = xy_grid(params['num_sites'], params['xy_side_length'],
                 min_sep=2.0 * params['avb_outer'])
    zs = site_heights(params['num_sites'], 'fixed' if params['placement'] == 'flat' else 'spread',
                      params['z_site'], params['z_span'])
    lines = [f"A {x:.6f} {y:.6f} {z:.6f}" for (x, y), z in zip(xy, zs)]
    with open(params['prefix']+'_initial.xyz', 'w', encoding='utf-8') as f:
        f.write("{n}\n-1 {L} {L} {Z} 0 0 0\n{body}".format(
            n=len(lines), L=params['xy_side_length'], Z=params['z_slab_width'],
            body="\n".join(lines)))
    with open(params['prefix']+'_shape_file.txt', 'w') as f:
        f.write("Slab dimension=2 bound0=-{b} bound1={b}".format(
            b=params['half_z_slab_width_minus_padding']))


def sim_job_dependent_params(params):
    params['sim_start'] = 0
    params['sim_end'] = params['num_sims'] - 1
    params['morph'] = params['sim'] % 2
    if params['morph'] == 1:
        params['rxn'] = """TrialMorph weight={weight} particle_type=A,B particle_type_morph=C,D print_num_accepted=true reference_index=0 write_weights=true""".format(**params)
        params['potential'] = """Potential Model=SquareWell VisitModel=VisitModelCell min_length=max_cutoff""".format(**params)
    else:
        params['rxn'] = """TrialRxVB weight={weight} target_particle_type=A target_site=A1 target_particle_type_morph=C particle_type=B site=B1 particle_type_morph=D print_num_accepted=true reference_index=0 write_weights=true""".format(**params)
        params['potential'] = """NeighborCriteria site_type0=A site_type1=B site_type0_alt=C site_type1_alt=D maximum_distance={avb_outer} minimum_distance={avb_inner}
Potential Model=SquareWell VisitModel=VisitModelCell min_length=max_cutoff EnergyMap=EnergyMapNeighborCriteria""".format(**params)


def write_feasst_script(params, script_file):
    with open(script_file, 'w', encoding='utf-8') as myfile:
        myfile.write("""
MonteCarlo
RandomMT19937 seed={seed}
# particle types: A <-> C (fixed sites); B <-> D (fluid)
Configuration particle_type=A:a.txt,B:b.txt,C:c.txt,D:d.txt \
    sigma={avb_inner} \
    epsilon=1 epsilonC_D={epsilonC_D} \
    cutoff={avb_outer} \
    group=fluid,product_fluid product_fluid_particle_type=D \
    fluid_particle_type=B,D \
    xyz_file={prefix}_initial.xyz
RefPotential VisitModel=DontVisitModel
{potential}
Potential Model=ModelHardShape shape_file={prefix}_shape_file.txt group=fluid
WriteModelParams output_file={prefix}_model_params.txt
ThermoParams beta={beta} chemical_potential={mu_init},{mu_init}
Metropolis
TrialTranslate weight_per_number_fraction=1 particle_type=B tunable_param=0.13
TrialTranslate weight_per_number_fraction=1 particle_type=D tunable_param=0.06
TrialSwapPosition weight=0.1 particle_types=B,D
CheckEnergy trials_per_update={tpc} decimal_places=6
Checkpoint checkpoint_file={prefix}{sim}_checkpoint.fst num_hours={hours_checkpoint} num_hours_terminate={hours_terminate}

# gcmc initialization: fill B to num_particles
Let [write]=trials_per_write={tpc} output_file={prefix}n{job}s{sim}
Log [write]_eq.csv
Tune
Run until_num_particles={num_particles} particle_type=B Trial=TrialAdd

# equilibrate
ThermoParams beta={beta} chemical_potential=0,0,0,{delta_mu_rxn}
{rxn}
Metropolis trials_per_cycle={tpc} cycles_to_complete={equilibration}
Run until=complete
Remove name=Tune,Log

# production
Metropolis trials_per_cycle={tpc} cycles_to_complete={production}
Log [write].csv
NumParticles [write]_num2.csv append=true rewrite_header=false particle_type=C
CPUTime [write]_cpu.csv append=true
Run until=complete

# continue until all simulations on the job are complete
WriteFileAndCheck sim={sim} sim_start={sim_start} sim_end={sim_end} file_prefix={prefix}n{job}s file_suffix=_finished.txt output_file={prefix}n{job}_terminate.txt
Run until_file_exists={prefix}n{job}_terminate.txt trials_per_file_check={tpc}
""".format(**params))


def post_process(params):
    pass


if __name__ == '__main__':
    parameters, arguments = parse()
    fstio.run_simulations(params=parameters,
                          sim_job_dependent_params=sim_job_dependent_params,
                          write_feasst_script=write_feasst_script,
                          post_process=post_process,
                          queue_function=fstio.slurm_single_job,
                          args=arguments)
