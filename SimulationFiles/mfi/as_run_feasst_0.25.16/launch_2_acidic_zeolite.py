"""
particle types:
(0) OH <---> (2) O- (fixed)
(1) H2O <---> (3) H3O (fluid)
(4) framework (fixed, unreactive)

site types:
0: O in OH particle type 0 (AV pair 1)
1: H in OH particle type 0
2: O in H2O particle type 1 (AV pair 1)
3: H in H2O particle type 1
4: Dummy in H2O particle type 1
5: O in O- particle type 2 (AV pair 2)
6: Dummy in O- particle type 2
7: O in H30 particle type 3 (AV pair 2)
8: H in H30 particle type 3
9: framework
10: framework
11: framework

Start with 1 OH, num_particles H2O and framework
"""

import argparse
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyfeasst import fstio
from pyfeasst import macrostate_distribution
from pyfeasst import physical_constants
from pathlib import Path


def parse():
    # Parse arguments from command line or change their default values.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--feasst_install', type=str, default='../../../../build/',
                        help='FEASST install directory (e.g., the path to build)')
    parser.add_argument('--temperature', type=float, default=373, help='temperature (K)')
    parser.add_argument('--mu', type=float, default=-1000, help='chemical potential of fixed site HS (adsorbed)')
    parser.add_argument('--mu_init', type=float, default=10, help='initial chemical potential')
    parser.add_argument('--num_particles', type=int, default=1, help='minimum number of particles')
    #parser.add_argument('--num_particles', type=int, default=250, help='minimum number of particles')
    parser.add_argument('--weight', type=float, default=0.01, help='weight for rxn trials')
    parser.add_argument('--tpc', type=int, default=int(1e6), help='trials per cycle')
    parser.add_argument('--equilibration', type=int, default=1e4, help='number of cycles for equilibration')
    parser.add_argument('--production', type=int, default=1e6, help='number of cycles for production')
    parser.add_argument('--hours_checkpoint', type=float, default=1, help='hours per checkpoint')
    parser.add_argument('--hours_terminate', type=float, default=24*30, help='hours until termination')
    parser.add_argument('--procs_per_node', type=int, default=32, help='number of processors')
    parser.add_argument('--run_type', '-r', type=int, default=0,
                        help='0: run, 1: submit to queue, 2: post-process')
    parser.add_argument('--seed', type=int, default=-1,
                        help='Random number generator seed. If -1, assign random seed to each sim.')
    parser.add_argument('--max_restarts', type=int, default=0, help='Number of restarts in queue')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes in queue')
    parser.add_argument('--scratch', type=str, default=None,
                        help='Optionally write scheduled job to scratch/logname/jobid.')
    parser.add_argument('--node', type=int, default=0, help='node ID')
    parser.add_argument('--queue_id', type=int, default=-1, help='If != -1, read args from file')
    parser.add_argument('--queue_task', type=int, default=0, help='If > 0, restart from checkpoint')
    parser.add_argument('--out', type=str, default=None, help='Output directory for this run')

    # Convert arguments into a parameter dictionary, and add argument-dependent parameters.
    args, unknown_args = parser.parse_known_args()
    assert len(unknown_args) == 0, 'An unknown argument was included: '+str(unknown_args)
    params = vars(args)
    params['script'] = __file__
    params['prefix'] = 'acid_zeo'
    params['sim_id_file'] = params['prefix']+ '_sim_ids.txt'
    params['minutes'] = int(params['hours_terminate']*60) # minutes allocated on queue
    params['hours_terminate'] = 0.95*params['hours_terminate'] - 0.05 # terminate FEASST before SLURM  
    params['procs_per_sim'] = 1
    params['num_sims'] = params['procs_per_node']*params['num_nodes']
    params['beta'] = 1./(params['temperature']*physical_constants.MolarGasConstant().value()/1e3) # mol/kJ
    params['avb_outer'] = 3.7
    params['avb_inner'] = 2.4
    temp_tag = f"temp{int(args.temperature)}"
    nw_tag = f"{args.num_particles}w"
    mu_tag = f"mu_{abs(int(args.mu))}"

    if args.out:
        outdir = Path(args.out)
    else:
        outdir = Path('runs') / temp_tag / nw_tag / mu_tag

    outdir.mkdir(parents=True, exist_ok=True)
    params['outdir'] = str(outdir)

    
    params['assets_dir'] = str(Path(__file__).parent.resolve())

    base_prefix = params['prefix']                     # "acid_zeo"
    params['prefix'] = str(outdir / base_prefix)       # "…/runs/mu_xxx/acid_zeo"
    params['sim_id_file'] = params['prefix'] + '_sim_ids.txt'

    params['assets_dir'] = str(Path(__file__).parent.resolve())
    return params, args






def sim_node_dependent_params(params):
    """ Define parameters that are dependent on the sim or node. """
    params['sim_start'] = 0
    params['sim_end'] = params['num_sims'] - 1
    params['morph'] = params['sim'] % 2
    if params['morph'] == 1:
        params['half_weight'] = params['weight']/2.
        params['rxn']="""TrialMorph weight {half_weight} particle_type0 0 particle_type_morph0 2 particle_type1 1 particle_type_morph1 3 print_num_accepted true reference_index 0
TrialMorph weight {half_weight} particle_type0 2 particle_type_morph0 0 particle_type1 3 particle_type_morph1 1 print_num_accepted true reference_index 0""".format(**params)
        params['two_body_potential'] = """Potential Model ModelTwoBodyFactory model0 LennardJonesForceShift model1 ChargeScreened erfc_table_size 2e4""".format(**params)
    else:
        params['rxn'] = """TrialRXNAVB weight {weight} target_particle_type 0 target_particle_type_morph 2 particle_type 1 particle_type_morph 3 print_num_accepted true reference_index 0""".format(**params)
        params['two_body_potential'] = """NeighborCriteria site_type0 0 site_type1 2 site_type0_alt 5 site_type1_alt 7 maximum_distance {avb_outer} minimum_distance {avb_inner} potential_index 1
Potential Model ModelTwoBodyFactory model0 LennardJonesForceShift model1 ChargeScreened erfc_table_size 2e4 EnergyMap EnergyMapNeighborCriteria""".format(**params)

def write_feasst_script(params, script_file):
    """ Write fst script for a single simulation with keys of params {} enclosed. """
    with open(script_file, 'w', encoding='utf-8') as myfile:
        myfile.write("""
MonteCarlo
RandomMT19937 seed {seed}
Configuration side_length0 20.022 side_length1 19.899 side_length2 13.383 \
    particle_type0 {assets_dir}/oh.fstprt particle_type1 {assets_dir}/spce_with_dummy.fstprt \
    particle_type2 {assets_dir}/o-.fstprt particle_type3 {assets_dir}/h3o.fstprt \
    particle_type4 {assets_dir}/framework.fstprt particle_type5 {assets_dir}/block.fstprt \
    group0 mobile mobile_particle_type0 0 mobile_particle_type1 1 mobile_particle_type2 2 mobile_particle_type3 3 \
    group1 framework framework_particle_type0 4 \
    group2 gas gas_particle_type0 5 \
    add_particles_of_type0 1 add_particles_of_type4 1
Potential VisitModel Ewald tolerance 1e-4 tolerance_num_sites 2000
{two_body_potential}
Potential Model ChargeScreenedIntra VisitModel VisitModelBond
Potential Model ChargeSelf
Potential Model ModelHardShape shape_file {assets_dir}/MFI_Zeolite_111_1-54_blocks.txt cavity false group gas
ZeroBackground
RefPotential VisitModel DontVisitModel
ThermoParams beta {beta} chemical_potential0 {mu_init} chemical_potential1 {mu_init}
Metropolis
TrialTranslate weight_per_number_fraction 1 number_fraction_exclude_type0 4 particle_type 1
TrialTranslate weight_per_number_fraction 1 number_fraction_exclude_type0 4 particle_type 3
TrialParticlePivot weight_per_number_fraction 1 number_fraction_exclude_type0 4 particle_type 0
TrialParticlePivot weight_per_number_fraction 1 number_fraction_exclude_type0 4 particle_type 1
TrialParticlePivot weight_per_number_fraction 1 number_fraction_exclude_type0 4 particle_type 2
TrialParticlePivot weight_per_number_fraction 1 number_fraction_exclude_type0 4 particle_type 3
TrialSwapPosition weight 0.1 particle_types 1,3
CheckEnergy trials_per_update {tpc} decimal_places 6
Checkpoint checkpoint_file {prefix}{sim}_checkpoint.fst num_hours {hours_checkpoint} num_hours_terminate {hours_terminate}

# write framework once for visualization
Movie output_file {prefix}n{node}s{sim}_framework.xyz group framework clear_file true
WriteStepper analyze_name Movie
Remove name Movie

# gcmc initialization
TrialAdd particle_type 1
Log trials_per_write {tpc} output_file {prefix}n{node}s{sim}_fill.csv
Tune
Run until_num_particles {num_particles} particle_type 1
WriteStepper analyze_name Log
Remove name0 TrialAdd name1 Log

# equilibrate
ThermoParams beta {beta} chemical_potential0 0 chemical_potential1 0 chemical_potential2 0 chemical_potential3 {mu}
{rxn}
#FlatHistogram Macrostate MacrostateNumParticles particle_type 0 width 1 max 1 min 0 \
#    Bias WLTM min_sweeps 200 min_flatness 22 collect_flatness 18 min_collect_sweeps 1
#Bias TransitionMatrix min_sweeps 200
#CriteriaWriter  trials_per_write {tpc} output_file {prefix}n{node}s{sim:03d}_crit.csv
#CriteriaUpdater trials_per_update 1e5
## Implement a position swap (that works with rigid particles?)
Metropolis trials_per_cycle {tpc} cycles_to_complete {equilibration}
Log          trials_per_write {tpc} output_file {prefix}n{node}s{sim}_eq.csv
Movie        trials_per_write {tpc} output_file {prefix}n{node}s{sim}_eq.xyz group mobile
NumParticles trials_per_write {tpc} output_file {prefix}n{node}s{sim}_eq_num2.csv append true rewrite_header false particle_type 2
NumParticles trials_per_write {tpc} output_file {prefix}n{node}s{sim}_eq_num3.csv append true rewrite_header false particle_type 3
Run until complete
Remove name0 Tune name1 Log name2 Movie name3 NumParticles
#
## production
Metropolis trials_per_cycle {tpc} cycles_to_complete {production}
Log            trials_per_write {tpc} output_file {prefix}n{node}s{sim}.csv
Movie          trials_per_write {tpc} output_file {prefix}n{node}s{sim}.xyz group mobile
Energy         trials_per_write {tpc} output_file {prefix}n{node}s{sim}_en.csv append true rewrite_header false
NumParticles   trials_per_write {tpc} output_file {prefix}n{node}s{sim}_num2.csv append true rewrite_header false particle_type 2
NumParticles   trials_per_write {tpc} output_file {prefix}n{node}s{sim}_num3.csv append true rewrite_header false particle_type 3
CPUTime        trials_per_write {tpc} output_file {prefix}n{node}s{sim}_cpu.csv  append true
ProfileCPU     trials_per_write {tpc} output_file {prefix}n{node}s{sim}_profile.csv
Run until complete

# continue until all simulations on the node are complete
WriteFileAndCheck sim {sim} sim_start {sim_start} sim_end {sim_end} file_prefix {prefix}n{node}s file_suffix _finished.txt output_file {prefix}n{node}_terminate.txt
Run until_file_exists {prefix}n{node}_terminate.txt trials_per_file_check {tpc}
""".format(**params))

def post_process(params):
    import numpy as np
    import pandas as pd
    slopes=list()
    #drs = ['./', '../morph/']
    #for dr in drs:
    for sim in range(params['num_sims']):
        morph = sim % 2
        label=''
        color='blue'
        if morph == 1:
            color='red'
        if sim <= 1:
            label='rxnavb'
            if morph == 1:
                label='rxn'
        log = pd.read_csv('acid_zeon0s'+str(sim)+'.csv')
        #log = pd.read_csv('chemn0s0_num2.csv')
        #print(log['TrialMorph'])
        #print(log[log['num_particles_of_type2'] == 1])
        #en = pd.read_csv('chemn0s'+str(sim)+'_en.csv')
        #assert en['min'][0] == -1 # only one particle can interact with fixed ghost site
        cpu = pd.read_csv('acid_zeon0s'+str(sim)+'_cpu.csv', header=None, sep='\s+')
        num2 = pd.read_csv('acid_zeon0s'+str(sim)+'_num2.csv')
        import matplotlib.pyplot as plt
        efficiency=True
        efficiency=False
        if efficiency:
            # plot ln stdev vs ln time for efficiency z metric
            lntime = np.log(cpu[1])
            #print(time)
            #lnstdev = np.log(en['block_stdev'])
            lnstdev = np.log(num2['block_stdev'])
            #print(stdev)
            plt.plot(lntime, lnstdev, label=label, color=color)
            equil=int(1e2)
            def linear_fit(x, b):
                return -0.5*x + b
            from scipy.optimize import curve_fit
            popt, pcov = curve_fit(linear_fit, lntime[equil:], lnstdev[equil:])
            slopes.append(popt[0])
            plt.plot(lntime[equil:], linear_fit(lntime[equil:], popt[0]), label=label, color=color)
            #plt.plot(lntime[equil:], linear_fit(lntime[equil:], popt[0]), label=label+' '+str(slopes[-1]))
            plt.xlabel(r'$\ln t$ (CPU-h)', fontsize=16)
            plt.ylabel(r'$\ln \sigma_{n}$', fontsize=16)
        elif False:
        #elif True:
            # plot trial acceptance
            df = log
            x = df.index
            if morph == 1:
                y = df['TrialMorph']
            else:
                y = df['TrialRXNAVBHalf']
            #y = np.log(y)
            plt.gca().set_yscale('log')
            plt.plot(x, y, color=color, label=label)
            plt.xlabel(str(params['tpc'])+' MC trial', fontsize=16)
            plt.ylabel(r'RXN acceptance', fontsize=16)
        else:
            # plot <N_2>
            #markers, caps, bars = plt.errorbar(num2.index, num2['average'], num2['block_stdev'], label=label, color=color)
            #[bar.set_alpha(0.5) for bar in bars]
            df = num2
            x = df.index
            y = df['average']
            std = df['block_stdev']
            plt.plot(x, y, color=color)
            plt.fill_between(x, y - std, y + std, alpha=0.2, label=label, color=color)
            plt.xscale('log')
            #assert params['tpc'] == 1e6
            plt.xlabel(r'$10^6$ MC trials', fontsize=16)
            plt.ylabel(r'$\langle N_2\rangle$', fontsize=16)
    if efficiency:
        print(slopes, slopes[1::2], slopes[::2])
        z12 = np.exp(2*(np.average(slopes[1::2]) - np.average(slopes[::2])))
        #assert z12 > 1
        title = r'efficiency of RXNAVB compared to RXN:'+str(round(z12,3))
        print(title)
        plt.title(title)
    #plt.legend()
    #plt.show()
    plt.savefig(params['prefix']+'.png', transparent=True)
    #plt.savefig(params['prefix']+'.eps', transparent=True)
    #plt.savefig(params['prefix']+'.png', transparent=True, bbox_inches=True)

if __name__ == '__main__':
    parameters, arguments = parse()
    fstio.run_simulations(params=parameters,
                          sim_node_dependent_params=sim_node_dependent_params,
                          write_feasst_script=write_feasst_script,
                          post_process=post_process,
                          queue_function=fstio.slurm_single_node,
                          args=arguments)
