import numpy as np
import pylab as pl
import sciris as sc
import covasim as cv
import scipy as sp
import optuna as op
import load_data as ld

# Saving and running
until = '05-30' # Note, update end day manually
do_save   = 1
name      = 'covasim'
n_trials  = 100
n_workers = 36
cv.check_version('1.5.1', die=True) # Ensure Covasim version is correct

# Control verbosity
vb = sc.objdict()
vb.base    = 0
vb.extra   = 0
vb.plot    = 0
vb.verbose = 0
to_plot = ['cum_infections', 'new_infections', 'cum_tests', 'new_tests', 'cum_diagnoses', 'new_diagnoses', 'cum_deaths', 'new_deaths']


def storage_path(st=None):
    if st: state = st
    return f'sqlite:///opt_v2_{until}_{state}.db'

def create_sim(x, vb=vb, st=None):
    ''' Create the simulation from the parameters '''

    if st: state = st

    # Define and load the data
    all_data = ld.load_data()
    data     = all_data[state]

    # Convert parameters
    pop_infected = x[0]
    beta         = x[1]
    beta_day     = x[2]
    beta_change  = x[3]
    symp_test    = x[4]

    # Create parameters
    pop_size = 200e3
    pars = dict(
        pop_size     = pop_size,
        pop_scale    = data.popsize/pop_size,
        pop_infected = pop_infected,
        beta         = beta,
        start_day    = '2020-03-01',
        end_day      = f'2020-05-30', # Change final day here
        rescale      = True,
        verbose      = vb.verbose,
    )

    #Create the sim
    sim = cv.Sim(pars, datafile=data.epi)

    # Add interventions
    interventions = [
        cv.change_beta(days=beta_day, changes=beta_change),
        cv.test_num(daily_tests=sim.data['new_tests'].dropna(), symp_test=symp_test),
        ]

    # Update
    sim.update_pars(interventions=interventions)

    return sim


def smooth(y, sigma=3):
    return sp.ndimage.gaussian_filter1d(y, sigma=sigma)


def run_msim(sim, n_runs=3, n_cpus=1, new_deaths=True):
    msim = cv.MultiSim(base_sim=sim)
    msim.run(n_runs=n_runs, n_cpus=n_cpus)
    sim = msim.reduce(use_mean=True, output=True)
    if new_deaths:
        offset = cv.daydiff(sim['start_day'], sim.data['date'][0])
        d_data = smooth(sim.data['new_deaths'].values)
        d_sim  = smooth(sim.results['new_deaths'].values[offset:])
        minlen = min(len(d_data), len(d_sim))
        d_data = d_data[:minlen]
        d_sim = d_sim[:minlen]
        deaths = {'deaths':dict(data=d_data, sim=d_sim, weights=1)}
        sim.compute_fit(custom=deaths, keys=['cum_diagnoses', 'cum_deaths'], weights={'cum_diagnoses':0.2, 'cum_deaths':0.2}, output=False)
    else:
        sim.compute_fit(output=False)
    return sim


def objective(x, vb=vb):
    ''' Define the objective function we are trying to minimize '''

    # Create and run the sim
    sim = create_sim(x=x, vb=vb)
    sim = run_msim(sim)
    return sim.results.fit.mismatch


def get_bounds():
    ''' Set parameter starting points and bounds '''
    pdict = sc.objdict(
        pop_infected = dict(best=10000,  lb=1000,   ub=50000),
        beta         = dict(best=0.015, lb=0.007, ub=0.020),
        beta_day     = dict(best=20,    lb=5,     ub=60),
        beta_change  = dict(best=0.5,   lb=0.2,   ub=0.9),
        symp_test    = dict(best=30,   lb=5,    ub=200),
    )

    # Convert from dicts to arrays
    pars = sc.objdict()
    for key in ['best', 'lb', 'ub']:
        pars[key] = np.array([v[key] for v in pdict.values()])

    return pars, pdict.keys()


#%% Calibration

def op_objective(trial):

    pars, pkeys = get_bounds() # Get parameter guesses
    x = np.zeros(len(pkeys))
    for k,key in enumerate(pkeys):
        x[k] = trial.suggest_uniform(key, pars.lb[k], pars.ub[k])

    return objective(x)

def worker():
    study = op.load_study(storage=storage_path(), study_name=name)
    return study.optimize(op_objective, n_trials=n_trials)


def run_workers():
    return sc.parallelize(worker, n_workers)


def make_study():
    try: op.delete_study(storage=storage_path(), study_name=name)
    except: pass
    return op.create_study(storage=storage_path(), study_name=name)


def load_study(st):
    if st: state = st
    return op.load_study(storage=storage_path(state), study_name=name)


def get_best_pars(st=None):
    if st: state = st
    study = load_study(state)
    output = study.best_params
    return output


def calibrate():
    ''' Perform the calibration '''
    make_study()
    run_workers()
    output = get_best_pars()
    return output


if __name__ == '__main__':

    for state in ['CA', 'IL', 'MA', 'MI', 'NJ', 'NY']:

        # Plot initial
        if vb.plot:
            print('Running initial...')
            pars, pkeys = get_bounds() # Get parameter guesses
            sim = create_sim(pars.best)
            sim.run()
            sim.plot(to_plot=to_plot)
            pl.gcf().axes[0].set_title('Initial parameter values')
            objective(pars.best)
            pl.pause(1.0) # Ensure it has time to render

        # Calibrate
        print(f'Starting calibration for {state}...')
        T = sc.tic()
        pars_calib = calibrate()
        sc.toc(T)

        if do_save:
            sc.savejson(f'calibrated_parameters_v2_{state}.json', pars_calib)

        # Plot result
        if vb.plot:
            print('Plotting result...')
            x = [pars_calib[k] for k in pkeys]
            sim = create_sim(x)
            sim.run()
            sim.plot(to_plot=to_plot)
            pl.gcf().axes[0].set_title('Calibrated parameter values')




print('Done.')
