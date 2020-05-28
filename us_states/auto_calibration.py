import numpy as np
import sciris as sc
import covasim as cv
import parestlib as pst
import optuna as op
import load_data as ld

# Control verbosity
vb = sc.objdict()
vb.base    = 0
vb.extra   = 0
vb.plot    = 0
vb.verbose = 0

# Define and load the data
state = 'NJ'
all_data = ld.load_data()
data = all_data[state]

cv.check_version('1.4.0', die=True)


def create_sim(x, vb=vb):
    ''' Create the simulation from the parameters '''

    # Convert parameters
    pop_infected = x[0]
    beta         = x[1]
    beta_day     = x[2]
    beta_change  = x[3]
    symp_test    = x[4]

    # Create parameters
    pop_size = 100e3
    pars = dict(
        pop_size     = pop_size,
        pop_scale    = data.popsize/pop_size,
        pop_infected = pop_infected,
        beta         = beta,
        start_day    = '2020-02-01',
        end_day      = '2020-06-01',
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


def objective(x, vb=vb):
    ''' Define the objective function we are trying to minimize '''

    # Set the weights for the data
    weights = dict(
        cum_deaths=10,
        cum_diagnoses=5,
        new_deaths=0,
        new_diagnoses=0,
        cum_severe=0,
        new_severe=0,
    )

    # Create and run the sim
    sim = create_sim(x=x, vb=vb)
    sim.run()

    # Two methods for calculating mismtach
    mismatch1 = -sim.compute_likelihood(weights=weights) # Built in mismatch
    mismatch2 = 0 # Custom mismatch
    for key,wt in weights.items():
        if wt:
            actual    = sim.data[key].values
            predicted = sim.results[key].values[sim.day(sim.data.date[0]):]
            inds1 = sc.findinds(~np.isnan(actual))
            inds2 = sc.findinds(~np.isnan(predicted))
            inds = np.intersect1d(inds1, inds2)
            mismatch2 += wt*pst.gof(actual[inds], predicted[inds], estimator='median fractional')
    mismatch = [mismatch1, mismatch2][1] # Choose which mismatch to use

    # Optionally show detail
    if vb.base:
        print(f'Mismatch {mismatch}, pars: {x}')
    if vb.extra:
        print('Summary:')
        print(sim.summary)
    if vb.plot:
        sim.plot(to_plot='overview', scatter_args=dict(alpha=0.1), fig_args=dict(figsize=(30,20)))

    return mismatch


def get_bounds():
    ''' Set parameter starting points and bounds '''
    pdict = sc.objdict(
        pop_infected = dict(best=100,   lb=10,    ub=5000),
        beta         = dict(best=0.015, lb=0.008, ub=0.025),
        beta_day     = dict(best=45,    lb=30,    ub=60),
        beta_change  = dict(best=0.5,   lb=0.2,   ub=0.8),
        symp_test    = dict(best=100,   lb=20,    ub=500),
    )

    # Convert from dicts to arrays
    pars = sc.objdict()
    for key in ['best', 'lb', 'ub']:
        pars[key] = np.array([v[key] for v in pdict.values()])

    return pars, pdict.keys()


#%% Calibration

name      = 'optuna'
storage   = f'sqlite:///{name}.db'
n_trials  = 20
n_workers = 8

pars, pkeys = get_bounds() # Get parameter guesses


def op_objective(trial):

    pars, pkeys = get_bounds() # Get parameter guesses
    x = np.zeros(len(pkeys))
    for k,key in enumerate(pkeys):
        x[k] = trial.suggest_uniform(key, pars.lb[k], pars.ub[k])

    return objective(x)

def worker():
    study = op.load_study(storage=storage, study_name=name)
    return study.optimize(op_objective, n_trials=n_trials)


def run_workers():
    return sc.parallelize(worker, n_workers)


def make_study():
    try: op.delete_study(storage=storage, study_name=name)
    except: pass
    return op.create_study(storage=storage, study_name=name)


def calibrate():
    ''' Perform the calibration '''
    make_study()
    run_workers()
    study = op.load_study(storage=storage, study_name=name)
    output = study.best_params
    return output


if __name__ == '__main__':

    # # Plot initial
    pars, pkeys = get_bounds() # Get parameter guesses
    # sim = create_sim(pars.best)
    # sim.run()
    # sim.plot(to_plot='overview')
    # objective(pars.best)

    # Calibrate
    T = sc.tic()
    output = calibrate()
    sc.toc(T)

    # Plot result
    x = [output[k] for k in pkeys]
    sim = create_sim(x)
    sim.run()
    sim.plot(to_plot=['cum_infections', 'new_infections', 'cum_diagnoses', 'new_diagnoses', 'cum_deaths', 'new_deaths'], n_cols=2)
    # sim.plot(to_plot='overview')


print('Done.')