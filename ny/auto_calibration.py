import numpy as np
import pylab as pl
import sciris as sc
import covasim as cv
import optuna as op


cv.check_version('1.4.7')


def create_sim(x):
    ''' Create the simulation from the parameters '''

    # Convert parameters
    pop_infected = x[0]
    beta         = x[1]
    symp_test    = x[2]
    beta_change1 = x[3]
    beta_change2 = x[4]
    beta_days    = ['2020-03-15', '2020-04-01'] # Days social distancing changed

    # Define the inputs
    datafile = 'NY.csv'
    pop_size = 100e3
    pars = dict(
        pop_size = pop_size,
        pop_scale = 19.45e6/pop_size,
        pop_infected = pop_infected,
        pop_type = 'hybrid',
        beta = beta,
        start_day = '2020-02-01',
        end_day   = '2020-06-14',
        rescale = True,
        )

    # Create the simulation
    sim = cv.Sim(pars, datafile=datafile)

    # Create the interventions
    interventions = [
        cv.change_beta(days=beta_days, changes=[beta_change1, beta_change2]),
        cv.test_num(daily_tests=sim.data['new_tests'].dropna(), symp_test=symp_test),
        ]

    # Run the simulation
    sim['interventions'] = interventions

    return sim


def objective(x):
    ''' Define the objective function we are trying to minimize '''

    # Create and run the sim
    sim = create_sim(x=x)
    sim.run()
    fit = sim.compute_fit()

    return fit.mismatch


def get_bounds():
    ''' Set parameter starting points and bounds '''
    pdict = sc.objdict(
        pop_infected = dict(best=1000,  lb=500,   ub=2000),
        beta         = dict(best=0.016, lb=0.012, ub=0.018),
        symp_test    = dict(best=30,    lb=20,    ub=40),
        beta_change1 = dict(best=0.7,   lb=0.5,   ub=0.9),
        beta_change2 = dict(best=0.3,   lb=0.2,   ub=0.5),
    )

    # Convert from dicts to arrays
    pars = sc.objdict()
    for key in ['best', 'lb', 'ub']:
        pars[key] = np.array([v[key] for v in pdict.values()])

    return pars, pdict.keys()


#%% Calibration

name      = 'optuna_ny'
storage   = f'sqlite:///{name}.db'
n_trials  = 5
n_workers = 4

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

    do_save = True

    to_plot = ['cum_infections', 'new_infections', 'cum_tests', 'new_tests', 'cum_diagnoses', 'new_diagnoses', 'cum_deaths', 'new_deaths']

    # # Plot initial
    print('Running initial...')
    pars, pkeys = get_bounds() # Get parameter guesses
    sim = create_sim(pars.best)
    sim.run()
    fig = sim.plot(to_plot=to_plot)
    fig.axes[0].set_title('Initial parameter values')
    objective(pars.best)
    pl.pause(1.0) # Ensure it has time to render

    # Calibrate
    print('Starting calibration for {state}...')
    T = sc.tic()
    pars_calib = calibrate()
    sc.toc(T)

    # Plot result
    print('Plotting result...')
    x = [pars_calib[k] for k in pkeys]
    sim = create_sim(x)
    sim.run()
    fig = sim.plot(to_plot=to_plot)
    fig.axes[0].set_title('Calibrated parameter values')

    if do_save:
        sc.savejson(f'calibrated_parameters_ny.json', pars_calib)


print('Done.')