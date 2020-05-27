import numpy as np
import sciris as sc
import covasim as cv
import parestlib as pst
import load_data as ld

# Control verbosity
vb = sc.objdict()
vb.base    = 1
vb.extra   = 0
vb.plot    = 0
vb.verbose = 0

# Define and load the data
state = 'NY'
all_data = ld.load_data()
data = all_data[state]

cv.check_version('1.3.4', die=True)


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
        actual    = sim.data[key].values
        predicted = sim.results[key].values
        inds1 = sc.findinds(~np.isnan(actual))
        inds2 = sc.findinds(~np.isnan(predicted))
        inds = np.intersect1d(inds1, inds2)
        mismatch2 += pst.gof(actual[inds], predicted[inds], estimator='mean absolute')
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
        pop_infected = dict(best=1000,  lb=10,    ub=10000),
        beta         = dict(best=0.020, lb=0.008, ub=0.025),
        beta_day     = dict(best=45,    lb=15,    ub=60),
        beta_change  = dict(best=0.33,  lb=0.1,   ub=0.9),
        symp_test    = dict(best=100,   lb=10,    ub=1000),
    )

    # Convert from dicts to arrays
    pars = sc.objdict()
    for key in ['best', 'lb', 'ub']:
        pars[key] = np.array([v[key] for v in pdict.values()])

    return pars, pdict.keys()


def calibrate(state):
    ''' Perform the calibration '''

    pars, pkeys = get_bounds() # Get parameter guesses
    shest = pst.ShellStep(objective, pars.best, pars.lb, pars.ub, optimum='min', maxiters=10, mp={'N':8}) # Create object
    output = shest.optimize() # Perform optimization
    output.pdict = sc.objdict({k:v for k,v in zip(pkeys, output.x)}) # Convert to a dict

    return output


if __name__ == '__main__':

    T = sc.tic()
    output = calibrate('NY')
    sc.toc(T)

    sim = create_sim(output.pdict.values())
    sim.run()
    sim.plot(to_plot='overview')


print('Done.')