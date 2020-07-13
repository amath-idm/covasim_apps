import numpy as np
import pylab as pl
import sciris as sc
import covasim as cv
import scipy as sp
import optuna as op
import load_data as ld



class Calibration:


    def __init__(self, state, until):
        self.state = state
        self.until = until

        # Saving and running
        self.do_save   = 1
        self.name      = 'covasim'
        self.n_trials  = 50
        self.n_workers = 36
        self.storage = f'sqlite:///opt_v3_{until}_{self.state}.db'

        cv.check_version('1.5.1', die=True) # Ensure Covasim version is correct

        # Control verbosity
        vb = sc.objdict()
        vb.base    = 0
        vb.extra   = 0
        vb.plot    = 0
        vb.verbose = 0
        self.vb = vb
        self.to_plot = ['cum_infections', 'new_infections', 'cum_tests', 'new_tests', 'cum_diagnoses', 'new_diagnoses', 'cum_deaths', 'new_deaths']


    def create_sim(self, x):
        ''' Create the simulation from the parameters '''

        # Define and load the data
        all_data = ld.load_data()
        data     = all_data[self.state]
        self.x   = x

        # Convert parameters
        pop_infected = x[0]
        beta         = x[1]
        beta_day     = x[2]
        beta_change  = x[3]
        symp_test    = x[4]

        if self.until:
            end_day = f'2020-{self.until}' # Change final day here
        else:
            end_day = f'2020-05-30' # Change final day here

        # Create parameters
        pop_size = 200e3
        pars = dict(
            pop_size     = pop_size,
            pop_scale    = data.popsize/pop_size,
            pop_infected = pop_infected,
            beta         = beta,
            start_day    = '2020-03-01',
            end_day      = end_day,
            rescale      = True,
            verbose      = self.vb.verbose,
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

        self.sim = sim

        return sim


    def smooth(self, y, sigma=3):
        return sp.ndimage.gaussian_filter1d(y, sigma=sigma)


    def run_msim(self, n_runs=1, n_cpus=1, new_deaths=True):
        msim = cv.MultiSim(base_sim=self.sim)
        msim.run(n_runs=n_runs, n_cpus=n_cpus)
        sim = msim.reduce(use_mean=True, output=True)
        if new_deaths:
            offset = cv.daydiff(sim['start_day'], sim.data['date'][0])
            d_data = self.smooth(sim.data['new_deaths'].values)
            d_sim  = self.smooth(sim.results['new_deaths'].values[offset:])
            minlen = min(len(d_data), len(d_sim))
            d_data = d_data[:minlen]
            d_sim = d_sim[:minlen]
            deaths = {'deaths':dict(data=d_data, sim=d_sim, weights=1)}
            sim.compute_fit(custom=deaths, keys=['cum_diagnoses', 'cum_deaths'], weights={'cum_diagnoses':0.2, 'cum_deaths':0.2}, output=False)
        else:
            sim.compute_fit(output=False)

        self.sim = sim
        self.mismatch = sim.results.fit.mismatch

        return sim


    def objective(self, x):
        ''' Define the objective function we are trying to minimize '''

        # Create and run the sim
        self.create_sim(x=x)
        self.run_msim()
        return self.mismatch


    def get_bounds(self):
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

    def op_objective(self, trial):

        pars, pkeys = self.get_bounds() # Get parameter guesses
        x = np.zeros(len(pkeys))
        for k,key in enumerate(pkeys):
            x[k] = trial.suggest_uniform(key, pars.lb[k], pars.ub[k])

        return self.objective(x)

    def worker(self):
        study = op.load_study(storage=self.storage, study_name=self.name)
        return study.optimize(self.op_objective, n_trials=self.n_trials)


    def run_workers(self):
        return sc.parallelize(self.worker, self.n_workers)


    def make_study(self):
        try: op.delete_study(storage=self.storage, study_name=self.name)
        except: pass
        return op.create_study(storage=self.storage, study_name=self.name)


    def load_study(self):
        return op.load_study(storage=self.storage, study_name=self.name)


    def get_best_pars(self):
        study = self.load_study()
        output = study.best_params
        return output


    def calibrate(self):
        ''' Perform the calibration '''
        self.make_study()
        self.run_workers()
        output = self.get_best_pars()
        return output


if __name__ == '__main__':

    for until in ['04-30']: # ['05-30', '04-30']

        for state in ['MA, MI']: # ['CA', 'IL', 'MA', 'MI', 'NJ', 'NY']

            cal = Calibration(state, until)

            # Plot initial
            if cal.vb.plot:
                print('Running initial...')
                pars, pkeys = cal.get_bounds() # Get parameter guesses
                sim = cal.create_sim(pars.best)
                sim.run()
                sim.plot(to_plot=cal.to_plot)
                pl.gcf().axes[0].set_title('Initial parameter values')
                cal.objective(pars.best)
                pl.pause(1.0) # Ensure it has time to render

            # Calibrate
            print(f'Starting calibration for {cal.state}...')
            T = sc.tic()
            pars_calib = cal.calibrate()
            sc.toc(T)

            if cal.do_save:
                sc.savejson(f'calibrated_parameters_v2_{cal.state}.json', pars_calib)

            # Plot result
            if cal.vb.plot:
                print('Plotting result...')
                x = [pars_calib[k] for k in pkeys]
                sim = cal.create_sim(x)
                sim.run()
                sim.plot(to_plot=cal.to_plot)
                pl.gcf().axes[0].set_title('Calibrated parameter values')




print('Done.')
