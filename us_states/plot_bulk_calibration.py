import pylab as pl
import covasim as cv
import auto_calibration as ac

do_save = 1
do_plot = 0
n_runs  = 1

for until in ['05-30', '04-30']:

    for state in ['CA', 'IL', 'MA', 'MI', 'NJ', 'NY']:

        cal = ac.Calibration(state, until)

        if do_save:
            cal.save()

        if do_plot:
            print('Plotting result...')
            pars = cal.get_best_pars()
            sim = cal.create_sim(pars)
            sim = cal.run_msim(n_runs=n_runs, n_cpus=n_runs)
            fit = sim.results.fit
            fit.plot()
            pl.gcf().axes[0].set_title(f'Calibration for {state} until {until}', fontweight='bold')
            if do_save:
                cv.savefig(f'calibration_{until}_{state}.png', dpi=75)