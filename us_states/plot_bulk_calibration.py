import pylab as pl
import covasim as cv
import auto_calibration as ac

do_save = 0
do_plot = 0
n_runs  = 1

for until in ['05-30', '04-30']:

    for state in ['CA', 'IL', 'MA', 'MI', 'NJ', 'NY']:

        print(f'Working on {state} until {until}...')

        cal = ac.Calibration(state, until)

        if do_save:
            cal.save()

        pars = cal.get_best_pars()
        if do_plot:
            print('Plotting result...')
            sim = cal.create_sim(pars)
            sim = cal.run_msim(n_runs=n_runs) # , n_cpus=n_runs # For 1.5.1
            fit = sim.results.fit
            fit.plot()
            pl.gcf().axes[0].set_title(f'Calibration for {state} until {until}', fontweight='bold')
            if do_save:
                cv.savefig(f'calibration_{until}_{state}.png', dpi=75)