import pylab as pl
import auto_calibration as ac
import sciris as sc

state = 'CA'
until = '05-30'

cal = ac.Calibration(state, until)

do_plot = 1
do_save = 1
run_init = 0

pars, pkeys = cal.get_bounds() # Get parameter guesses

if run_init:
    print('Running initial...')
    sim = cal.create_sim(pars.best)
    sim.run()
    sim.plot(to_plot=cal.to_plot)
    pl.gcf().axes[0].set_title('Initial parameter values')
    cal.objective(pars.best)
    pl.pause(1.0) # Ensure it has time to render

print('Plotting result...')
pars_calib = cal.get_best_pars()
x = [pars_calib[k] for k in pkeys]
print(x)
sim = cal.create_sim(x)
sim = cal.run_msim(n_runs=3, n_cpus=3)
fit = sim.results.fit

if do_save:
    sc.savejson(f'covasim_calibration_v2_{cal.state}_{until}.json', pars_calib)

if do_plot:
    sim.plot(to_plot=cal.to_plot)
    pl.gcf().axes[0].set_title('Calibrated parameter values')
    fit.plot()