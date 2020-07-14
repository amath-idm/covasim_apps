import pylab as pl
import auto_calibration as ac

state = 'CA'
until = '05-30'

cal = ac.Calibration(state, until)

do_plot = 1
do_save = 0
run_init = 0
n_runs = 1

# Manual
x = None
# x = [8903, 0.009234690304264684, 41.956685173981896, 0.65, 18.300745581426092]

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
if x is None:
    pars_calib = cal.get_best_pars()
    x = [pars_calib[k] for k in pkeys]
print(x)
sim = cal.create_sim(x)
sim = cal.run_msim(n_runs=n_runs) # , n_cpus=n_runs # For 1.5.1
fit = sim.results.fit

if do_save:
    cal.save()

if do_plot:
    sim.plot(to_plot=cal.to_plot)
    pl.gcf().axes[0].set_title('Calibrated parameter values')
    fit.plot()