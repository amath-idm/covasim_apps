import pylab as pl
import auto_calibration as ac

# State set in auto_calibration.py
do_plot = 1
run_init = 0

pars, pkeys = ac.get_bounds() # Get parameter guesses

if run_init:
    print('Running initial...')
    sim = ac.create_sim(pars.best)
    sim.run()
    sim.plot(to_plot=ac.to_plot)
    pl.gcf().axes[0].set_title('Initial parameter values')
    ac.objective(pars.best)
    pl.pause(1.0) # Ensure it has time to render

print('Plotting result...')
pars_calib = ac.get_best_pars()
x = [pars_calib[k] for k in pkeys]
print(x)
sim = ac.create_sim(x)
sim.run()
fit = sim.compute_fit()

if do_plot:
    sim.plot(to_plot=ac.to_plot)
    pl.gcf().axes[0].set_title('Calibrated parameter values')
    fit.plot()