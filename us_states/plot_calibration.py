import pylab as pl
import auto_calibration as ac

print('Running initial...')
pars, pkeys = ac.get_bounds() # Get parameter guesses
sim = ac.create_sim(pars.best)
sim.run()
sim.plot(to_plot=ac.to_plot)
pl.gcf().axes[0].set_title('Initial parameter values')
ac.objective(pars.best)
pl.pause(1.0) # Ensure it has time to render

print('Plotting result...')
pars_calib = ac.get_best_pars()
x = [pars_calib[k] for k in pkeys]
sim = ac.create_sim(x)
sim.run()
sim.plot(to_plot=ac.to_plot)
pl.gcf().axes[0].set_title('Calibrated parameter values')