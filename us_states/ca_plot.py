import covasim as cv
import load_data as ld

do_run = 0
do_save = 1
state = 'CA'
msimfile = 'ca_calibration.msim'
figfile  = 'ca_calibration.png'

all_data = ld.load_data()
data = all_data[state]

pop_size = 200e3
pars = dict(
    pop_size = pop_size,
    pop_scale = data.popsize/pop_size,
    pop_infected = 5000,
    beta = 0.015,
    start_day = '2020-03-01',
    end_day   = '2020-06-17',
    rescale = True,
    )


sim = cv.Sim(pars, datafile=data.epi)

interventions = [
    cv.change_beta(days=20, changes=0.49),
    cv.test_num(daily_tests=sim.data['new_tests'].dropna(), symp_test=17),
    ]

sim.update_pars(interventions=interventions)
if do_run:
    msim = cv.MultiSim(sim)
    msim.run(n_runs=20, par_args={'ncpus':5})
    msim.reduce()
    msim.save(msimfile)
else:
    msim = cv.load(msimfile)


#%% Plotting
for interv in msim.base_sim['interventions']:
    interv.do_plot = False

to_plot = ['cum_diagnoses', 'new_diagnoses', 'cum_deaths', 'new_deaths']
fig_args = dict(figsize=(18,18))
scatter_args = dict(alpha=0.3, marker='o')
dateformat = '%d %b'

fig = msim.plot(to_plot=to_plot, n_cols=1, fig_args=fig_args, scatter_args=scatter_args, dateformat=dateformat)
fit = msim.base_sim.compute_fit(weights={'cum_diagnoses':1, 'cum_deaths':1})
print('Average daily mismatch: ', fit.mismatch/msim.base_sim['n_days']/2*100)

for ax in fig.axes:
    ax.legend(['Model', '80% modeled interval', 'Data'], loc='upper left')

if do_save:
    cv.savefig(figfile)

print('Done.')