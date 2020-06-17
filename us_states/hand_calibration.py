import covasim as cv
import load_data as ld

state = 'CA'

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
    cv.change_beta(days=20, changes=0.50),
    cv.test_num(daily_tests=sim.data['new_tests'].dropna(), symp_test=15),
    ]

sim.update_pars(interventions=interventions)
msim = cv.MultiSim(sim)
msim.run(n_runs=5)
# sim.run()
msim.plot(to_plot=['cum_infections', 'new_infections', 'cum_diagnoses', 'new_diagnoses', 'cum_deaths', 'new_deaths'], n_cols=2)
# sim.plot(to_plot='overview', scatter_args=dict(alpha=0.1), fig_args=dict(figsize=(30,20)))

print('Done.')