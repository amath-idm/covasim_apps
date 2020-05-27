import covasim as cv
import load_data as ld

state = 'NY'

all_data = ld.load_data()
data = all_data[state]

pop_size = 100e3
pars = dict(
    pop_size = pop_size,
    pop_scale = data.popsize/pop_size,
    pop_infected = 1000,
    beta = 0.020,
    start_day = '2020-02-01',
    end_day   = '2020-06-01',
    rescale = True,
    )


sim = cv.Sim(pars, datafile=data.epi)

interventions = [
    cv.change_beta(days=45, changes=0.33),
    cv.test_num(daily_tests=sim.data['new_tests'].dropna(), symp_test=100),
    ]

sim.update_pars(interventions=interventions)
sim.run()
sim.plot(to_plot='overview', scatter_args=dict(alpha=0.1), fig_args=dict(figsize=(30,20)))

print('Done.')