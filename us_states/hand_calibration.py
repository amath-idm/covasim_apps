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
    beta = 0.018,
    start_day = '2020-02-01',
    end_day   = '2020-06-01',
    rescale = True,
    )


sim = cv.Sim(pars, datafile=data.epi)
sim.run()
sim.plot(to_plot='overview', scatter_args=dict(alpha=0.1))

print('Done.')