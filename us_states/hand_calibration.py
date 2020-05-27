import covasim as cv
import load_data as ld

state = 'NY'

data = ld.load_data()

sim = cv.Sim(datafile=data[state].epi)
sim.run()
sim.plot(to_plot='overview')

print('Done.')