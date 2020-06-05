'''
Set up a simulation for Burkina.
'''

import covasim as cv


# Specify the key calibration parameters
pop_infected = 100 # Number of seed infections
beta         = 0.009 # Overall transmissibility
symp_prob    = 0.005 # Probability of a person with symptoms testing
beta_day     = '2020-04-01' # Day social distancing began
beta_change  = 0.35 # Impact of social distancing

# Define the inputs
datafile = 'Burkina_Faso.csv'
pop_size = 100e3
pars = dict(
    pop_size = pop_size,
    pop_scale = 19.75e6/pop_size, # Burkina population
    pop_infected = pop_infected,
    pop_type = 'hybrid',
    beta = beta,
    start_day = '2020-02-01',
    end_day   = '2020-06-05',
    rescale = True,
    location = 'Burkina Faso',
    )

# Create the interventions
interventions = [
    cv.change_beta(days=beta_day, changes=beta_change),
    cv.test_prob(start_day='2020-03-04', symp_prob=symp_prob),
    ]

# Create and run the simulation
sim = cv.Sim(pars, interventions=interventions, datafile=datafile)
sim.run()

# Plotting
to_plot = ['cum_infections', 'new_infections', 'cum_diagnoses', 'new_diagnoses', 'cum_deaths', 'new_deaths']
sim.plot(to_plot=to_plot)

print('Done.')