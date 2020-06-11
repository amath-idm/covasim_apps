'''
Set up a simulation for New York.
'''

import covasim as cv


# Specify the key calibration parameters
pop_infected = 1000 # Number of seed infections
beta         = 0.016 # Overall transmissibility
symp_test    = 30 # Odds ratio of a person with symptoms testing
beta_days    = ['2020-03-15', '2020-04-01'] # Days social distancing changed
beta_changes = [0.7, 0.3] # Impact of social distancing

# Define the inputs
datafile = 'NY.csv'
pop_size = 100e3
pars = dict(
    pop_size = pop_size,
    pop_scale = 19.45e6/pop_size,
    pop_infected = pop_infected,
    pop_type = 'hybrid',
    beta = beta,
    start_day = '2020-02-01',
    end_day   = '2020-06-14',
    rescale = True,
    )

# Create the simulation
sim = cv.Sim(pars, datafile=datafile)

# Create the interventions
interventions = [
    cv.change_beta(days=beta_days, changes=beta_changes),
    cv.test_num(daily_tests=sim.data['new_tests'].dropna(), symp_test=symp_test),
    ]

# Run the simulation
sim['interventions'] = interventions
sim.run()

# Plotting
to_plot = ['cum_infections', 'new_infections', 'cum_diagnoses', 'new_diagnoses', 'cum_deaths', 'new_deaths']
sim.plot(to_plot=to_plot)

# For more detailed plotting
# sim.plot(to_plot='overview', fig_args=dict(figsize=(30,20)))

print('Done.')
