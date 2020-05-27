import numpy   as np; np
import pylab   as pl; pl
import pandas  as pd; pd
import sciris  as sc; sc
import covasim as cv; cv


#%% Initialize

mapping = {
    'IL':'Illinois',
    'MA':'Massachusetts',
    'MI':'Michigan',
    'NJ':'New Jersey',
    'NY':'New York',
    'PA':'Pennsylvania',
}
folder = 'data'

states = list(mapping.keys())


#%% Get the raw data
data = sc.objdict()
for state in states:
    filename = f'{folder}/{state}.csv'
    data[state] = pd.read_csv(filename)


# From https://github.com/covid-modeling/covasim-connector/blob/master/runsim.py
raw_pop_sizes = pd.read_csv("http://www2.census.gov/programs-surveys/popest/datasets/2010-2019/national/totals/nst-est2019-alldata.csv")

popsizes = sc.objdict()
for state in states:
    popsizes[state] = int(raw_pop_sizes[raw_pop_sizes.NAME == mapping[state]].POPESTIMATE2019)

print('Done.')