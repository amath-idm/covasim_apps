import numpy   as np; np
import pylab   as pl; pl
import pandas  as pd; pd
import sciris  as sc; sc
import covasim as cv; cv

states = ['IL', 'MA', 'MI', 'NJ', 'NY', 'PA']
folder = 'data'

raw = sc.objdict()
for state in states:
    filename = f'{folder}/{state}.csv'
    raw[state] = pd.read_csv(filename)




print('Done.')