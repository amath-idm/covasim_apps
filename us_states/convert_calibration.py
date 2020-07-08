import numpy as np
import pandas as pd
import sciris as sc
import auto_calibration as ac

state = 'NY'
until = '05-30'

cal = ac.Calibration(state, until)

#%% Load and analyze the data
sc.heading('Loading data...')
best, pkeys = cal.get_bounds()
best = best['best']
study = cal.load_study()

sc.heading('Making results structure...')
results = []
n_trials = len(study.trials)
failed_trials = []
for trial in study.trials:
    data = {'index':trial.number, 'mismatch': trial.value}
    for key,val in trial.params.items():
        data[key] = val
    if data['mismatch'] is None:
        failed_trials.append(data['index'])
    else:
        results.append(data)
print(f'Processed {len(study.trials)} trials; {len(failed_trials)} failed')

sc.heading('Making data structure...')
keys = ['index', 'mismatch'] + pkeys
print(keys)
data = sc.objdict().make(keys=keys, vals=[])
for i,r in enumerate(results):
    for key in keys:
        if key not in r:
            print(f'Warning! Key {key} is missing from trial {i}, replacing with default')
            r[key] = best[key]
        data[key].append(r[key])
df = pd.DataFrame.from_dict(data)


sc.heading('Saving...')

# Save data to JSON
order = np.argsort(df['mismatch'])
json = []
for o in order:
    row = df.iloc[o,:].to_dict()
    rowdict = dict(index=row.pop('index'), mismatch=row.pop('mismatch'), pars={})
    for key,val in row.items():
        rowdict['pars'][key] = val
    json.append(rowdict)
sc.savejson(f'{state}-processed.json', json, indent=2)
saveobj = False
