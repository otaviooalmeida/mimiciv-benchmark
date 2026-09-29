import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle
import argparse
import json
pd.set_option('mode.chained_assignment', None)

parser = argparse.ArgumentParser(description='Convert extracted events to time series using as-of availability.')
parser.add_argument('--missing-availability-policy', choices=('exclude', 'measurement_time'), default='exclude',
                    help='Explicit sensitivity fallback for missing storetime; default excludes those events.')
args = parser.parse_args()

# Read extracted time series data, retaining storetime until each forecast cutoff is applied.
events = pd.read_csv('data/mimic_iv_events.csv', low_memory = False, usecols=['HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'STORETIME', 'VALUENUM', 'TABLE', 'NAME'])
icu = pd.read_csv('data/mimic_iv_icu.csv')
# Convert times to type datetime.
events.CHARTTIME = pd.to_datetime(events.CHARTTIME, errors='coerce')
events.STORETIME = pd.to_datetime(events.STORETIME, errors='coerce')
icu.INTIME = pd.to_datetime(icu.INTIME)
icu.OUTTIME = pd.to_datetime(icu.OUTTIME)

# Labs have no stay_id: assign by admission and ICU time interval, as before.
# Keep the stay_id already supplied by ICU tables. Drop unassignable rows.
icu['icustay_times'] = icu.apply(lambda x:[x.ICUSTAY_ID, x.INTIME, x.OUTTIME], axis=1)
adm_icu_times = icu.groupby('HADM_ID').agg({'icustay_times':list}).reset_index()
icu.drop(columns=['icustay_times'], inplace=True)
events = events.merge(adm_icu_times, on=['HADM_ID'], how='left')
idx = events.ICUSTAY_ID.isna()
tqdm.pandas()
def f(x):
    chart_time = x.CHARTTIME
    for icu_times in x.icustay_times:
        if icu_times[1]<=chart_time<=icu_times[2]:
            return icu_times[0]
events.loc[idx, 'ICUSTAY_ID'] = (events.loc[idx]).progress_apply(f, axis=1)
events.drop(columns=['icustay_times'], inplace=True)
events = events.loc[events.ICUSTAY_ID.notna()]
events.drop(columns=['HADM_ID'], inplace=True)

# Filter icu table.
icu = icu.loc[icu.ICUSTAY_ID.isin(events.ICUSTAY_ID)]

# Keep measurement and availability times distinct; missing storetime is never silently immediate.
events = events.merge(icu[['ICUSTAY_ID', 'INTIME']], on='ICUSTAY_ID', how='left')
events['rel_charttime'] = (events.CHARTTIME - events.INTIME).dt.total_seconds() / 60.0
events['rel_storetime'] = (events.STORETIME - events.INTIME).dt.total_seconds() / 60.0
missing_storetime = events.STORETIME.isna()
missing_charttime = events.CHARTTIME.isna()
storetime_before_charttime = events.STORETIME.notna() & events.CHARTTIME.notna() & (events.STORETIME < events.CHARTTIME)
source_missing = events.loc[missing_storetime].groupby(['TABLE', 'NAME'], dropna=False).size()
availability_report = {
    'input_rows_after_stay_linkage': int(len(events)),
    'missing_storetime_rows': int(missing_storetime.sum()),
    'missing_charttime_rows': int(missing_charttime.sum()),
    'storetime_before_charttime_rows': int(storetime_before_charttime.sum()),
    'excluded_missing_availability_rows': int((missing_storetime & ~missing_charttime).sum()) if args.missing_availability_policy == 'exclude' else 0,
    'excluded_storetime_before_measurement_rows': int(storetime_before_charttime.sum()),
    'missing_storetime_by_table_variable': {
        '{}|{}'.format(table, name): int(count) for (table, name), count in source_missing.items()
    },
    'missing_availability_policy': args.missing_availability_policy,
    'fallback_rows_included': int((missing_storetime & ~missing_charttime).sum()) if args.missing_availability_policy == 'measurement_time' else 0,
}
if args.missing_availability_policy == 'measurement_time':
    events.loc[missing_storetime, 'rel_storetime'] = events.loc[missing_storetime, 'rel_charttime']
with open('data/availability_audit.json', 'w', encoding='utf-8') as audit_file:
    json.dump(availability_report, audit_file, indent=2, ensure_ascii=False, allow_nan=False)
events['available_minute'] = events.rel_storetime
# Default operational population excludes events missing either time; the sensitivity fallback is explicit above.
events = events.loc[events.rel_charttime.notna() & events.available_minute.notna() & ~storetime_before_charttime].copy()
events['minute'] = np.floor(events.rel_charttime)
events.drop(columns=['INTIME', 'CHARTTIME', 'STORETIME', 'rel_charttime', 'rel_storetime'], inplace=True)

all_icustays = np.array(icu.ICUSTAY_ID)

# Get ts_ind.
def inv_list(x):
    d = {}
    for i in range(len(x)):
        d[x[i]] = i
    return d
icustay_to_ind = inv_list(all_icustays)
events['ts_ind'] = events.ICUSTAY_ID.map(icustay_to_ind)

# Rename some columns.
events.rename(columns={'rel_charttime':'minute', 'NAME':'variable', 'VALUENUM':'value'}, inplace=True)

# Add gender and age.
icu['ts_ind'] = icu.ICUSTAY_ID.map(icustay_to_ind)
data_age = icu[['ts_ind', 'AGE']]
data_age['variable'] = 'Age'
data_age.rename(columns={'AGE':'value'}, inplace=True)
data_gen = icu[['ts_ind', 'GENDER']]
data_gen.loc[data_gen.GENDER=='M', 'GENDER'] = 0
data_gen.loc[data_gen.GENDER=='F', 'GENDER'] = 1
data_gen['variable'] = 'Gender'
data_gen.rename(columns={'GENDER':'value'}, inplace=True)
data = pd.concat((data_age, data_gen), ignore_index=True)
data['minute'] = 0
data['available_minute'] = 0
events = pd.concat((data, events), ignore_index=True)

# Drop duplicate events.
events.drop_duplicates(inplace=True)

events = events.merge(icu[['ts_ind', 'HADM_ID', 'SUBJECT_ID']], on='ts_ind', how='left')
events.rename(columns={'HADM_ID':'hadm_id', 'SUBJECT_ID':'sub_id'}, inplace=True)

# Filter columns.
events = events[['ts_ind', 'minute', 'available_minute', 'variable', 'value', 'hadm_id', 'sub_id']]

# Preserve distinct availability times so future cutoffs cannot see late-charted observations.
events['value'] = events['value'].astype(float)
events = events.groupby(['ts_ind', 'minute', 'variable', 'available_minute']).agg(
    {'value':'mean', 'hadm_id':'first', 'sub_id':'first'}
).reset_index()

# Get variable indices.
static_varis = ['Age', 'Gender']
ii = events.variable.isin(static_varis)
events = events.loc[~ii]
var = sorted(list(set(events.variable)))
def inv_list(l):
    d = {}
    for i in range(len(l)):
        d[l[i]] = i
    return d
var_to_ind = inv_list(var)

# Peripheral pulse oximetry is the target; blood-gas SO2 remains a separate covariate.
target_names = ['HR', 'SBP', 'DBP', 'Temperature', 'SpO2_peripheral']
missing_targets = [name for name in target_names if name not in var_to_ind]
if missing_targets:
    raise ValueError('Target variables missing from extracted events: {}'.format(missing_targets))
target_var = np.array([var_to_ind[name] for name in target_names])
events['vind'] = events.variable.map(var_to_ind)
pickle.dump([var, target_var], open('data/var.pkl','wb'))

# 20 threads split
sets = np.array_split(events, 20)
pickle.dump(sets, open('data/sets.pkl','wb'))
