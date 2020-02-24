import sys
from functools import partial
import pandas as pd
from scipy import stats
from catboost import CatBoostRegressor as reg
import numpy as np

def load_tasks(tasks_filename):
    return pd.read_csv(tasks_filename, sep="\t")

def user_ages_median(tasks,users):
    user_ids = tasks.user_ids.values
    for i in range(len(tasks.values)):
        user_ids[i] = user_ids[i].split(',')
        user_ids[i] = [int(user_ids[i][j]) for j in range(len(user_ids[i]))]
    for i in range(len(tasks.values)):
        for j in range(len(user_ids[i])):
            user_ids[i][j] = users.age.values[user_ids[i][j]]
    
    user_ages_median = [np.median(user_ids[i]) for i in range(len(tasks.values))]
    
    return user_ages_median

def main():
    tests_filename = sys.argv[1]
    tasks = load_tasks(tests_filename)
    users = load_tasks('users.tsv')
    pub_count = [len(tasks.publishers[i].split(',')) for i in range(len(tasks))]
    median_ages = user_ages_median(tasks = tasks,users=users)
    tasks = tasks[['cpm','hour_start','hour_end','audience_size']]
    tasks['delta_hour'] = tasks['hour_end'] - tasks['hour_start']
    tasks['cpm*hour'] = tasks['cpm']*tasks['delta_hour']
    tasks['cpm*pub'] = tasks['cpm']*pd.Series(pub_count)
    tasks['median_age'] = pd.Series(median_ages)


    tasks = tasks[['hour_start','audience_size','delta_hour','cpm*hour','cpm*pub','median_age']]

    model = reg().load_model('model.cbm')
    pd.DataFrame(data = abs(model.predict(tasks)),columns = [['at_least_one', 'at_least_two', 'at_least_three']]).to_csv(sys.stdout, sep="\t", index=False, header=True)

if __name__ == '__main__':
    main()

