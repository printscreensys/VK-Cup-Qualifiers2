import sys
from functools import partial
import pandas as pd
from scipy import stats
from catboost import CatBoostRegressor as reg
import numpy as np
import json
from math import log as ln

def load_tasks(tasks_filename):
    return pd.read_csv(tasks_filename, sep="\t")

def to_list(tasks, col, to_int):
    for i in range(len(tasks)):
        tasks[col][i] = tasks[col][i].split(',')
        if to_int:
            tasks[col][i] = [int(tasks[col][i][j]) for j in range(len(tasks[col][i]))]
    
    return tasks[col]

def user_ages_median(tasks):
    
    for i in range(len(tasks.values)):
        for j in range(len(user_ids[i])):
            user_ids[i][j] = users.age.values[user_ids[i][j]]
    
    user_ages_median = [np.median(user_ids[i]) for i in range(len(tasks.values))]
    
    return user_ages_median

def main():
    tests_filename = sys.argv[1]
    tasks = load_tasks(tests_filename)
    users = load_tasks('users.tsv')
    tasks['user_ids'] = to_list(tasks,'user_ids', to_int = True )
    tasks['publishers'] = to_list(tasks, 'publishers', to_int = False)

    pub_count = [len(tasks.publishers[i]) for i in range(len(tasks))]
    
    tasks['delta_hour'] = tasks['hour_end'] - tasks['hour_start']
    tasks['cpm*hour'] = tasks['cpm']*tasks['delta_hour']
    tasks['cpm*pub'] = tasks['cpm']*pd.Series(pub_count)

    median_ages = []
    for i in range(len(tasks)):
        median_ages.append(
            [users.age.values[tasks.user_ids[i][j]] for j in range(len(tasks.user_ids[i]))]
        )
    tasks['median_age'] = pd.Series([np.median(median_ages[i]) for i in range(len(tasks))])
    
    n_views = []
    user_pub = pd.read_csv('user_pub.csv')
    with open('user_views.json', 'r') as file:
        user_views = json.loads(file.read())
        
    n_views = []
    for i in range(len(tasks)):
        s = 0
        for j in range(len(tasks.user_ids[i])):
            try:
                s += user_views[str(tasks.user_ids[i][j])] + 1
            except KeyError:
                s += 1
        n_views.append(s/tasks.audience_size[i])
    
    tasks['n_views'] = pd.Series(n_views)
    
    tasks['n_views'] = pd.Series(n_views)+0.001
    tasks['views_per_pub'] = tasks['n_views']/pd.Series(pub_count)
    tasks = tasks[['hour_start','audience_size','delta_hour','cpm*hour','cpm*pub','median_age', 'n_views', 'views_per_pub']]

    p_win_reg = reg().load_model('pwin.cbm')
    tasks['p_win'] = pd.Series(p_win_reg.predict(tasks))
    tasks['1-p_reach'] = (1-tasks['p_win'])**(tasks['n_views'])

    model = reg().load_model('model.cbm')
    
    pd.DataFrame(data = abs(model.predict(tasks)),columns = [['at_least_one', 'at_least_two', 'at_least_three']]).to_csv(sys.stdout, sep="\t", index=False, header=True)

if __name__ == '__main__':
    main()

