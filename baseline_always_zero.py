import sys
from functools import partial
import pandas as pd
from scipy import stats
from catboost import CatBoostRegressor as reg

def load_tasks(tasks_filename):
    return pd.read_csv(tasks_filename, sep="\t")

def main():
    tests_filename = sys.argv[1]
    tasks = load_tasks(tests_filename)
    pub_count = [len(tasks.publishers[i].split(',')) for i in range(len(tasks))]

    tasks = tasks[['cpm','hour_start','hour_end','audience_size']]
    tasks['delta_hour'] = tasks['hour_end'] - tasks['hour_start']
    tasks['cpm*hour'] = tasks['cpm']*tasks['delta_hour']
    tasks['cpm*pub'] = tasks['cpm']*pd.Series(pub_count)


    tasks = tasks[['hour_start','audience_size','delta_hour','cpm*hour','cpm*pub']]

    model = reg().load_model('model.cbm')
    pd.DataFrame(data = abs(model.predict(tasks)),columns = [['at_least_one', 'at_least_two', 'at_least_three']]).to_csv(sys.stdout, sep="\t", index=False, header=True)

if __name__ == '__main__':
    main()
