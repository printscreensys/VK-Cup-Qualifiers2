# VK-Cup-Qualifiers2
1.baseline\
2.WORKS\
  Dockerfile: `RUN pip install catboost`\
  Python: `from catboost import CatBoostRegressor as CatBoostRegressor`\
          `tasks = tasks[['cpm','hour_start','hour_end','audience_size']]`\
3.WORKS\
  Python: `model = reg().load_model('model.cbm')`\
  Deockerfile: `COPY model.cbm model.cbm`\
4.WORKS\
  Python: ...`at_least_one = 1/len(tasks)`\
5.WORKS\
  Python: `at_least_one=1/len(model.predict(tasks))`\
6.Python: WORKS\
  DELETED tasks = `tasks.assign(
            at_least_one = model.predict(tasks)[0][0],
            at_least_two=0.01,
            at_least_three=0.005,\
            )`\
