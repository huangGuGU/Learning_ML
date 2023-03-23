from sklearn.ensemble import RandomForestRegressor

import numpy as np
from sklearn.model_selection import GridSearchCV


def My_Model(target, labels):
    forest_reg = RandomForestRegressor(random_state=42)
    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                               scoring='neg_mean_squared_error',
                               return_train_score=True)
    grid_search.fit(target, labels) # 用训练集找出最佳模型
    print(grid_search.best_params_)
    print(grid_search.best_estimator_)



    return grid_search.best_estimator_
