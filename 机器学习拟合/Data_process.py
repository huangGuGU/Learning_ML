from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit  # 引入的库可以进行分层抽样
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


class Data_Get:
    def __init__(self, data):
        self.data = data

    def split_train_test(self):
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

        for train_index, test_index in split.split(self.data, self.data["income_cat"]): # 使用中位数属性来分层抽样
            strat_train_set = self.data.loc[train_index]
            strat_test_set = self.data.loc[test_index]

        housing = strat_train_set.drop("median_house_value", axis=1)  # drop labels for training set
        housing_labels = strat_train_set["median_house_value"].copy()
        X_test = strat_test_set.drop("median_house_value", axis=1)
        y_test = strat_test_set["median_house_value"].copy()
        return housing, housing_labels, X_test, y_test


# 自定义转换器
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.rooms_ix = 3
        self.bedrooms_ix = 4
        self.population_ix = 5
        self.households_ix = 6

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        rooms_per_household = X[:, self.rooms_ix] / X[:, self.households_ix]
        population_per_household = X[:, self.population_ix] / X[:, self.households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, self.bedrooms_ix] / X[:, self.rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


# 数字和object分开处理
class Data_process:
    def __init__(self, data):
        self.data = data

    def columnTransformer(self):
        data_num = self.data.drop("ocean_proximity", axis=1)
        num_attribs = list(data_num)
        cat_attribs = ["ocean_proximity"]

        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler()),
        ])

        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])

        housing_prepared = full_pipeline.fit_transform(self.data)
        return housing_prepared, full_pipeline
