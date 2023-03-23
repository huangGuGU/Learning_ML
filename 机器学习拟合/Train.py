import os
import joblib
import pandas as pd
from Model import My_Model
from Data_process import *
from sklearn.metrics import mean_squared_error
from scipy import stats

'''get data'''
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
csv_path = os.path.join(HOUSING_PATH, "housing.csv")
data = pd.read_csv(csv_path)

data["income_cat"] = pd.cut(data["median_income"],
                            bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                            labels=[1, 2, 3, 4, 5])


'''data process'''
data_get = Data_Get(data)
housing, housing_labels, X_test, y_test = data_get.split_train_test()

data_process = Data_process(housing)
data_prepared, full_pipeline = data_process.columnTransformer()  # 数字通过上述缩放，object就热编码
X_test_prepared = full_pipeline.transform(X_test)  # 不要fit_transform


'''model process'''
model = My_Model(data_prepared, housing_labels)

predictions = model.predict(X_test_prepared)

print('prediction', predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print('rmse ', rmse)

squared_errors = (predictions - y_test) ** 2
result = np.sqrt(stats.t.interval(0.95, len(squared_errors) - 1,
                                  loc=squared_errors.mean(), scale=stats.sem(squared_errors)))
print(result)





full_pipeline_with_predictor = Pipeline([
        ("preparation", full_pipeline),
        ("Mode", My_Model(data_prepared, housing_labels))
    ])

full_pipeline_with_predictor.fit(housing, housing_labels)
full_pipeline_with_predictor.predict(X_test_prepared)



'''save model'''
def save_model():
    my_model = full_pipeline_with_predictor
    joblib.dump(my_model, "weight/my_model.pkl")  # DIFF
    my_model_loaded = joblib.load("weight/my_model.pkl")  # DIFF



eval()
save_model()